# -*- coding: utf-8 -*-
"""
Training script for VSCDNet.

This public version keeps the paper-relevant training path only:
- sparse keyframe training with a fixed number of reference/query frames;
- frozen SAM image encoder with an optional Alignment Token;
- frame-level alignment, local patch matching, confidence-aware feature fusion,
  and query-guided decoding;
- BCE-with-logits + soft Dice loss;
- validation with frame-wise F1 at the output resolution.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import VSCDSequenceDataset
from dataset.transforms import build_transforms
from dataset.collate import vscd_collate_fn

from model.backbone import BackboneConfig, FrameBackboneSAM1
from model.alignment import FrameAlignment
from model.patch_matching import LocalPatchMatcher, PatchMatchConfig
from model.decoder import ChangeMaskDecoder, ChangeDecoderConfig
from model.fusion import FusionConfig, patch_confidence

from loss import build_criterion


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _vit_dim(model_type: str) -> int:
    if model_type == "vit_b":
        return 768
    if model_type == "vit_l":
        return 1024
    if model_type == "vit_h":
        return 1280
    raise ValueError(f"Unsupported model_type={model_type}")


@dataclass
class TrainConfig:
    # Paths
    data_root: str
    sam_root: str
    sam_ckpt: str
    out_dir: str = "./runs/vscd"

    # Data
    split_train: str = "train"
    split_val: str = "test"
    img_size: int = 1024
    batch_size: int = 1
    num_workers: int = 0
    num_frames_fixed: int = 32

    # Backbone
    model_type: str = "vit_b"
    backbone_chunk: int = 2
    use_at: bool = True
    mixer_layers: int = 1
    mixer_heads: int = 8

    # Frame-level alignment
    max_msp_len: int = 5
    diag_delta: int = 2
    min_msp_len: int = 2
    logit_tau: float = -1e9
    softmax_temp: float = 0.5

    # Patch matching / fusion
    patch_hw: int = 64
    local_k: int = 5
    use_unfold: bool = True
    cp: float = 1.0
    c: float = 1.0

    # Candidate selection
    topk_ref_per_t: int = 4
    max_ref_cands_per_t: int = 6

    # Module ablation switches
    use_cf: bool = True
    use_csp: bool = True

    # Optimization
    device: str = "cuda"
    seed: int = 0
    epochs: int = 80
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = False

    # Checkpointing / validation
    save_every: int = 1
    save_best_last_only: bool = True
    val_thr: float = 0.5
    resume_ckpt: str = ""
    resume_optimizer: bool = False
    resume_epoch: bool = False


class VSCDSystem(nn.Module):
    """VSCDNet training model."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        dim = _vit_dim(cfg.model_type)

        backbone_cfg = BackboneConfig(
            model_type=cfg.model_type,
            checkpoint=cfg.sam_ckpt,
            sam_root=cfg.sam_root,
            use_trunk_tokens=True,
            freeze_image_encoder=True,
            use_at=cfg.use_at,
            mixer_layers=cfg.mixer_layers,
            mixer_heads=cfg.mixer_heads,
            forward_chunk_size=cfg.backbone_chunk,
        )
        self.backbone = FrameBackboneSAM1(backbone_cfg)

        self.alignment = FrameAlignment(
            logit_tau=cfg.logit_tau,
            diag_delta=cfg.diag_delta,
            min_len=cfg.min_msp_len,
            max_len=cfg.max_msp_len,
            ensure_full_coverage=True,
            softmax_temp=cfg.softmax_temp,
        )

        self.patch_hw = (cfg.patch_hw, cfg.patch_hw)
        self.matcher = LocalPatchMatcher(
            PatchMatchConfig(
                patch_hw=self.patch_hw,
                local_k=cfg.local_k,
                temperature=1.0,
                use_unfold=cfg.use_unfold,
            )
        )
        self.decoder = ChangeMaskDecoder(ChangeDecoderConfig(in_dim=dim, patch_hw=self.patch_hw))
        self.fusion_cfg = FusionConfig(cp=cfg.cp, c=cfg.c)

        self.topk_ref_per_t = int(cfg.topk_ref_per_t)
        self.max_ref_cands_per_t = int(cfg.max_ref_cands_per_t)
        self.use_cf = bool(cfg.use_cf)
        self.use_csp = bool(cfg.use_csp)

    def _select_reference_candidates(
        self,
        P_frame: torch.Tensor,
        msps_b: List[Dict[str, Any]],
        aux: Dict[str, Any],
        b: int,
        t: int,
    ) -> List[int]:
        """Select one MSP-consistent anchor and fill remaining slots by P_frame."""
        _, _, Tr = P_frame.shape

        msp_set: List[int] = []
        for m in msps_b:
            if int(m["t0"]) <= t <= int(m["t1"]):
                s0, s1 = int(m["s0"]), int(m["s1"])
                lo, hi = (s0, s1) if s0 <= s1 else (s1, s0)
                msp_set.extend(range(lo, hi + 1))
        msp_set = sorted(set(msp_set))

        candidates: List[int] = []
        if msp_set:
            msp_idx = torch.tensor(msp_set, device=P_frame.device, dtype=torch.long)
            best_local = int(torch.argmax(P_frame[b, t, msp_idx]).item())
            candidates.append(int(msp_set[best_local]))

        k_lookup = min(max(self.topk_ref_per_t, self.max_ref_cands_per_t), Tr)
        topk = torch.topk(P_frame[b, t], k=k_lookup, dim=-1).indices.tolist()
        for s in topk:
            s = int(s)
            if s not in candidates:
                candidates.append(s)
            if len(candidates) >= self.max_ref_cands_per_t:
                break

        if not candidates:
            candidates = [int(aux["top1_s"][b, t].item())]

        return candidates

    def forward(
        self,
        ref_frames: torch.Tensor,
        qry_frames: torch.Tensor,
        ref_valid: Optional[torch.Tensor] = None,
        qry_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        vr_vec, Er_tok, ref_patch_hw = self.backbone(ref_frames)
        vq_vec, Eq_tok, qry_patch_hw = self.backbone(qry_frames)

        if ref_patch_hw != self.patch_hw or qry_patch_hw != self.patch_hw:
            raise RuntimeError(
                f"Patch grid mismatch: ref={ref_patch_hw}, query={qry_patch_hw}, expected={self.patch_hw}."
            )

        P_frame, msps, aux = self.alignment(vq_vec, vr_vec, qry_valid=qry_valid, ref_valid=ref_valid)
        B, Tq, _ = P_frame.shape

        logits_per_t: List[torch.Tensor] = []
        for t in range(Tq):
            logits_per_b: List[torch.Tensor] = []
            for b in range(B):
                if qry_valid is not None and not bool(qry_valid[b, t].item()):
                    logits_per_b.append(
                        torch.zeros(1, 1, 1024, 1024, device=qry_frames.device, dtype=P_frame.dtype)
                    )
                    continue

                ref_candidates = self._select_reference_candidates(P_frame, msps[b], aux, b, t)

                Eq_bt = Eq_tok[b, t].unsqueeze(0)
                Eq_map = self.matcher.tokens_to_map(Eq_bt, self.patch_hw)
                qry_rgb_bt = qry_frames[b, t].unsqueeze(0)

                numerator = None
                denominator = None

                for s in ref_candidates:
                    if ref_valid is not None and not bool(ref_valid[b, s].item()):
                        continue

                    Er_bs = Er_tok[b, s].unsqueeze(0)
                    Er_map = self.matcher.tokens_to_map(Er_bs, self.patch_hw)
                    Er_w, P_patch = self.matcher.forward_pair(Eq_map, Er_map)
                    feat_ts = self.decoder.make_feat(Eq_map, Er_w)

                    if self.use_cf:
                        Cf = P_frame[b, t, s].view(1, 1, 1, 1)
                    else:
                        Cf = torch.ones((1, 1, 1, 1), device=feat_ts.device, dtype=feat_ts.dtype)

                    if self.use_csp:
                        Csp = patch_confidence(P_patch, self.fusion_cfg)
                    else:
                        Csp = torch.ones(
                            (1, 1, self.patch_hw[0], self.patch_hw[1]),
                            device=feat_ts.device,
                            dtype=feat_ts.dtype,
                        )

                    weight = Cf * Csp
                    if numerator is None:
                        numerator = weight * feat_ts
                        denominator = weight
                    else:
                        numerator = numerator + weight * feat_ts
                        denominator = denominator + weight

                if numerator is None or denominator is None:
                    channels = self.decoder.cfg.fuse_ch
                    numerator = torch.zeros(
                        (1, channels, self.patch_hw[0], self.patch_hw[1]),
                        device=qry_frames.device,
                        dtype=qry_frames.dtype,
                    )
                    denominator = torch.ones(
                        (1, 1, self.patch_hw[0], self.patch_hw[1]),
                        device=qry_frames.device,
                        dtype=qry_frames.dtype,
                    )

                fused_feat = numerator / (denominator + 1e-6)
                logits = self.decoder.decode_1024(fused_feat, qry_rgb_bt)
                logits_per_b.append(logits)

            logits_per_t.append(torch.cat(logits_per_b, dim=0))

        return {
            "Mfuse_logits": torch.stack(logits_per_t, dim=1),
            "P_frame": P_frame,
            "msps": msps,
            "aux": aux,
        }


def build_loader(cfg: TrainConfig, split: str, is_train: bool) -> DataLoader:
    transform_cfg = SimpleNamespace(
        img_size=cfg.img_size,
        already_normalized=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        hflip_p=0.0,
        vflip_p=0.0,
        rot90_p=0.0,
        color_jitter_p=0.0,
        jitter_brightness=0.0,
        jitter_contrast=0.0,
        jitter_saturation=0.0,
        jitter_same_for_ref_qry=False,
    )
    transforms = build_transforms(is_train=False, cfg=transform_cfg)

    dataset = VSCDSequenceDataset(
        root=cfg.data_root,
        split=split,
        num_frames_fixed=cfg.num_frames_fixed,
        img_wh=(cfg.img_size, cfg.img_size),
        transforms=transforms,
        return_full_query=False,
        drop_query_without_mask=True,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=is_train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=vscd_collate_fn,
    )


def save_ckpt(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict()}, path)


def load_ckpt(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    if optimizer is not None and isinstance(ckpt, dict) and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return epoch


def build_optimizer(model: VSCDSystem, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_one_epoch(
    model: VSCDSystem,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        ref_frames = batch["ref_frames"].to(device, non_blocking=True)
        qry_frames = batch["qry_frames"].to(device, non_blocking=True)
        ref_valid = batch["ref_valid"].to(device, non_blocking=True)
        qry_valid = batch["qry_valid"].to(device, non_blocking=True)
        targets = {
            "qry_masks": batch["qry_masks"].to(device, non_blocking=True),
            "qry_valid": qry_valid,
        }

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            outputs = model(ref_frames, qry_frames, ref_valid=ref_valid, qry_valid=qry_valid)
            loss = criterion(outputs, targets)["loss_total"]

        scaler.scale(loss).backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        num_batches += 1
        pbar.set_postfix(loss=total_loss / max(num_batches, 1))

    return {"loss_total": total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate_one_epoch(
    model: VSCDSystem,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    num_batches = 0
    eps = 1e-6

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        ref_frames = batch["ref_frames"].to(device, non_blocking=True)
        qry_frames = batch["qry_frames"].to(device, non_blocking=True)
        ref_valid = batch["ref_valid"].to(device, non_blocking=True)
        qry_valid = batch["qry_valid"].to(device, non_blocking=True)
        gt = batch["qry_masks"].to(device, non_blocking=True)
        targets = {"qry_masks": gt, "qry_valid": qry_valid}

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            outputs = model(ref_frames, qry_frames, ref_valid=ref_valid, qry_valid=qry_valid)
            loss = criterion(outputs, targets)["loss_total"]

        pred = torch.sigmoid(outputs["Mfuse_logits"]) >= float(cfg.val_thr)
        gt_bin = gt >= 0.5
        valid = qry_valid.bool()
        B, T = valid.shape

        pred_f = pred.view(B * T, -1)
        gt_f = gt_bin.view(B * T, -1)
        valid_f = valid.view(B * T)

        if valid_f.any():
            pred_f = pred_f[valid_f]
            gt_f = gt_f[valid_f]
            tp = (pred_f & gt_f).sum(dim=1).float()
            fp = (pred_f & ~gt_f).sum(dim=1).float()
            fn = (~pred_f & gt_f).sum(dim=1).float()
            f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            f1_value = float(f1.mean().item())
        else:
            f1_value = 0.0

        total_loss += float(loss.item())
        total_f1 += f1_value
        num_batches += 1
        pbar.set_postfix(loss=total_loss / max(num_batches, 1), f1=total_f1 / max(num_batches, 1))

    return {
        "loss_total": total_loss / max(num_batches, 1),
        "f1": total_f1 / max(num_batches, 1),
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train VSCDNet.")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--sam_root", type=str, required=True)
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./runs/vscd")

    parser.add_argument("--split_train", type=str, default="train")
    parser.add_argument("--split_val", type=str, default="test")
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_frames_fixed", type=int, default=32)

    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--backbone_chunk", type=int, default=2)
    parser.add_argument("--use_at", action="store_true", help="Deprecated: AT is enabled by default.")
    parser.add_argument("--no_at", action="store_true", help="Disable the Alignment Token ablation.")
    parser.add_argument("--mixer_layers", type=int, default=1)
    parser.add_argument("--mixer_heads", type=int, default=8)

    parser.add_argument("--max_msp_len", type=int, default=5)
    parser.add_argument("--diag_delta", type=int, default=2)
    parser.add_argument("--min_msp_len", type=int, default=2)
    parser.add_argument("--logit_tau", type=float, default=-1e9)
    parser.add_argument("--softmax_temp", type=float, default=0.5)

    parser.add_argument("--patch_hw", type=int, default=64)
    parser.add_argument("--local_k", type=int, default=5)
    parser.add_argument("--cp", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--topk_ref_per_t", type=int, default=4)
    parser.add_argument("--max_ref_cands_per_t", type=int, default=6)
    parser.add_argument("--no_unfold", action="store_true", help="Use a lower-memory local-correlation backend.")

    parser.add_argument("--no_cf", action="store_true", help="Disable frame-level confidence Cf.")
    parser.add_argument("--no_csp", action="store_true", help="Disable patch-level confidence Csp.")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_best_last_only", action="store_true", default=True)
    parser.add_argument("--save_all_epochs", action="store_true", help="Save periodic epoch snapshots as well.")
    parser.add_argument("--val_thr", type=float, default=0.5)

    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--resume_epoch", action="store_true")

    # Deprecated no-op arguments kept so older local shell scripts fail less often.
    parser.add_argument("--no_vpt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use_vpt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prompt_len", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--frame_stride", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument("--use_unfold", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root,
        sam_root=args.sam_root,
        sam_ckpt=args.sam_ckpt,
        out_dir=args.out_dir,
        split_train=args.split_train,
        split_val=args.split_val,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames_fixed=args.num_frames_fixed,
        model_type=args.model_type,
        backbone_chunk=args.backbone_chunk,
        use_at=not args.no_at,
        mixer_layers=args.mixer_layers,
        mixer_heads=args.mixer_heads,
        max_msp_len=args.max_msp_len,
        diag_delta=args.diag_delta,
        min_msp_len=args.min_msp_len,
        logit_tau=args.logit_tau,
        softmax_temp=args.softmax_temp,
        patch_hw=args.patch_hw,
        local_k=args.local_k,
        cp=args.cp,
        c=args.c,
        topk_ref_per_t=args.topk_ref_per_t,
        max_ref_cands_per_t=args.max_ref_cands_per_t,
        use_unfold=not args.no_unfold,
        use_cf=not args.no_cf,
        use_csp=not args.no_csp,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=args.amp,
        save_every=args.save_every,
        save_best_last_only=(not args.save_all_epochs),
        val_thr=args.val_thr,
        resume_ckpt=args.resume_ckpt,
        resume_optimizer=args.resume_optimizer,
        resume_epoch=args.resume_epoch,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_loader = build_loader(cfg, cfg.split_train, is_train=True)
    val_loader = build_loader(cfg, cfg.split_val, is_train=False)

    model = VSCDSystem(cfg).to(device)
    criterion = build_criterion(
        bce_weight=1.0,
        dice_weight=1.0,
        downsample_gt_to_pred=False,
        use_qry_valid_mask=True,
    ).to(device)
    optimizer = build_optimizer(model, cfg)

    start_epoch = 0
    if cfg.resume_ckpt:
        last_epoch = load_ckpt(cfg.resume_ckpt, model, optimizer if cfg.resume_optimizer else None)
        if cfg.resume_epoch and last_epoch >= 0:
            start_epoch = last_epoch + 1
        print(f"[INFO] Resumed from {cfg.resume_ckpt} (start_epoch={start_epoch})")

    print(f"[INFO] Train iters={len(train_loader)}, Val iters={len(val_loader)}")
    print(
        "[INFO] "
        f"device={device}, amp={cfg.amp}, T_key={cfg.num_frames_fixed}, "
        f"K={cfg.topk_ref_per_t}, max_ref={cfg.max_ref_cands_per_t}, "
        f"k={cfg.local_k}, Lmax={cfg.max_msp_len}, AT={cfg.use_at}, "
        f"Cf={cfg.use_cf}, Csp={cfg.use_csp}"
    )

    best_f1 = -1.0
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        final_epoch = epoch
        start_time = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg)
        val_stats = validate_one_epoch(model, val_loader, criterion, device, cfg)
        elapsed = time.time() - start_time

        val_loss = float(val_stats["loss_total"])
        val_f1 = float(val_stats["f1"])
        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_stats['loss_total']:.6f} "
            f"val_loss={val_loss:.6f} val_f1={val_f1:.6f} time={elapsed:.1f}s"
        )

        save_ckpt(os.path.join(cfg.out_dir, "ckpt_current.pt"), model, optimizer, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(cfg.out_dir, "ckpt_best.pt")
            save_ckpt(save_path, model, optimizer, epoch)
            print(f"[INFO] New best val_f1={best_f1:.6f} -> {save_path}")

        if not cfg.save_best_last_only and (epoch + 1) % cfg.save_every == 0:
            save_path = os.path.join(cfg.out_dir, f"ckpt_epoch_{epoch:03d}.pt")
            save_ckpt(save_path, model, optimizer, epoch)
            print(f"[INFO] Saved epoch checkpoint: {save_path}")

    if final_epoch >= 0:
        save_path = os.path.join(cfg.out_dir, "ckpt_last.pt")
        save_ckpt(save_path, model, optimizer, final_epoch)
        print(f"[INFO] Saved last checkpoint: {save_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
