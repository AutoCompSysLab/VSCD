# -*- coding: utf-8 -*-
"""
Evaluation script for VSCDNet.

This public version keeps the paper-relevant evaluation path only:
- fixed-keyframe evaluation on the VSCD Test split;
- frame-level alignment, local patch matching, confidence-aware feature fusion,
  and query-guided decoding;
- frame-wise IoU and F1 at the output resolution;
- optional saving of predicted masks for qualitative inspection.

Dense optical-flow propagation and internal diagnostic visualizations are not part
of the released evaluation path.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import VSCDSequenceDataset
from dataset.transforms import build_transforms
from dataset.collate import vscd_collate_fn

from model.backbone import BackboneConfig, FrameBackboneSAM1
from model.alignment import FrameAlignment
from model.patch_matching import LocalPatchMatcher, PatchMatchConfig
from model.decoder import ChangeDecoderConfig, ChangeMaskDecoder
from model.fusion import FusionConfig, patch_confidence


@dataclass
class EvalConfig:
    # Paths
    data_root: str
    sam_root: str
    sam_ckpt: str
    model_ckpt: str
    save_dir: str = ""

    # Data
    split: str = "test"
    img_size: int = 1024
    batch_size: int = 1
    num_workers: int = 0
    num_frames_fixed: int = 32
    spaces: Tuple[str, ...] = ()

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

    # Evaluation
    device: str = "cuda"
    amp: bool = False
    thr: float = 0.5
    max_samples: int = 0
    save_pred: bool = False
    save_limit: int = 5


def _vit_dim(model_type: str) -> int:
    if model_type == "vit_b":
        return 768
    if model_type == "vit_l":
        return 1024
    if model_type == "vit_h":
        return 1280
    raise ValueError(f"Unsupported model_type={model_type}")


class VSCDSystem(nn.Module):
    """VSCDNet inference model."""

    def __init__(self, cfg: EvalConfig):
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

    @torch.no_grad()
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


def build_loader(cfg: EvalConfig) -> DataLoader:
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
        split=cfg.split,
        num_frames_fixed=cfg.num_frames_fixed,
        img_wh=(cfg.img_size, cfg.img_size),
        transforms=transforms,
        return_full_query=False,
        drop_query_without_mask=True,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=vscd_collate_fn,
    )


def load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")


def compute_framewise_iou_f1(
    pred_bin: torch.Tensor,
    gt_bin: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[float, float, int]:
    """
    Args:
        pred_bin: [B, T, 1, H, W] bool
        gt_bin:   [B, T, 1, H, W] bool
        valid:    [B, T] bool, optional

    Returns:
        mean IoU, mean F1, number of valid frames.
    """
    B, T = pred_bin.shape[:2]
    pred_f = pred_bin.bool().view(B * T, -1)
    gt_f = gt_bin.bool().view(B * T, -1)

    if valid is None:
        valid_f = torch.ones(B * T, device=pred_bin.device, dtype=torch.bool)
    else:
        valid_f = valid.bool().view(B * T)

    if not bool(valid_f.any().item()):
        return 0.0, 0.0, 0

    pred_f = pred_f[valid_f]
    gt_f = gt_f[valid_f]

    tp = (pred_f & gt_f).sum(dim=1).float()
    fp = (pred_f & ~gt_f).sum(dim=1).float()
    fn = (~pred_f & gt_f).sum(dim=1).float()
    union = (pred_f | gt_f).sum(dim=1).float()

    iou = (tp + eps) / (union + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(iou.mean().item()), float(f1.mean().item()), int(valid_f.sum().item())


def save_mask_png(mask_hw: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask_hw.astype(np.uint8) * 255)).save(path)


def save_predictions(
    pred_bin: torch.Tensor,
    batch: Dict[str, Any],
    save_dir: str,
    save_limit: int,
) -> None:
    """Save predicted binary masks for the first few sampled query frames."""
    pred_cpu = pred_bin.detach().cpu().numpy().astype(np.uint8)
    batch_size, T = pred_cpu.shape[:2]

    for b in range(batch_size):
        meta = batch["meta"][b]
        space = str(meta["space"]).strip()
        pair = meta["pair"]
        a, q = int(pair[0]), int(pair[1])
        qids = batch.get("qry_frame_ids", [[] for _ in range(batch_size)])[b]

        out_dir = os.path.join(save_dir, "pred_masks", space, f"pair_{a}_to_{q}")
        for t in range(min(T, int(save_limit))):
            frame_id = str(qids[t]) if t < len(qids) else f"t{t:04d}"
            save_mask_png(pred_cpu[b, t, 0], os.path.join(out_dir, f"pred_{frame_id}.png"))


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate VSCDNet.")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--sam_root", type=str, required=True)
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_frames_fixed", type=int, default=32)
    parser.add_argument("--spaces", nargs="*", default=[])

    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--backbone_chunk", type=int, default=2)
    parser.add_argument("--use_at", action="store_true", help="Deprecated: AT is enabled by default.")
    parser.add_argument("--no_at", action="store_true", help="Disable Alignment Token for ablation checkpoints.")
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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means evaluate all samples after filtering.")
    parser.add_argument("--save_pred", action="store_true", help="Save predicted masks for qualitative inspection.")
    parser.add_argument("--save_limit", type=int, default=5, help="Number of sampled query frames to save per sample.")

    # Deprecated no-op arguments kept so older local scripts do not fail during cleanup.
    parser.add_argument("--save_sparse", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dense", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save_dense", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--flow_res", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--mode", type=str, default="bidir", help=argparse.SUPPRESS)
    parser.add_argument("--viz_matches", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--viz_warp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--viz_limit", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--diag", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--thr_sweep", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_vpt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use_vpt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prompt_len", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--chunk_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--use_unfold", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.chunk_size is not None:
        args.backbone_chunk = int(args.chunk_size)
    if args.save_sparse:
        args.save_pred = True
    if args.dense or args.save_dense or args.viz_warp:
        print("[WARN] Dense RAFT propagation and warp visualization were removed from the public eval path; ignoring those flags.")
    if args.viz_matches or args.diag or args.thr_sweep:
        print("[WARN] Internal diagnostics/visualizations were removed from the public eval path; ignoring those flags.")

    spaces = tuple(s.strip() for s in args.spaces if str(s).strip())

    return EvalConfig(
        data_root=args.data_root,
        sam_root=args.sam_root,
        sam_ckpt=args.sam_ckpt,
        model_ckpt=args.model_ckpt,
        save_dir=args.save_dir,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames_fixed=args.num_frames_fixed,
        spaces=spaces,
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
        amp=args.amp,
        thr=args.thr,
        max_samples=args.max_samples,
        save_pred=args.save_pred,
        save_limit=args.save_limit,
    )


@torch.no_grad()
def evaluate(model: VSCDSystem, loader: DataLoader, device: torch.device, cfg: EvalConfig) -> None:
    model.eval()

    per_pair: DefaultDict[Tuple[str, str], List[Tuple[float, float, int]]] = defaultdict(list)
    overall: List[Tuple[float, float, int]] = []

    evaluated = 0
    for batch in tqdm(loader, desc="eval"):
        space = str(batch["meta"][0]["space"]).strip()
        if cfg.spaces and space not in cfg.spaces:
            continue

        if cfg.max_samples > 0 and evaluated >= cfg.max_samples:
            break
        evaluated += 1

        pair = batch["meta"][0]["pair"]
        pair_key = f"{int(pair[0])}->{int(pair[1])}"

        ref_frames = batch["ref_frames"].to(device, non_blocking=True)
        qry_frames = batch["qry_frames"].to(device, non_blocking=True)
        ref_valid = batch["ref_valid"].to(device, non_blocking=True)
        qry_valid = batch["qry_valid"].to(device, non_blocking=True)
        gt = batch["qry_masks"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            outputs = model(ref_frames, qry_frames, ref_valid=ref_valid, qry_valid=qry_valid)

        pred = torch.sigmoid(outputs["Mfuse_logits"]) >= float(cfg.thr)
        gt_bin = gt >= 0.5
        iou, f1, n_valid = compute_framewise_iou_f1(pred, gt_bin, valid=qry_valid)

        per_pair[(space, pair_key)].append((iou, f1, n_valid))
        overall.append((iou, f1, n_valid))

        if cfg.save_dir and cfg.save_pred:
            save_predictions(pred, batch, cfg.save_dir, cfg.save_limit)

    if evaluated == 0:
        if cfg.spaces:
            print(f"[WARN] No samples were evaluated. Check --spaces: {list(cfg.spaces)}")
        else:
            print("[WARN] No samples were evaluated.")
        return

    print("\nPer-space / per-pair results")
    print("----------------------------")
    for (space, pair_key), values in sorted(per_pair.items(), key=lambda x: (x[0][0], x[0][1])):
        iou = float(np.mean([v[0] for v in values]))
        f1 = float(np.mean([v[1] for v in values]))
        frames = int(sum(v[2] for v in values))
        print(f"{space:40s} pair {pair_key:5s} | IoU {iou * 100:6.2f} | F1 {f1 * 100:6.2f} | frames {frames}")

    mean_iou = float(np.mean([v[0] for v in overall]))
    mean_f1 = float(np.mean([v[1] for v in overall]))
    total_frames = int(sum(v[2] for v in overall))
    print("\nOverall")
    print("-------")
    print(f"samples {len(overall)} | frames {total_frames} | IoU {mean_iou * 100:.2f} | F1 {mean_f1 * 100:.2f}")


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    loader = build_loader(cfg)
    model = VSCDSystem(cfg).to(device)
    load_checkpoint(model, cfg.model_ckpt)

    print(
        "[INFO] "
        f"split={cfg.split}, device={device}, amp={cfg.amp}, thr={cfg.thr}, "
        f"T_key={cfg.num_frames_fixed}, K={cfg.topk_ref_per_t}, "
        f"max_ref={cfg.max_ref_cands_per_t}, k={cfg.local_k}, "
        f"Lmax={cfg.max_msp_len}, AT={cfg.use_at}, Cf={cfg.use_cf}, Csp={cfg.use_csp}"
    )
    evaluate(model, loader, device, cfg)


if __name__ == "__main__":
    main()
