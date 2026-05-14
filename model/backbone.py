# -*- coding: utf-8 -*-
"""
SAM ViT backbone for VSCD.

This module implements the backbone used in VSCDNet:
- load a SAM image encoder;
- keep the SAM image encoder frozen;
- extract patch tokens on the SAM token grid;
- build a frame-level descriptor from average pooled patch tokens;
- optionally add an Alignment Token (AT) descriptor for frame-level matching.

Outputs:
    frame_vecs:   [B, T, D]
    patch_tokens: [B, T, N, D]
    patch_hw:     (H_p, W_p), typically (64, 64) for 1024x1024 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BackboneConfig:
    """Configuration for the SAM backbone used by VSCDNet.

    The public implementation keeps only the paper-relevant behavior: frozen SAM
    features with optional Alignment Token (AT). Some legacy fields are kept as
    no-op arguments so that the existing train/eval scripts do not break while
    they are being cleaned up file-by-file.
    """

    # SAM image encoder
    model_type: str = "vit_b"  # vit_b | vit_l | vit_h
    checkpoint: str = ""
    sam_root: Optional[str] = None

    # Paper setting: frozen SAM image encoder and trunk tokens.
    use_trunk_tokens: bool = True
    freeze_image_encoder: bool = True

    # Frame descriptor head.
    use_at: bool = True
    mixer_layers: int = 1
    mixer_heads: int = 8
    mixer_mlp_ratio: float = 4.0
    mixer_dropout: float = 0.0

    # Forward frames in chunks over the flattened B*T dimension.
    forward_chunk_size: int = 8

    # Legacy no-op fields kept temporarily for compatibility with the current
    # train.py/eval.py. They should be removed from the callers later.
    finetune_last_n_blocks: int = 0
    use_vpt: bool = False
    prompt_len: int = 8
    use_dora: bool = False
    dora_rank: int = 8
    dora_alpha: float = 16.0
    use_llrd: bool = False
    llrd_layer_decay: float = 0.75
    llrd_mixer_lr_mult: float = 1.0
    llrd_prompt_lr_mult: float = 1.0
    llrd_dora_lr_mult: float = 1.0
    use_grad_checkpointing: bool = False


def _vit_dim(model_type: str) -> int:
    model_type = model_type.lower()
    if model_type == "vit_b":
        return 768
    if model_type == "vit_l":
        return 1024
    if model_type == "vit_h":
        return 1280
    raise ValueError(f"Unsupported model_type={model_type}. Expected one of: vit_b, vit_l, vit_h.")


def _import_sam_registry(sam_root: Optional[str]):
    """Import SAM's model registry from either a repo root or package path."""
    import importlib
    import os
    import sys

    if sam_root is None or str(sam_root).strip() == "":
        raise ImportError("BackboneConfig.sam_root must point to the Segment Anything repository or package path.")

    sam_root = os.path.abspath(str(sam_root))
    candidate_paths = [sam_root, os.path.join(sam_root, "segment_anything")]
    for path in candidate_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    errors = []
    for module_name in ("segment_anything.build_sam", "build_sam"):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "sam_model_registry")
        except Exception as exc:  # pragma: no cover - error message only
            errors.append(f"{module_name}: {repr(exc)}")

    raise ImportError("Failed to import SAM model registry. Tried: " + " | ".join(errors))


class CrossTokenMixer(nn.Module):
    """Update a single Alignment Token by attending to patch tokens.

    The mixer uses a tiny query stream: the Alignment Token first goes through
    self-attention and then cross-attends to all patch tokens. This keeps the
    frame descriptor trainable while the SAM image encoder remains frozen.
    """

    def __init__(self, dim: int, layers: int = 1, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if layers < 1:
            raise ValueError("mixer_layers must be >= 1 when use_at=True.")
        self.layers = nn.ModuleList(
            [_QSelfAndCrossBlock(dim, heads, mlp_ratio, dropout) for _ in range(int(layers))]
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tokens:  [B, Q, D], usually Q=1 for the Alignment Token.
            kv_tokens: [B, N, D], SAM patch tokens.

        Returns:
            Updated Alignment Token descriptor [B, D].
        """
        x = q_tokens
        for block in self.layers:
            x = block(x, kv_tokens)
        return x[:, 0, :]


class _QSelfAndCrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )

        hidden_dim = int(dim * float(mlp_ratio))
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q_self(q)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        x = q + self_out

        q_cross = self.norm_q(x)
        kv_cross = self.norm_kv(kv)
        cross_out, _ = self.cross_attn(q_cross, kv_cross, kv_cross, need_weights=False)
        x = x + cross_out

        return x + self.ff(self.norm_ff(x))


class FrameBackboneSAM1(nn.Module):
    """Frozen SAM image encoder with an optional Alignment Token head.

    Args:
        frames: [B, T, 3, H, W]

    Returns:
        frame_vecs:   [B, T, D]
        patch_tokens: [B, T, N, D]
        patch_hw:     (H_p, W_p)
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        if not cfg.checkpoint:
            raise ValueError("BackboneConfig.checkpoint is required.")
        if not cfg.use_trunk_tokens:
            raise ValueError("VSCDNet expects SAM trunk tokens; use_trunk_tokens must be True.")

        self.cfg = cfg
        self.dim = _vit_dim(cfg.model_type)
        self.use_at = bool(cfg.use_at)

        sam_model_registry = _import_sam_registry(cfg.sam_root)
        sam = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint)
        self.encoder = sam.image_encoder
        self._freeze_encoder()

        if self.use_at:
            self.at = nn.Parameter(torch.zeros(1, 1, self.dim))
            nn.init.trunc_normal_(self.at, std=0.02)
            self.mixer = CrossTokenMixer(
                dim=self.dim,
                layers=int(cfg.mixer_layers),
                heads=int(cfg.mixer_heads),
                mlp_ratio=float(cfg.mixer_mlp_ratio),
                dropout=float(cfg.mixer_dropout),
            )

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def _forward_trunk_tokens_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Run the SAM image encoder and return patch tokens before the neck."""
        encoder = self.encoder

        xt = encoder.patch_embed(x)  # [B, H_p, W_p, D]

        if hasattr(encoder, "pos_embed") and encoder.pos_embed is not None and encoder.pos_embed.dim() == 4:
            if encoder.pos_embed.shape[1:3] == xt.shape[1:3]:
                xt = xt + encoder.pos_embed

        if hasattr(encoder, "blocks"):
            for block in encoder.blocks:
                xt = block(xt)

        if hasattr(encoder, "norm") and encoder.norm is not None:
            xt = encoder.norm(xt)

        B, Hp, Wp, D = xt.shape
        tokens = xt.view(B, Hp * Wp, D).contiguous()
        return tokens, (Hp, Wp)

    def _forward_trunk_tokens_chunked(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        chunk_size = int(getattr(self.cfg, "forward_chunk_size", 0) or 0)
        if chunk_size <= 0 or x.shape[0] <= chunk_size:
            return self._forward_trunk_tokens_impl(x)

        tokens_list = []
        patch_hw = None
        for start in range(0, x.shape[0], chunk_size):
            tokens_i, patch_hw_i = self._forward_trunk_tokens_impl(x[start : start + chunk_size])
            if patch_hw is None:
                patch_hw = patch_hw_i
            elif patch_hw != patch_hw_i:
                raise RuntimeError(f"Inconsistent patch grid size across chunks: {patch_hw} vs {patch_hw_i}.")
            tokens_list.append(tokens_i)

        if patch_hw is None:
            raise RuntimeError("No frames were processed by the SAM backbone.")
        return torch.cat(tokens_list, dim=0), patch_hw

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        if frames.dim() != 5:
            raise ValueError(f"frames must be [B,T,3,H,W], got {tuple(frames.shape)}")

        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)

        tokens, patch_hw = self._forward_trunk_tokens_chunked(x)
        sv = tokens.mean(dim=1)  # [B*T, D]

        if self.use_at:
            at = self.at.expand(tokens.shape[0], 1, self.dim)
            v_at = self.mixer(at, tokens)
            frame_vec = sv + v_at
        else:
            frame_vec = sv

        frame_vecs = frame_vec.view(B, T, -1)
        patch_tokens = tokens.view(B, T, tokens.shape[1], -1)
        return frame_vecs, patch_tokens, patch_hw
