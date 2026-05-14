# -*- coding: utf-8 -*-
"""
model/patch_matching.py

Local patch correspondence for VSCDNet.

Given query and reference patch-token maps on the same 64 x 64 grid, this
module computes a local correlation volume within a k x k neighborhood,
converts it into a patch-matching distribution, estimates an expected
2D displacement, and warps the reference feature map toward the query.

Inputs for a pair:
    Eq_map: [B, D, H, W]
    Er_map: [B, D, H, W]

Outputs:
    Er_w:    [B, D, H, W]
    P_patch: [B, k*k, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchMatchConfig:
    patch_hw: Tuple[int, int] = (64, 64)
    local_k: int = 5
    temperature: float = 1.0

    # Two equivalent correlation backends.
    # True  : faster unfold-based implementation with higher peak memory.
    # False : lower-memory shift-dot implementation.
    use_unfold: bool = True


class LocalPatchMatcher(nn.Module):
    """Local k x k patch matcher on ViT token grids."""

    def __init__(self, cfg: PatchMatchConfig):
        super().__init__()
        if len(cfg.patch_hw) != 2:
            raise ValueError(f"patch_hw must be a tuple of length 2, got {cfg.patch_hw}")
        if cfg.patch_hw[0] <= 0 or cfg.patch_hw[1] <= 0:
            raise ValueError(f"patch_hw must be positive, got {cfg.patch_hw}")
        if cfg.local_k <= 0 or cfg.local_k % 2 != 1:
            raise ValueError(f"local_k must be a positive odd integer, got {cfg.local_k}")
        if cfg.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {cfg.temperature}")

        self.cfg = cfg
        self.H, self.W = int(cfg.patch_hw[0]), int(cfg.patch_hw[1])
        self.K = int(cfg.local_k)
        self.r = self.K // 2

        offsets = []
        for dy in range(-self.r, self.r + 1):
            for dx in range(-self.r, self.r + 1):
                offsets.append((dx, dy))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)

    @staticmethod
    def tokens_to_map(tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """
        Convert flattened patch tokens into a feature map.

        Args:
            tokens: [B, N, D], where N = H * W.
            hw: spatial token-grid size (H, W).

        Returns:
            Feature map [B, D, H, W].
        """
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        if len(hw) != 2:
            raise ValueError(f"hw must be a tuple of length 2, got {hw}")

        B, N, D = tokens.shape
        H, W = int(hw[0]), int(hw[1])
        if N != H * W:
            raise ValueError(f"Token count mismatch: N={N}, H*W={H * W} for hw={hw}")

        return tokens.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

    def _check_feature_maps(self, q: torch.Tensor, r: torch.Tensor) -> None:
        if q.dim() != 4 or r.dim() != 4:
            raise ValueError(f"q and r must be [B,D,H,W], got q={tuple(q.shape)}, r={tuple(r.shape)}")
        if q.shape != r.shape:
            raise ValueError(f"q and r must have identical shape, got q={tuple(q.shape)}, r={tuple(r.shape)}")
        if q.shape[-2:] != (self.H, self.W):
            raise ValueError(
                f"Feature map size mismatch: got {tuple(q.shape[-2:])}, expected {(self.H, self.W)}"
            )

    def local_corr(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Compute local dot-product correlation.

        Args:
            q: [B, D, H, W] query feature map.
            r: [B, D, H, W] reference feature map.

        Returns:
            Correlation volume [B, k*k, H, W].
        """
        self._check_feature_maps(q, r)
        if bool(self.cfg.use_unfold):
            return self._local_corr_unfold(q, r)
        return self._local_corr_shift(q, r)

    def _local_corr_unfold(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Fast im2col implementation of local correlation."""
        B, D, H, W = q.shape
        r_pad = F.pad(r, (self.r, self.r, self.r, self.r), mode="constant", value=0.0)

        neigh = F.unfold(r_pad, kernel_size=self.K, padding=0, stride=1)
        neigh = neigh.view(B, D, self.K * self.K, H * W)
        q_flat = q.view(B, D, H * W).unsqueeze(2)

        corr = (neigh * q_flat).sum(dim=1)
        return corr.view(B, self.K * self.K, H, W)

    def _local_corr_shift(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Lower-memory shift-dot implementation of local correlation."""
        _, _, H, W = q.shape
        r_pad = F.pad(r, (self.r, self.r, self.r, self.r), mode="constant", value=0.0)

        corrs = []
        for dy in range(-self.r, self.r + 1):
            y0 = self.r + dy
            y1 = y0 + H
            for dx in range(-self.r, self.r + 1):
                x0 = self.r + dx
                x1 = x0 + W
                r_shift = r_pad[:, :, y0:y1, x0:x1]
                corrs.append((q * r_shift).sum(dim=1, keepdim=True))

        return torch.cat(corrs, dim=1).contiguous()

    def soft_argmax_disp(self, P_patch: torch.Tensor) -> torch.Tensor:
        """
        Estimate expected 2D displacement from a patch-matching distribution.

        Args:
            P_patch: [B, k*k, H, W], softmax-normalized over k*k offsets.

        Returns:
            Displacement field [B, 2, H, W] in token-grid pixel units, ordered as
            (dx, dy).
        """
        if P_patch.dim() != 4:
            raise ValueError(f"P_patch must be [B,KK,H,W], got {tuple(P_patch.shape)}")

        B, KK, H, W = P_patch.shape
        expected_kk = self.K * self.K
        if KK != expected_kk:
            raise ValueError(f"P_patch offset dimension must be {expected_kk}, got {KK}")
        if (H, W) != (self.H, self.W):
            raise ValueError(f"P_patch spatial size must be {(self.H, self.W)}, got {(H, W)}")

        offsets = self.offsets.to(device=P_patch.device, dtype=P_patch.dtype).view(1, KK, 2, 1, 1)
        return (P_patch.unsqueeze(2) * offsets).sum(dim=1)

    def warp_ref(self, r: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Warp the reference feature map using a displacement field.

        Args:
            r: [B, D, H, W] reference feature map.
            disp: [B, 2, H, W] displacement field in token-grid pixel units.

        Returns:
            Warped reference feature map [B, D, H, W].
        """
        if r.dim() != 4:
            raise ValueError(f"r must be [B,D,H,W], got {tuple(r.shape)}")
        if disp.dim() != 4 or disp.size(1) != 2:
            raise ValueError(f"disp must be [B,2,H,W], got {tuple(disp.shape)}")
        if disp.shape[0] != r.shape[0] or disp.shape[-2:] != r.shape[-2:]:
            raise ValueError(f"disp shape {tuple(disp.shape)} is incompatible with r shape {tuple(r.shape)}")

        B, _, H, W = r.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=r.device, dtype=r.dtype),
            torch.arange(W, device=r.device, dtype=r.dtype),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        coords = base + disp.to(dtype=r.dtype)

        grid_x = coords[:, 0] / max(W - 1, 1) * 2 - 1
        grid_y = coords[:, 1] / max(H - 1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)

        return F.grid_sample(r, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward_pair(self, Eq_map: torch.Tensor, Er_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match and warp one query-reference feature-map pair.

        Args:
            Eq_map: [B, D, H, W] query feature map.
            Er_map: [B, D, H, W] reference feature map.

        Returns:
            Er_w: [B, D, H, W], warped reference feature map.
            P_patch: [B, k*k, H, W], patch-matching distribution.
        """
        corr = self.local_corr(Eq_map, Er_map) / float(self.cfg.temperature)
        P_patch = F.softmax(corr, dim=1)
        disp = self.soft_argmax_disp(P_patch)
        Er_w = self.warp_ref(Er_map, disp)
        return Er_w, P_patch
