# -*- coding: utf-8 -*-
"""
loss.py

Loss functions for VSCDNet.

The model predicts query-frame change logits:
    outputs["Mfuse_logits"]: [B, Tq, 1, H, W]

The dataset provides query-frame binary masks:
    batch["qry_masks"]: [B, Tq, 1, H, W]
    batch["qry_valid"]: [B, Tq]  (optional)

The training objective follows the paper implementation:
    BCE-with-logits loss + soft Dice loss

No alignment supervision is used in this loss file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    eps: float = 1e-6
    downsample_gt_to_pred: bool = False
    use_qry_valid_mask: bool = True


def _downsample_mask_nearest(gt: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Downsample binary masks with nearest-neighbor interpolation.

    Args:
        gt: [B, T, 1, H, W]
        size_hw: target spatial size (H_out, W_out)

    Returns:
        [B, T, 1, H_out, W_out]
    """
    if gt.dim() != 5 or gt.size(2) != 1:
        raise ValueError(f"gt must be [B,T,1,H,W], got {tuple(gt.shape)}")

    B, T, C, H, W = gt.shape
    H_out, W_out = size_hw
    gt_flat = gt.reshape(B * T, C, H, W)
    gt_flat = F.interpolate(gt_flat, size=(H_out, W_out), mode="nearest")
    return gt_flat.reshape(B, T, C, H_out, W_out)


def soft_dice_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute soft Dice loss for binary segmentation logits.

    Args:
        logits: [N, 1, H, W]
        targets: [N, 1, H, W], values in {0, 1}
    """
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.float().flatten(1)

    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice).mean()


class VSCDCriterion(nn.Module):
    """
    BCE-with-logits + soft Dice loss for VSCDNet.

    Returns:
        {
            "loss_bce": Tensor,
            "loss_dice": Tensor,
            "loss_total": Tensor,
        }
    """

    def __init__(self, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else LossConfig()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if "Mfuse_logits" not in outputs:
            raise KeyError("outputs must contain 'Mfuse_logits'")
        if "qry_masks" not in batch:
            raise KeyError("batch must contain 'qry_masks'")

        pred = outputs["Mfuse_logits"]
        gt = batch["qry_masks"]

        if pred.dim() != 5 or pred.size(2) != 1:
            raise ValueError(f"Mfuse_logits must be [B,T,1,H,W], got {tuple(pred.shape)}")
        if gt.dim() != 5 or gt.size(2) != 1:
            raise ValueError(f"qry_masks must be [B,T,1,H,W], got {tuple(gt.shape)}")

        B, T, _, H_pred, W_pred = pred.shape

        if gt.shape[:3] != pred.shape[:3]:
            raise ValueError(
                "GT and prediction batch/time/channel dimensions must match: "
                f"gt={tuple(gt.shape[:3])}, pred={tuple(pred.shape[:3])}"
            )

        if self.cfg.downsample_gt_to_pred:
            gt = _downsample_mask_nearest(gt, (H_pred, W_pred))
        elif gt.shape[-2:] != (H_pred, W_pred):
            raise ValueError(
                f"GT and prediction resolution mismatch: gt={tuple(gt.shape[-2:])}, "
                f"pred={(H_pred, W_pred)}. Set downsample_gt_to_pred=True if needed."
            )

        if self.cfg.use_qry_valid_mask and "qry_valid" in batch:
            valid = batch["qry_valid"]
            if valid.dim() != 2 or tuple(valid.shape) != (B, T):
                raise ValueError(f"qry_valid must be [B,T]={B,T}, got {tuple(valid.shape)}")
            valid = valid.to(device=pred.device, dtype=torch.bool)
        else:
            valid = torch.ones((B, T), device=pred.device, dtype=torch.bool)

        pred_flat = pred.reshape(B * T, 1, H_pred, W_pred)
        gt_flat = gt.to(device=pred.device, dtype=pred.dtype).reshape(B * T, 1, H_pred, W_pred)
        valid_flat = valid.reshape(B * T)

        if not valid_flat.any():
            zero = pred.sum() * 0.0
            return {"loss_bce": zero, "loss_dice": zero, "loss_total": zero}

        pred_valid = pred_flat[valid_flat]
        gt_valid = gt_flat[valid_flat]

        loss_bce = F.binary_cross_entropy_with_logits(pred_valid, gt_valid)
        loss_dice = soft_dice_loss_with_logits(pred_valid, gt_valid, eps=self.cfg.eps)
        loss_total = self.cfg.bce_weight * loss_bce + self.cfg.dice_weight * loss_dice

        return {
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_total": loss_total,
        }


def build_criterion(
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    downsample_gt_to_pred: bool = True,
    use_qry_valid_mask: bool = True,
    **deprecated_kwargs,
) -> VSCDCriterion:
    """
    Build the VSCD training criterion.

    Deprecated keyword arguments are accepted and ignored for compatibility with
    older training scripts. For example, older code may still pass
    use_pos_weight or pos_weight_cap, but the public loss follows the paper's
    BCE-with-logits + soft Dice objective.
    """
    allowed_deprecated = {"use_pos_weight", "pos_weight_cap"}
    unknown = set(deprecated_kwargs) - allowed_deprecated
    if unknown:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(unknown)}")

    cfg = LossConfig(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        downsample_gt_to_pred=downsample_gt_to_pred,
        use_qry_valid_mask=use_qry_valid_mask,
    )
    return VSCDCriterion(cfg)
