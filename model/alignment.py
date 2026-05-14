# -*- coding: utf-8 -*-
"""
Frame-level alignment for VSCDNet.

This module builds a soft frame-matching distribution between query and
reference keyframes and constructs matched segment proposals from the top-1
alignment path. The resulting distribution is used both for reference candidate
ranking and for frame-level confidence during feature fusion.

Inputs:
    vq:        [B, Tq, D] query frame descriptors
    vr:        [B, Tr, D] reference frame descriptors
    qry_valid: [B, Tq] optional valid-frame mask
    ref_valid: [B, Tr] optional valid-frame mask

Outputs:
    P_frame: [B, Tq, Tr] row-wise reference matching distribution
    msps:    list of matched segment proposals per batch item
    aux:     diagnostic tensors used by training/evaluation code
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MSP:
    """Matched segment proposal on the query-reference top-1 path."""

    t0: int
    t1: int
    s0: int
    s1: int
    score: float


class FrameAlignment(nn.Module):
    """
    Frame-level alignment module.

    The module first computes cosine similarities between query and reference
    frame descriptors, refines the similarity grid with a shallow residual
    convolutional head, and applies a row-wise softmax over reference frames.
    It then groups the top-1 path into short near-diagonal matched segment
    proposals (MSPs), while optionally adding singleton proposals so that every
    valid query frame is covered.
    """

    def __init__(
        self,
        conv_hidden: int = 16,
        conv_layers: int = 2,
        logit_tau: float = -1e9,
        diag_delta: int = 2,
        min_len: int = 2,
        max_len: int = 5,
        ensure_full_coverage: bool = True,
        softmax_temp: float = 0.5,
    ):
        super().__init__()

        if conv_layers < 1:
            raise ValueError("conv_layers must be >= 1")
        if diag_delta < 0:
            raise ValueError("diag_delta must be >= 0")
        if min_len < 1:
            raise ValueError("min_len must be >= 1")
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        if softmax_temp <= 0:
            raise ValueError("softmax_temp must be > 0")

        self.logit_tau = float(logit_tau)
        self.diag_delta = int(diag_delta)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.ensure_full_coverage = bool(ensure_full_coverage)
        self.softmax_temp = float(softmax_temp)

        layers: List[nn.Module] = []
        in_ch = 1
        for _ in range(conv_layers - 1):
            layers.append(nn.Conv2d(in_ch, conv_hidden, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = conv_hidden

        last = nn.Conv2d(in_ch, 1, kernel_size=1)
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
        layers.append(last)

        self.conv = nn.Sequential(*layers)

    @staticmethod
    def _cosine_sim_map(vq: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        """Return the cosine similarity grid [B, Tq, Tr]."""
        vq_n = F.normalize(vq, dim=-1)
        vr_n = F.normalize(vr, dim=-1)
        return torch.einsum("btd,bsd->bts", vq_n, vr_n)

    @staticmethod
    def _validate_optional_mask(mask: Optional[torch.Tensor], shape: Tuple[int, int], name: str) -> None:
        if mask is None:
            return
        if mask.dim() != 2 or tuple(mask.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")

    @staticmethod
    def _mask_invalid(
        S: torch.Tensor,
        qry_valid: Optional[torch.Tensor],
        ref_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mask invalid query/reference entries before softmax."""
        if qry_valid is not None:
            S = S.masked_fill((~qry_valid.bool()).unsqueeze(-1), -1e9)
        if ref_valid is not None:
            S = S.masked_fill((~ref_valid.bool()).unsqueeze(1), -1e9)
        return S

    def _build_msps_top1(
        self,
        top1_s: torch.Tensor,
        top1_prob: torch.Tensor,
        top1_logit: torch.Tensor,
        qry_valid_1d: Optional[torch.Tensor],
    ) -> List[MSP]:
        """
        Group the top-1 path into near-diagonal matched segment proposals.

        A segment is extended when the reference index approximately advances
        by one step between adjacent query frames. Short segments are removed,
        and uncovered valid query frames are optionally represented by singleton
        MSPs to keep candidate construction well-defined.
        """
        Tq = top1_s.numel()
        valid_t = (
            torch.ones(Tq, dtype=torch.bool, device=top1_s.device)
            if qry_valid_1d is None
            else qry_valid_1d.bool()
        )

        msps: List[MSP] = []
        covered = torch.zeros(Tq, dtype=torch.bool, device=top1_s.device)
        cur_t0: Optional[int] = None
        cur_t1: Optional[int] = None

        def flush_segment(t0: Optional[int], t1: Optional[int]) -> None:
            if t0 is None or t1 is None or t1 < t0:
                return
            s0 = int(top1_s[t0].item())
            s1 = int(top1_s[t1].item())
            score = float(top1_prob[t0 : t1 + 1].mean().item())
            msps.append(MSP(t0=t0, t1=t1, s0=s0, s1=s1, score=score))
            covered[t0 : t1 + 1] = True

        for t in range(Tq):
            if not bool(valid_t[t].item()):
                flush_segment(cur_t0, cur_t1)
                cur_t0 = cur_t1 = None
                continue

            if float(top1_logit[t].item()) < self.logit_tau:
                flush_segment(cur_t0, cur_t1)
                cur_t0 = cur_t1 = None
                continue

            if cur_t0 is None:
                cur_t0 = cur_t1 = t
                continue

            assert cur_t1 is not None
            prev_s = int(top1_s[cur_t1].item())
            cur_s = int(top1_s[t].item())
            near_diagonal = abs((cur_s - prev_s) - 1) <= self.diag_delta
            within_max_len = (t - cur_t0 + 1) <= self.max_len

            if near_diagonal and within_max_len:
                cur_t1 = t
            else:
                flush_segment(cur_t0, cur_t1)
                cur_t0 = cur_t1 = t

        flush_segment(cur_t0, cur_t1)

        filtered: List[MSP] = []
        for m in msps:
            if (m.t1 - m.t0 + 1) >= self.min_len:
                filtered.append(m)
            else:
                covered[m.t0 : m.t1 + 1] = False
        msps = filtered

        if self.ensure_full_coverage:
            for t in range(Tq):
                if not bool(valid_t[t].item()) or bool(covered[t].item()):
                    continue
                s = int(top1_s[t].item())
                score = float(top1_prob[t].item())
                msps.append(MSP(t0=t, t1=t, s0=s, s1=s, score=score))
                covered[t] = True

        msps.sort(key=lambda m: (m.t0, m.t1, m.s0, m.s1))
        return msps

    def forward(
        self,
        vq: torch.Tensor,
        vr: torch.Tensor,
        qry_valid: Optional[torch.Tensor] = None,
        ref_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]], Dict[str, Any]]:
        if vq.dim() != 3 or vr.dim() != 3:
            raise ValueError(f"vq and vr must be 3D tensors, got {tuple(vq.shape)} and {tuple(vr.shape)}")

        B, Tq, D = vq.shape
        Br, Tr, Dr = vr.shape
        if Br != B or Dr != D:
            raise ValueError(f"Batch/dim mismatch: vq={tuple(vq.shape)}, vr={tuple(vr.shape)}")

        self._validate_optional_mask(qry_valid, (B, Tq), "qry_valid")
        self._validate_optional_mask(ref_valid, (B, Tr), "ref_valid")

        if ref_valid is not None and (ref_valid.bool().sum(dim=1) == 0).any():
            raise RuntimeError("Each sample must contain at least one valid reference frame.")

        S_frame = self._cosine_sim_map(vq, vr)
        S_frame = self._mask_invalid(S_frame, qry_valid, ref_valid)

        logits = S_frame + self.conv(S_frame.unsqueeze(1)).squeeze(1)
        P_frame = F.softmax(logits / self.softmax_temp, dim=-1)

        top1_prob, top1_s = torch.max(P_frame, dim=-1)
        top1_logit = torch.gather(logits, dim=-1, index=top1_s.unsqueeze(-1)).squeeze(-1)

        msps_out: List[List[Dict[str, Any]]] = []
        for b in range(B):
            qv = qry_valid[b] if qry_valid is not None else None
            msps_b = self._build_msps_top1(top1_s[b], top1_prob[b], top1_logit[b], qv)
            msps_out.append(
                [
                    {
                        "t0": m.t0,
                        "t1": m.t1,
                        "s0": m.s0,
                        "s1": m.s1,
                        "score": m.score,
                    }
                    for m in msps_b
                ]
            )

        aux = {
            "S_frame": S_frame,
            "logits": logits,
            "top1_s": top1_s,
            "top1_prob": top1_prob,
            "top1_logit": top1_logit,
        }
        return P_frame, msps_out, aux
