# -*- coding: utf-8 -*-
"""
Collation utilities for the VSCD sequence dataset.

The dataset returns one reference-query video pair per sample. This collate
function pads variable-length frame sequences within a batch and creates valid
frame masks so the model can ignore padded frames.

For the cleaned VSCD training/evaluation pipeline, sequences usually have a
fixed number of sampled keyframes. The padding logic is kept as a safety measure
and for compatibility with older dataset variants.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def _pad_5d_video(
    seqs: List[torch.Tensor],
    pad_to: Optional[int] = None,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Pad a list of video tensors to a shared temporal length.

    Args:
        seqs: List of tensors, each with shape [T, C, H, W].
        pad_to: Optional target temporal length. If omitted, the maximum length
            in the current batch is used.
        pad_value: Value used for padded frames.

    Returns:
        padded: Tensor with shape [B, T_max, C, H, W].
        valid: Boolean tensor with shape [B, T_max], where True marks real
            frames and False marks padding.
        lengths: Original temporal lengths before padding/clipping.
    """
    assert len(seqs) > 0
    assert seqs[0].dim() == 4, f"Expected [T,C,H,W], got {seqs[0].shape}"

    lengths = [int(x.shape[0]) for x in seqs]
    batch_size = len(seqs)
    t_max = max(lengths) if pad_to is None else int(pad_to)

    _, channels, height, width = seqs[0].shape
    dtype = seqs[0].dtype

    padded = torch.full((batch_size, t_max, channels, height, width), fill_value=pad_value, dtype=dtype)
    valid = torch.zeros((batch_size, t_max), dtype=torch.bool)

    for i, x in enumerate(seqs):
        t = min(int(x.shape[0]), t_max)
        if t <= 0:
            continue
        padded[i, :t] = x[:t]
        valid[i, :t] = True

    return padded, valid, lengths


def _pad_5d_mask(
    seqs: List[torch.Tensor],
    pad_to: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad a list of binary mask sequences to a shared temporal length.

    Args:
        seqs: List of tensors, each with shape [T, 1, H, W].
        pad_to: Target temporal length.
        pad_value: Value used for padded masks.

    Returns:
        Tensor with shape [B, pad_to, 1, H, W].
    """
    assert len(seqs) > 0
    assert seqs[0].dim() == 4, f"Expected [T,1,H,W], got {seqs[0].shape}"

    batch_size = len(seqs)
    _, channels, height, width = seqs[0].shape
    assert channels == 1, f"Mask channel must be 1, got {channels}"
    dtype = seqs[0].dtype

    padded = torch.full((batch_size, pad_to, 1, height, width), fill_value=pad_value, dtype=dtype)
    for i, x in enumerate(seqs):
        t = min(int(x.shape[0]), pad_to)
        if t <= 0:
            continue
        padded[i, :t] = x[:t]
    return padded


def _clip_list(xs: List[Any], n: int) -> List[Any]:
    """Clip a metadata list to match a padded/clipped temporal length."""
    if xs is None:
        return xs
    return xs[: int(n)]


def vscd_collate_fn(
    batch: List[Dict[str, Any]],
    max_ref_frames_in_batch: Optional[int] = None,
    max_qry_frames_in_batch: Optional[int] = None,
    max_full_qry_frames_in_batch: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Collate VSCDSequenceDataset samples into a batch dictionary.

    Args:
        batch: List of samples returned by VSCDSequenceDataset.
        max_ref_frames_in_batch: Optional upper bound on the padded reference
            sequence length.
        max_qry_frames_in_batch: Optional upper bound on the padded query
            sequence length.
        max_full_qry_frames_in_batch: Optional upper bound for the optional
            full-query sequence retained for backward compatibility.

    Returns:
        A dictionary containing padded reference/query frames, query masks,
        valid-frame masks, frame IDs, and sample metadata.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    ref_seqs = [item["ref_frames"] for item in batch]
    qry_seqs = [item["qry_frames"] for item in batch]
    mask_seqs = [item["qry_masks"] for item in batch]

    ref_lengths = [int(x.shape[0]) for x in ref_seqs]
    qry_lengths = [int(x.shape[0]) for x in qry_seqs]

    ref_pad_to = max(ref_lengths)
    qry_pad_to = max(qry_lengths)

    if max_ref_frames_in_batch is not None:
        ref_pad_to = min(ref_pad_to, int(max_ref_frames_in_batch))
    if max_qry_frames_in_batch is not None:
        qry_pad_to = min(qry_pad_to, int(max_qry_frames_in_batch))

    ref_frames, ref_valid, _ = _pad_5d_video(ref_seqs, pad_to=ref_pad_to, pad_value=0.0)
    qry_frames, qry_valid, _ = _pad_5d_video(qry_seqs, pad_to=qry_pad_to, pad_value=0.0)
    qry_masks = _pad_5d_mask(mask_seqs, pad_to=qry_pad_to, pad_value=0.0)

    ref_frame_ids = [_clip_list(item.get("ref_frame_ids", []), ref_pad_to) for item in batch]
    qry_frame_ids = [_clip_list(item.get("qry_frame_ids", []), qry_pad_to) for item in batch]
    meta = [item.get("meta", {}) for item in batch]

    output: Dict[str, Any] = {
        "ref_frames": ref_frames,
        "ref_valid": ref_valid,
        "qry_frames": qry_frames,
        "qry_valid": qry_valid,
        "qry_masks": qry_masks,
        "ref_frame_ids": ref_frame_ids,
        "qry_frame_ids": qry_frame_ids,
        "meta": meta,
    }

    # Optional fields for older evaluation code that may request the full query
    # timeline. The cleaned train/eval scripts do not require these fields, but
    # preserving them keeps the dataset/collate pair backward compatible.
    has_full_query = all("qry_frames_full" in item for item in batch)
    if has_full_query:
        full_seqs = [item["qry_frames_full"] for item in batch]
        full_lengths = [int(x.shape[0]) for x in full_seqs]
        full_pad_to = max(full_lengths)

        if max_full_qry_frames_in_batch is not None:
            full_pad_to = min(full_pad_to, int(max_full_qry_frames_in_batch))

        qry_frames_full, qry_valid_full, _ = _pad_5d_video(full_seqs, pad_to=full_pad_to, pad_value=0.0)
        qry_frame_ids_full = [_clip_list(item.get("qry_frame_ids_full", []), full_pad_to) for item in batch]

        sampled_qry_indices = []
        for item in batch:
            indices = item.get("sampled_qry_indices", [])
            indices = [int(i) for i in indices if int(i) < full_pad_to]
            sampled_qry_indices.append(indices)

        output.update(
            {
                "qry_frames_full": qry_frames_full,
                "qry_valid_full": qry_valid_full,
                "qry_frame_ids_full": qry_frame_ids_full,
                "sampled_qry_indices": sampled_qry_indices,
                "full_query_stride": [item.get("full_query_stride", 1) for item in batch],
            }
        )

    return output
