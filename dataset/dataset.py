# -*- coding: utf-8 -*-
"""
VSCD sequence dataset.

Each sample corresponds to one directed scene pair from a single space:
    reference: scene a
    query:     scene b
    mask:      query-aligned change masks for scene a -> scene b

Expected dataset layout:
    <root>/
      train/
        <space_name>/
          scene0_frames/scene0_0000.jpg
          scene1_frames/scene1_0000.jpg
          ...
          change_mask_scene0_to_scene1_frames/
            change_mask_scene0_to_scene1_0000.jpg
      test/
        <space_name>/
          ...

The default directed pairs follow the VSCD benchmark protocol:
    0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 0

The dataset returns a fixed number of uniformly sampled frames from both
reference and query videos. If a video is shorter than the requested number,
frame IDs are repeated so that the tensor shape remains fixed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_PAIRS: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1] in IMG_EXTS


def _frame_regex(scene_idx: int) -> re.Pattern:
    return re.compile(rf"^scene{scene_idx}_(\d{{4}})\.jpg$", re.IGNORECASE)


def _list_frame_ids_strict(scene_dir: Path, scene_idx: int) -> List[str]:
    """Return sorted four-digit frame IDs for files named scene{idx}_{####}.jpg."""
    if not scene_dir.is_dir():
        return []

    pattern = _frame_regex(scene_idx)
    frame_ids: List[str] = []

    for path in scene_dir.iterdir():
        if not path.is_file() or not _is_image(str(path)):
            continue
        match = pattern.match(path.name)
        if match is not None:
            frame_ids.append(match.group(1))

    frame_ids.sort(key=lambda x: int(x))
    return frame_ids


def _frame_path(scene_dir: Path, scene_idx: int, frame_id: str) -> Path:
    return scene_dir / f"scene{scene_idx}_{str(frame_id).zfill(4)}.jpg"


def _mask_dir(space_dir: Path, a: int, b: int) -> Path:
    return space_dir / f"change_mask_scene{a}_to_scene{b}_frames"


def _mask_path(space_dir: Path, a: int, b: int, frame_id: str) -> Path:
    return _mask_dir(space_dir, a, b) / f"change_mask_scene{a}_to_scene{b}_{str(frame_id).zfill(4)}.jpg"


def _sample_ids_fixed_count(frame_ids: Sequence[str], num_frames: int) -> List[str]:
    """
    Uniformly sample exactly `num_frames` IDs.

    If the source sequence is shorter than `num_frames`, repeated IDs are used.
    """
    ids = list(frame_ids)
    length = len(ids)
    num_frames = int(num_frames)

    if length == 0 or num_frames <= 0:
        return []
    if length == 1:
        return [ids[0]] * num_frames

    indices = torch.linspace(0, length - 1, steps=num_frames).round().long().tolist()
    indices = [max(0, min(length - 1, i)) for i in indices]
    return [ids[i] for i in indices]


def _merge_sampled_ids_into_full_ids(full_ids: Sequence[str], sampled_ids: Sequence[str]) -> List[str]:
    """Ensure that optional full-query IDs contain every sampled query ID."""
    merged = set(full_ids)
    merged.update(sampled_ids)
    return sorted(merged, key=lambda x: int(x))


class VSCDSequenceDataset(Dataset):
    """
    Dataset for fixed-keyframe VSCD training and evaluation.

    Returned fields:
        ref_frames:    Tensor [T, 3, H, W]
        qry_frames:    Tensor [T, 3, H, W]
        qry_masks:     Tensor [T, 1, H, W]
        ref_frame_ids: list[str]
        qry_frame_ids: list[str]
        meta:          dict with space name and directed pair

    The optional full-query fields are kept for backward compatibility with
    older evaluation scripts. They are not required by the cleaned training or
    evaluation pipeline.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames_fixed: int = 32,
        pairs: Optional[List[Tuple[int, int]]] = None,
        img_wh: Tuple[int, int] = (1024, 1024),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        transforms: Optional[Any] = None,
        return_full_query: bool = False,
        full_query_stride: int = 1,
        mask_pos_threshold: int = 0,
        drop_query_without_mask: bool = True,
        max_ref_frames: Optional[int] = None,
        max_qry_frames: Optional[int] = None,
        include_spaces: Optional[Sequence[str]] = None,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split

        self.num_frames_fixed = int(num_frames_fixed)
        if self.num_frames_fixed <= 0:
            raise ValueError("num_frames_fixed must be positive")

        self.pairs = list(pairs) if pairs is not None else list(DEFAULT_PAIRS)
        self.W, self.H = int(img_wh[0]), int(img_wh[1])
        if self.W <= 0 or self.H <= 0:
            raise ValueError(f"img_wh must contain positive integers, got {img_wh}")

        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.transforms = transforms

        self.return_full_query = bool(return_full_query)
        self.full_query_stride = max(1, int(full_query_stride))
        self.mask_pos_threshold = int(mask_pos_threshold)
        self.drop_query_without_mask = bool(drop_query_without_mask)
        self.max_ref_frames = max_ref_frames
        self.max_qry_frames = max_qry_frames

        self.include_spaces = None
        if include_spaces is not None:
            self.include_spaces = {str(name).strip() for name in include_spaces if str(name).strip()}

        self.items = self._scan_items()
        if len(self.items) == 0:
            raise RuntimeError(f"[VSCDSequenceDataset] No valid (space, pair) found under: {self.split_dir}")

    def _list_spaces(self) -> List[Path]:
        if not self.split_dir.is_dir():
            return []

        spaces = [path for path in self.split_dir.iterdir() if path.is_dir()]
        if self.include_spaces is not None:
            spaces = [path for path in spaces if path.name in self.include_spaces]

        spaces.sort(key=lambda path: path.name)
        return spaces

    def _query_ids_with_masks(self, space_dir: Path, a: int, b: int, qry_ids_all: Sequence[str]) -> List[str]:
        if not self.drop_query_without_mask:
            return list(qry_ids_all)

        valid_ids = []
        for frame_id in qry_ids_all:
            if _mask_path(space_dir, a, b, frame_id).is_file():
                valid_ids.append(frame_id)
        return valid_ids

    def _scan_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        for space_dir in self._list_spaces():
            for a, b in self.pairs:
                ref_dir = space_dir / f"scene{a}_frames"
                qry_dir = space_dir / f"scene{b}_frames"
                mask_dir = _mask_dir(space_dir, a, b)

                if not (ref_dir.is_dir() and qry_dir.is_dir() and mask_dir.is_dir()):
                    continue

                ref_ids_all = _list_frame_ids_strict(ref_dir, a)
                qry_ids_all = _list_frame_ids_strict(qry_dir, b)
                qry_ids_for_sampling = self._query_ids_with_masks(space_dir, a, b, qry_ids_all)

                if len(ref_ids_all) == 0 or len(qry_ids_for_sampling) == 0:
                    continue

                ref_ids = _sample_ids_fixed_count(ref_ids_all, self.num_frames_fixed)
                qry_ids = _sample_ids_fixed_count(qry_ids_for_sampling, self.num_frames_fixed)

                if self.max_ref_frames is not None:
                    ref_ids = ref_ids[: int(self.max_ref_frames)]
                if self.max_qry_frames is not None:
                    qry_ids = qry_ids[: int(self.max_qry_frames)]

                if len(ref_ids) == 0 or len(qry_ids) == 0:
                    continue

                items.append(
                    {
                        "space": space_dir.name,
                        "space_dir": str(space_dir),
                        "pair": (a, b),
                        "ref_dir": str(ref_dir),
                        "qry_dir": str(qry_dir),
                        "ref_ids": ref_ids,
                        "qry_ids": qry_ids,
                        "qry_ids_all": qry_ids_all,
                    }
                )

        return items

    def _load_rgb(self, path: Path) -> np.ndarray:
        image = Image.open(str(path)).convert("RGB")
        image = image.resize((self.W, self.H), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return np.transpose(array, (2, 0, 1))

    def _load_mask01(self, path: Path) -> np.ndarray:
        image = Image.open(str(path)).convert("L")
        image = image.resize((self.W, self.H), Image.NEAREST)
        array = np.asarray(image, dtype=np.float32)
        array = (array > self.mask_pos_threshold).astype(np.float32)
        return np.expand_dims(array, axis=0)

    def _load_frames(self, scene_dir: Path, scene_idx: int, frame_ids: Sequence[str]) -> torch.Tensor:
        frames = []
        for frame_id in frame_ids:
            path = _frame_path(scene_dir, scene_idx, frame_id)
            if not path.is_file():
                raise FileNotFoundError(f"Missing frame: {path}")
            frames.append(self._load_rgb(path))
        return torch.from_numpy(np.stack(frames, axis=0)).float()

    def _load_masks(self, space_dir: Path, a: int, b: int, qry_frame_ids: Sequence[str]) -> torch.Tensor:
        masks = []
        for frame_id in qry_frame_ids:
            path = _mask_path(space_dir, a, b, frame_id)
            if not path.is_file():
                raise FileNotFoundError(f"Missing mask: {path}")
            masks.append(self._load_mask01(path))
        return torch.from_numpy(np.stack(masks, axis=0)).float()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        a, b = item["pair"]

        space_dir = Path(item["space_dir"])
        ref_dir = Path(item["ref_dir"])
        qry_dir = Path(item["qry_dir"])
        ref_ids: List[str] = item["ref_ids"]
        qry_ids: List[str] = item["qry_ids"]

        output: Dict[str, Any] = {
            "ref_frames": self._load_frames(ref_dir, a, ref_ids),
            "qry_frames": self._load_frames(qry_dir, b, qry_ids),
            "qry_masks": self._load_masks(space_dir, a, b, qry_ids),
            "ref_frame_ids": ref_ids,
            "qry_frame_ids": qry_ids,
            "meta": {
                "space": item["space"],
                "pair": (a, b),
                "num_frames_fixed": self.num_frames_fixed,
            },
        }

        if self.return_full_query:
            qry_ids_all: List[str] = item["qry_ids_all"]
            qry_ids_full = qry_ids_all[:: self.full_query_stride]
            qry_ids_full = _merge_sampled_ids_into_full_ids(qry_ids_full, qry_ids)
            id_to_idx = {frame_id: i for i, frame_id in enumerate(qry_ids_full)}

            output.update(
                {
                    "qry_frames_full": self._load_frames(qry_dir, b, qry_ids_full),
                    "qry_frame_ids_full": qry_ids_full,
                    "sampled_qry_indices": [id_to_idx[frame_id] for frame_id in qry_ids],
                    "full_query_stride": self.full_query_stride,
                }
            )

        if self.transforms is not None:
            output = self.transforms(output)

        return output
