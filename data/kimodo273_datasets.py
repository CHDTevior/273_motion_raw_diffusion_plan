"""Dataset wrappers for converted HumanML3D/MotionFix Kimodo273 motion files."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from models.raw_motion.hy273_slices import DIM_HY273, FALLBACK_SHORT_CLIPS


def parse_hml_text_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if "#" in line:
        return line.split("#", 1)[0].strip()
    return line


def _read_captions(path: Path) -> list[str]:
    if not path.is_file():
        return [""]
    captions: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = parse_hml_text_line(line)
            if text:
                captions.append(text)
    return captions or [""]


@dataclass
class Kimodo273Item:
    motion: torch.Tensor
    length: int
    text: str
    rel_path: str
    motion_id: str


class Kimodo273TextDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        text_root: str | Path | None = None,
        max_frames: int = 300,
        min_frames: int = 16,
        random_crop: bool = True,
        exclude_fallback_short_clips: bool = True,
        deterministic_text: bool = False,
        cache_manifest: bool = True,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = str(split)
        self.text_root = Path(text_root).expanduser().resolve() if text_root else self.data_root.parent / "texts"
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.random_crop = bool(random_crop)
        self.deterministic_text = bool(deterministic_text)
        split_path = self.data_root / "split_existing" / f"{self.split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        frame_by_rel: dict[str, int] = {}
        fallback_by_rel: dict[str, bool] = {}
        manifest_path = self.data_root / "manifest.jsonl"
        if cache_manifest and manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    rel = str(row["relative_path"])
                    frame_by_rel[rel] = int(row.get("frames", 0))
                    fallback_by_rel[rel] = bool(row.get("smooth_root_fallback", False))
        records: list[dict[str, Any]] = []
        for motion_id in ids:
            rel = f"motion_data/{motion_id}.npy"
            path = self.data_root / rel
            if not path.is_file():
                continue
            frames = frame_by_rel.get(rel)
            if frames is None:
                frames = int(np.load(path, mmap_mode="r").shape[0])
            if frames < self.min_frames:
                continue
            if exclude_fallback_short_clips and (rel in FALLBACK_SHORT_CLIPS or fallback_by_rel.get(rel, False)):
                continue
            records.append({"id": motion_id, "rel": rel, "frames": frames, "path": path})
        if not records:
            raise RuntimeError(f"No usable records for split={split} under {self.data_root}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _crop(self, motion: np.ndarray) -> np.ndarray:
        if self.max_frames <= 0 or motion.shape[0] <= self.max_frames:
            return motion
        span = motion.shape[0] - self.max_frames
        if self.random_crop:
            start = random.randint(0, span)
        else:
            start = span // 2
        return motion[start : start + self.max_frames]

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        motion = np.load(rec["path"]).astype(np.float32, copy=False)
        if motion.ndim != 2 or motion.shape[1] != DIM_HY273:
            raise ValueError(f"Expected [T,{DIM_HY273}] at {rec['path']}, got {motion.shape}")
        motion = self._crop(motion)
        motion_id = rec["id"]
        captions = _read_captions(self.text_root / f"{motion_id}.txt")
        text = captions[0] if self.deterministic_text else random.choice(captions)
        return {
            "motion": torch.from_numpy(motion.copy()),
            "length": int(motion.shape[0]),
            "text": text,
            "rel_path": rec["rel"],
            "motion_id": motion_id,
        }


def collate_kimodo273_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(int(item["length"]) for item in batch)
    motion = torch.zeros(len(batch), max_len, DIM_HY273, dtype=torch.float32)
    valid = torch.zeros(len(batch), max_len, dtype=torch.bool)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    texts: list[str] = []
    rel_paths: list[str] = []
    motion_ids: list[str] = []
    for i, item in enumerate(batch):
        cur = item["motion"].float()
        length = int(cur.shape[0])
        motion[i, :length] = cur
        valid[i, :length] = True
        lengths[i] = length
        texts.append(str(item["text"]))
        rel_paths.append(str(item["rel_path"]))
        motion_ids.append(str(item["motion_id"]))
    return {
        "motion": motion,
        "lengths": lengths,
        "valid": valid,
        "texts": texts,
        "rel_paths": rel_paths,
        "motion_ids": motion_ids,
    }
