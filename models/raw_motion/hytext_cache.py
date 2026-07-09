"""Cached HY-Motion text embeddings for HY273 raw-flow training."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .text_condition import RawTextCondition


_SPACE_RE = re.compile(r"\s+")


def normalize_text_key(text: str) -> str:
    """Normalize only formatting so training captions still keep their wording."""
    return _SPACE_RE.sub(" ", str(text).strip())


def hytext_key(text: str) -> str:
    return hashlib.sha1(normalize_text_key(text).encode("utf-8")).hexdigest()


class HYTextMemmapCache:
    """Lazy row-level reader for Qwen3 token and CLIP-L sentence embeddings."""

    def __init__(self, cache_dir: str | Path, max_open_shards: int = 8, strict: bool = True) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.strict = bool(strict)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"HYText cache directory not found: {self.cache_dir}")
        index_path = self.cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"HYText cache index not found: {index_path}")
        self.index: dict[str, dict[str, object]] = json.loads(index_path.read_text())
        manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        fmt = self.manifest.get("format")
        if fmt is not None and fmt != "hytext_memmap_v1":
            raise ValueError(f"Unsupported HYText cache format={fmt!r} under {self.cache_dir}")
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _open_shard(self, shard: str) -> dict[str, np.ndarray]:
        shard = str(shard)
        if shard in self._shards:
            opened = self._shards.pop(shard)
            self._shards[shard] = opened
            return opened
        shard_dir = self.cache_dir / "shards" / shard
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"HYText shard directory not found: {shard_dir}")
        opened = {
            "ctxt": np.load(shard_dir / "ctxt.npy", mmap_mode="r"),
            "vtxt": np.load(shard_dir / "vtxt.npy", mmap_mode="r"),
            "ctxt_len": np.load(shard_dir / "ctxt_len.npy", mmap_mode="r"),
        }
        self._shards[shard] = opened
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return opened

    def lookup_rows(self, texts: Iterable[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entries: list[dict[str, object]] = []
        empty_entry = self.index.get(hytext_key(""))
        for text in texts:
            key = hytext_key(str(text))
            entry = self.index.get(key)
            if entry is None:
                if self.strict or empty_entry is None:
                    raise KeyError(
                        f"HYText cache miss for key={key} text={normalize_text_key(str(text))!r} "
                        f"under {self.cache_dir}"
                    )
                entry = empty_entry
            entries.append(entry)
        if not entries:
            ctxt_dim = int(self.manifest.get("ctxt_dim", 4096))
            vtxt_dim = int(self.manifest.get("vtxt_dim", 768))
            max_len = int(self.manifest.get("max_length_llm", 0))
            return (
                torch.empty(0, 1, vtxt_dim),
                torch.empty(0, max_len, ctxt_dim),
                torch.empty(0, dtype=torch.long),
            )

        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for out_i, entry in enumerate(entries):
            groups[str(entry["shard"])].append((out_i, int(entry["row"])))

        ctxt_rows: list[np.ndarray | None] = [None] * len(entries)
        vtxt_rows: list[np.ndarray | None] = [None] * len(entries)
        len_rows: list[int] = [0] * len(entries)
        for shard, pairs in groups.items():
            opened = self._open_shard(shard)
            row_ids = np.array([row for _, row in pairs], dtype=np.int64)
            ctxt_batch = np.asarray(opened["ctxt"][row_ids]).copy()
            vtxt_batch = np.asarray(opened["vtxt"][row_ids]).copy()
            len_batch = np.asarray(opened["ctxt_len"][row_ids]).copy()
            for j, (out_i, _) in enumerate(pairs):
                ctxt_rows[out_i] = ctxt_batch[j]
                vtxt_rows[out_i] = vtxt_batch[j]
                len_rows[out_i] = int(len_batch[j].item())
        if any(row is None for row in ctxt_rows) or any(row is None for row in vtxt_rows):
            raise RuntimeError("Internal HYText cache grouping error: missing output rows")
        ctxt = torch.from_numpy(np.stack(ctxt_rows, axis=0))
        vtxt = torch.from_numpy(np.stack(vtxt_rows, axis=0))
        lengths = torch.tensor(len_rows, dtype=torch.long)
        return vtxt, ctxt, lengths


class CachedHYTextEncoder(nn.Module):
    """Projection bridge from cached HY-Motion HYText embeddings to RawTextCondition."""

    def __init__(
        self,
        hidden_dim: int,
        cache_dir: str | Path,
        max_text_tokens: int = 128,
        ctxt_dim: int = 4096,
        vtxt_dim: int = 768,
        max_open_shards: int = 8,
        strict_cache: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_text_tokens = int(max_text_tokens)
        self.ctxt_dim = int(ctxt_dim)
        self.vtxt_dim = int(vtxt_dim)
        self.cache = HYTextMemmapCache(cache_dir, max_open_shards=max_open_shards, strict=strict_cache)
        self._validate_manifest()
        self.token_proj = nn.Linear(self.ctxt_dim, self.hidden_dim)
        self.pooled_proj = nn.Sequential(
            nn.Linear(self.vtxt_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _validate_manifest(self) -> None:
        manifest = self.cache.manifest
        cache_ctxt_dim = manifest.get("ctxt_dim")
        cache_vtxt_dim = manifest.get("vtxt_dim")
        cache_max_len = manifest.get("max_length_llm")
        if cache_ctxt_dim is not None and int(cache_ctxt_dim) != self.ctxt_dim:
            raise ValueError(f"HYText ctxt_dim mismatch: cache={cache_ctxt_dim}, model={self.ctxt_dim}")
        if cache_vtxt_dim is not None and int(cache_vtxt_dim) != self.vtxt_dim:
            raise ValueError(f"HYText vtxt_dim mismatch: cache={cache_vtxt_dim}, model={self.vtxt_dim}")
        if cache_max_len is not None and int(cache_max_len) < self.max_text_tokens:
            raise ValueError(
                f"HYText cache max_length_llm={cache_max_len} is shorter than model max_text_tokens={self.max_text_tokens}"
            )
        if self.cache.index:
            first = next(iter(self.cache.index.values()))
            opened = self.cache._open_shard(str(first["shard"]))
            ctxt = opened["ctxt"]
            vtxt = opened["vtxt"]
            if ctxt.ndim != 3 or int(ctxt.shape[-1]) != self.ctxt_dim:
                raise ValueError(f"HYText ctxt array shape {ctxt.shape} does not match ctxt_dim={self.ctxt_dim}")
            if vtxt.ndim != 3 or int(vtxt.shape[-1]) != self.vtxt_dim:
                raise ValueError(f"HYText vtxt array shape {vtxt.shape} does not match vtxt_dim={self.vtxt_dim}")

    def forward(
        self,
        texts: Iterable[str],
        device: torch.device,
        dtype: torch.dtype,
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> RawTextCondition:
        text_list = [str(t) for t in texts]
        if force_drop:
            text_list = [""] * len(text_list)
        elif drop_prob > 0.0 and text_list:
            keep = torch.rand(len(text_list), device=device) >= float(drop_prob)
            text_list = [text if bool(keep[i].item()) else "" for i, text in enumerate(text_list)]

        vtxt, ctxt, lengths = self.cache.lookup_rows(text_list)
        tokens = ctxt[:, : self.max_text_tokens]
        if tokens.shape[1] < self.max_text_tokens:
            pad = tokens.new_zeros(tokens.shape[0], self.max_text_tokens - tokens.shape[1], tokens.shape[2])
            tokens = torch.cat([tokens, pad], dim=1)
        pooled = vtxt[:, 0]
        lengths = lengths.clamp(min=0, max=self.max_text_tokens)
        arange = torch.arange(self.max_text_tokens).view(1, self.max_text_tokens)
        padding = arange >= lengths.view(-1, 1)
        if padding.all(dim=1).any():
            padding = padding.clone()
            padding[padding.all(dim=1), 0] = False

        tokens = self.token_proj(tokens.to(device=device, dtype=dtype))
        pooled = self.pooled_proj(pooled.to(device=device, dtype=dtype))
        padding = padding.to(device=device)
        return RawTextCondition(tokens=tokens, pooled=pooled, padding_mask=padding)
