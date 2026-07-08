"""Cached HY-Motion text embeddings for HY273 raw-flow training."""

from __future__ import annotations

import hashlib
import json
import re
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
        ctxt_rows: list[np.ndarray] = []
        vtxt_rows: list[np.ndarray] = []
        len_rows: list[int] = []
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
            shard = str(entry["shard"])
            row = int(entry["row"])
            opened = self._open_shard(shard)
            ctxt_rows.append(np.asarray(opened["ctxt"][row]).copy())
            vtxt_rows.append(np.asarray(opened["vtxt"][row]).copy())
            len_rows.append(int(np.asarray(opened["ctxt_len"][row]).item()))
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
        self.token_proj = nn.Linear(self.ctxt_dim, self.hidden_dim)
        self.pooled_proj = nn.Sequential(
            nn.Linear(self.vtxt_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

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
