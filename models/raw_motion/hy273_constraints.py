"""Synthetic HY273 observed_motion / motion_mask construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import torch

from .hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    FIVE_POINT_JOINTS,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    KIMODO_EE_JOINTS,
    ROOT_SLICE,
    joint_pos_slice_for,
)

ControlMode = Literal["none", "root", "endpoints", "fullpose", "mixed"]


@dataclass
class ControlBatch:
    observed_motion: torch.Tensor
    motion_mask: torch.Tensor
    mode_ids: list[str]


def _select_frames(length: int, count: int, device: torch.device) -> torch.Tensor:
    if length <= 0 or count <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    count = min(int(count), int(length))
    if count == 1:
        return torch.randint(0, int(length), (1,), device=device)
    lin = torch.linspace(0, int(length) - 1, count, device=device)
    jitter = torch.randint(-1, 2, (count,), device=device)
    return (lin.round().long() + jitter).clamp(0, int(length) - 1).unique()


def _apply_indices(
    obs: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    batch_idx: int,
    frames: torch.Tensor,
    indices: Sequence[int],
) -> None:
    if frames.numel() == 0 or not indices:
        return
    idx = torch.as_tensor(indices, device=source.device, dtype=torch.long)
    obs[batch_idx, frames[:, None], idx[None, :]] = source[batch_idx, frames[:, None], idx[None, :]]
    mask[batch_idx, frames[:, None], idx[None, :]] = True


def build_synthetic_control_batch(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    modes: Sequence[ControlMode] = ("none", "root", "endpoints", "fullpose", "mixed"),
    endpoint_preset: str = "kimodo_ee",
    min_keyframes: int = 1,
    max_keyframes: int = 8,
    include_root_ref_for_endpoints: bool = True,
    generator: Optional[torch.Generator] = None,
) -> ControlBatch:
    """Sample observed controls from the same clean motion.

    Inputs/outputs are in the same domain as ``motion``; call before normalization for
    source-domain controls, or after normalization for model-domain controls.
    """
    if motion.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected motion [B,T,{DIM_HY273}], got {tuple(motion.shape)}")
    bsz, _frames, _ = motion.shape
    device = motion.device
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    mode_names: list[str] = []
    endpoint_joints = KIMODO_EE_JOINTS if endpoint_preset == "kimodo_ee" else FIVE_POINT_JOINTS
    endpoint_indices = joint_pos_slice_for(endpoint_joints)
    root_indices = list(range(ROOT_SLICE.start, HEADING_SLICE.stop))
    fullpose_indices = list(range(JOINT_POS_SLICE.start, JOINT_POS_SLICE.stop))

    for b in range(bsz):
        mode = modes[int(torch.randint(0, len(modes), (1,), device=device, generator=generator).item())]
        mode_names.append(str(mode))
        length = int(lengths[b].item())
        if mode == "none":
            continue
        k = int(torch.randint(min_keyframes, max_keyframes + 1, (1,), device=device, generator=generator).item())
        keyframes = _select_frames(length, k, device)
        if mode == "root":
            _apply_indices(obs, mask, motion, b, keyframes, root_indices)
        elif mode == "endpoints":
            if include_root_ref_for_endpoints:
                _apply_indices(obs, mask, motion, b, keyframes, root_indices)
            _apply_indices(obs, mask, motion, b, keyframes, endpoint_indices)
        elif mode == "fullpose":
            _apply_indices(obs, mask, motion, b, keyframes, root_indices)
            _apply_indices(obs, mask, motion, b, keyframes, fullpose_indices)
        elif mode == "mixed":
            _apply_indices(obs, mask, motion, b, keyframes, root_indices)
            _apply_indices(obs, mask, motion, b, keyframes, endpoint_indices)
            if torch.rand((), device=device, generator=generator) < 0.25:
                _apply_indices(obs, mask, motion, b, keyframes, list(range(CONTACT_SLICE.start, CONTACT_SLICE.stop)))
        else:
            raise ValueError(f"Unknown control mode: {mode}")
    return ControlBatch(observed_motion=obs, motion_mask=mask, mode_ids=mode_names)


def compile_root_control(motion: torch.Tensor, frames: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    f = torch.as_tensor(list(frames), device=motion.device, dtype=torch.long)
    idx = torch.arange(ROOT_SLICE.start, HEADING_SLICE.stop, device=motion.device)
    obs[f[:, None], idx[None, :]] = motion[f[:, None], idx[None, :]]
    mask[f[:, None], idx[None, :]] = True
    return obs, mask
