"""Synthetic HY273 observed_motion / motion_mask construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import torch

from .hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    FIVE_POINT_GROUPS,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    KIMODO_EE_GROUPS,
    ROOT_SLICE,
    joint_pos_slice_for,
)

ControlMode = Literal["none", "root", "endpoints", "fullpose", "mixed"]
EndpointPreset = Literal["kimodo_ee", "five_point"]
EndpointSubsetMode = Literal["all", "random_nonempty"]


@dataclass
class ControlBatch:
    observed_motion: torch.Tensor
    motion_mask: torch.Tensor
    mode_ids: list[str]


def _select_frames(
    length: int,
    count: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if length <= 0 or count <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    count = min(int(count), int(length))
    if count == 1:
        return torch.randint(0, int(length), (1,), device=device, generator=generator)
    lin = torch.linspace(0, int(length) - 1, count, device=device)
    jitter = torch.randint(-1, 2, (count,), device=device, generator=generator)
    return (lin.round().long() + jitter).clamp(0, int(length) - 1).unique()


def _endpoint_groups(endpoint_preset: EndpointPreset) -> tuple[tuple[int, ...], ...]:
    if endpoint_preset == "kimodo_ee":
        return KIMODO_EE_GROUPS
    if endpoint_preset == "five_point":
        return FIVE_POINT_GROUPS
    raise ValueError(f"Unknown endpoint preset: {endpoint_preset}")


def _sample_endpoint_joints(
    groups: tuple[tuple[int, ...], ...],
    subset_mode: EndpointSubsetMode,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> tuple[int, ...]:
    if subset_mode == "all":
        return tuple(joint_id for group in groups for joint_id in group)
    if subset_mode != "random_nonempty":
        raise ValueError(f"Unknown endpoint subset mode: {subset_mode}")

    selected = torch.rand(len(groups), device=device, generator=generator) < 0.5
    if not bool(selected.any().item()):
        selected[
            torch.randint(0, len(groups), (1,), device=device, generator=generator).item()
        ] = True
    return tuple(
        joint_id
        for group_idx, group in enumerate(groups)
        if bool(selected[group_idx].item())
        for joint_id in group
    )


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
    endpoint_preset: EndpointPreset = "kimodo_ee",
    endpoint_subset_mode: EndpointSubsetMode = "random_nonempty",
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
    if not modes:
        raise ValueError("At least one control mode is required")
    if min_keyframes < 1 or max_keyframes < min_keyframes:
        raise ValueError(
            f"Expected 1 <= min_keyframes <= max_keyframes, got {min_keyframes}, {max_keyframes}"
        )
    bsz, _frames, _ = motion.shape
    device = motion.device
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    mode_names: list[str] = []
    endpoint_groups = _endpoint_groups(endpoint_preset)
    root_indices = list(range(ROOT_SLICE.start, HEADING_SLICE.stop))
    fullpose_indices = list(range(JOINT_POS_SLICE.start, JOINT_POS_SLICE.stop))

    for b in range(bsz):
        mode = modes[int(torch.randint(0, len(modes), (1,), device=device, generator=generator).item())]
        mode_names.append(str(mode))
        length = int(lengths[b].item())
        if mode == "none":
            continue
        k = int(torch.randint(min_keyframes, max_keyframes + 1, (1,), device=device, generator=generator).item())
        keyframes = _select_frames(length, k, device, generator=generator)
        if mode == "root":
            _apply_indices(obs, mask, motion, b, keyframes, root_indices)
        elif mode == "endpoints":
            endpoint_joints = _sample_endpoint_joints(
                endpoint_groups, endpoint_subset_mode, device, generator
            )
            if include_root_ref_for_endpoints:
                _apply_indices(obs, mask, motion, b, keyframes, root_indices)
            _apply_indices(obs, mask, motion, b, keyframes, joint_pos_slice_for(endpoint_joints))
        elif mode == "fullpose":
            _apply_indices(obs, mask, motion, b, keyframes, root_indices)
            _apply_indices(obs, mask, motion, b, keyframes, fullpose_indices)
        elif mode == "mixed":
            endpoint_joints = _sample_endpoint_joints(
                endpoint_groups, endpoint_subset_mode, device, generator
            )
            _apply_indices(obs, mask, motion, b, keyframes, root_indices)
            _apply_indices(obs, mask, motion, b, keyframes, joint_pos_slice_for(endpoint_joints))
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
