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
    global_rot_slice_for,
    joint_pos_slice_for,
)

ControlMode = Literal[
    "none", "root", "root_sparse", "root_dense", "endpoints", "fullpose", "mixed"
]
KimodoPattern = Literal["root_sparse", "root_dense", "endpoints", "fullpose"]
EndpointPreset = Literal["kimodo_ee", "five_point"]
EndpointSubsetMode = Literal["all", "random_nonempty"]


@dataclass
class ControlBatch:
    observed_motion: torch.Tensor
    motion_mask: torch.Tensor
    mode_ids: list[str]


@dataclass(frozen=True)
class KimodoControlCurriculum:
    none_prob: float = 0.10
    mixed_prob: float = 0.25
    max_sparse_keyframes: int = 20
    dense_min_fraction: float = 0.25
    endpoint_preset: EndpointPreset = "five_point"
    endpoint_subset_mode: EndpointSubsetMode = "random_nonempty"
    include_root_ref_for_endpoints: bool = True
    include_endpoint_rotations: bool = False

    def validate(self) -> None:
        if self.none_prob < 0 or self.mixed_prob < 0 or self.none_prob + self.mixed_prob > 1:
            raise ValueError("Expected non-negative none/mixed probabilities summing to at most one")
        if self.max_sparse_keyframes < 1:
            raise ValueError("max_sparse_keyframes must be positive")
        if not 0 < self.dense_min_fraction <= 1:
            raise ValueError("dense_min_fraction must be in (0,1]")


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


def _low_biased_keyframe_count(
    max_count: int,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> int:
    if max_count <= 1:
        return 1
    draw = torch.rand((), device=device, generator=generator)
    return 1 + min(int((draw.square() * max_count).floor().item()), max_count - 1)


def _dense_interval_frames(
    length: int,
    min_fraction: float,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    min_span = max(1, min(length, int(round(length * float(min_fraction)))))
    if min_span >= length:
        return torch.arange(length, device=device, dtype=torch.long)
    span = int(torch.randint(min_span, length + 1, (1,), device=device, generator=generator).item())
    start = int(torch.randint(0, length - span + 1, (1,), device=device, generator=generator).item())
    return torch.arange(start, start + span, device=device, dtype=torch.long)


def build_kimodo_control_curriculum_batch(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    progress: float,
    config: KimodoControlCurriculum,
    generator: Optional[torch.Generator] = None,
) -> ControlBatch:
    """Build project-domain Kimodo-like constraints from the paired clean motion."""
    if motion.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected motion [B,T,{DIM_HY273}], got {tuple(motion.shape)}")
    config.validate()
    progress = min(max(float(progress), 0.0), 1.0)
    current_kmax = 1 + int(progress * (config.max_sparse_keyframes - 1))
    bsz = motion.shape[0]
    device = motion.device
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    mode_ids: list[str] = []
    patterns: tuple[KimodoPattern, ...] = (
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
    )
    root_path_indices = [ROOT_SLICE.start, ROOT_SLICE.start + 2] + list(
        range(HEADING_SLICE.start, HEADING_SLICE.stop)
    )
    root_full_indices = list(range(ROOT_SLICE.start, HEADING_SLICE.stop))
    fullpose_indices = list(range(JOINT_POS_SLICE.start, JOINT_POS_SLICE.stop))
    endpoint_groups = _endpoint_groups(config.endpoint_preset)

    for batch_idx in range(bsz):
        length = max(1, int(lengths[batch_idx].item()))
        mode_draw = float(torch.rand((), device=device, generator=generator).item())
        if mode_draw < config.none_prob:
            chosen: tuple[KimodoPattern, ...] = ()
            mode_ids.append("none")
        elif mode_draw < config.none_prob + config.mixed_prob:
            order = torch.randperm(len(patterns), device=device, generator=generator)
            chosen = (patterns[int(order[0].item())], patterns[int(order[1].item())])
            mode_ids.append(f"mixed:{chosen[0]}+{chosen[1]}")
        else:
            chosen = (
                patterns[int(torch.randint(0, len(patterns), (1,), device=device, generator=generator).item())],
            )
            mode_ids.append(chosen[0])

        for pattern in chosen:
            if pattern == "root_dense":
                frames = _dense_interval_frames(
                    length,
                    config.dense_min_fraction,
                    device,
                    generator,
                )
                _apply_indices(obs, mask, motion, batch_idx, frames, root_path_indices)
                continue

            count = _low_biased_keyframe_count(min(current_kmax, length), device, generator)
            frames = _select_frames(length, count, device, generator=generator)
            if pattern == "root_sparse":
                _apply_indices(obs, mask, motion, batch_idx, frames, root_path_indices)
            elif pattern == "fullpose":
                _apply_indices(obs, mask, motion, batch_idx, frames, root_full_indices)
                _apply_indices(obs, mask, motion, batch_idx, frames, fullpose_indices)
            elif pattern == "endpoints":
                endpoint_joints = _sample_endpoint_joints(
                    endpoint_groups,
                    config.endpoint_subset_mode,
                    device,
                    generator,
                )
                if config.include_root_ref_for_endpoints:
                    _apply_indices(obs, mask, motion, batch_idx, frames, root_full_indices)
                _apply_indices(
                    obs,
                    mask,
                    motion,
                    batch_idx,
                    frames,
                    joint_pos_slice_for(endpoint_joints),
                )
                if config.include_endpoint_rotations:
                    _apply_indices(
                        obs,
                        mask,
                        motion,
                        batch_idx,
                        frames,
                        global_rot_slice_for(endpoint_joints),
                    )
            else:
                raise AssertionError(f"Unhandled control pattern: {pattern}")
    return ControlBatch(observed_motion=obs, motion_mask=mask, mode_ids=mode_ids)


def build_synthetic_control_batch(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    modes: Sequence[ControlMode] = (
        "none",
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
        "mixed",
    ),
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
    root_path_indices = [ROOT_SLICE.start, ROOT_SLICE.start + 2] + list(
        range(HEADING_SLICE.start, HEADING_SLICE.stop)
    )
    root_full_indices = list(range(ROOT_SLICE.start, HEADING_SLICE.stop))
    fullpose_indices = list(range(JOINT_POS_SLICE.start, JOINT_POS_SLICE.stop))
    available_patterns: tuple[KimodoPattern, ...] = (
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
    )

    for b in range(bsz):
        mode = modes[int(torch.randint(0, len(modes), (1,), device=device, generator=generator).item())]
        length = int(lengths[b].item())
        if mode == "none":
            mode_names.append("none")
            continue
        if mode == "mixed":
            order = torch.randperm(len(available_patterns), device=device, generator=generator)
            selected_patterns = (
                available_patterns[int(order[0].item())],
                available_patterns[int(order[1].item())],
            )
            mode_names.append(f"mixed:{selected_patterns[0]}+{selected_patterns[1]}")
        elif mode == "root":
            selected_patterns = ("root_sparse",)
            mode_names.append("root_sparse")
        elif mode in available_patterns:
            selected_patterns = (mode,)
            mode_names.append(str(mode))
        else:
            raise ValueError(f"Unknown control mode: {mode}")

        for pattern in selected_patterns:
            if pattern == "root_dense":
                dense_frames = _dense_interval_frames(length, 0.25, device, generator)
                _apply_indices(obs, mask, motion, b, dense_frames, root_path_indices)
                continue
            k = int(
                torch.randint(
                    min_keyframes,
                    max_keyframes + 1,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()
            )
            keyframes = _select_frames(length, k, device, generator=generator)
            if pattern == "root_sparse":
                _apply_indices(obs, mask, motion, b, keyframes, root_path_indices)
            elif pattern == "endpoints":
                endpoint_joints = _sample_endpoint_joints(
                    endpoint_groups, endpoint_subset_mode, device, generator
                )
                if include_root_ref_for_endpoints:
                    _apply_indices(obs, mask, motion, b, keyframes, root_full_indices)
                _apply_indices(
                    obs,
                    mask,
                    motion,
                    b,
                    keyframes,
                    joint_pos_slice_for(endpoint_joints),
                )
            elif pattern == "fullpose":
                _apply_indices(obs, mask, motion, b, keyframes, root_full_indices)
                _apply_indices(obs, mask, motion, b, keyframes, fullpose_indices)
            else:
                raise AssertionError(f"Unhandled v1 control pattern: {pattern}")
    return ControlBatch(observed_motion=obs, motion_mask=mask, mode_ids=mode_names)


def compile_root_control(motion: torch.Tensor, frames: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    f = torch.as_tensor(list(frames), device=motion.device, dtype=torch.long)
    idx = torch.as_tensor(
        [ROOT_SLICE.start, ROOT_SLICE.start + 2, HEADING_SLICE.start, HEADING_SLICE.start + 1],
        device=motion.device,
        dtype=torch.long,
    )
    obs[f[:, None], idx[None, :]] = motion[f[:, None], idx[None, :]]
    mask[f[:, None], idx[None, :]] = True
    return obs, mask
