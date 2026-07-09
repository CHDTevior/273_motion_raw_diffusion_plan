from __future__ import annotations

import itertools
from argparse import Namespace

import pytest
import torch

from models.raw_motion.hy273_constraints import build_synthetic_control_batch
from models.raw_motion.hy273_slices import (
    DIM_HY273,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    KIMODO_EE_GROUPS,
    ROOT_SLICE,
    joint_pos_slice_for,
)
from train_hy273_raw_flow import validate_ema_contract, validate_resume_contract


def _selected_joint_ids(mask_at_frame: torch.Tensor) -> tuple[int, ...]:
    position_mask = mask_at_frame[JOINT_POS_SLICE].reshape(22, 3)
    assert torch.equal(position_mask, position_mask[:, :1].expand_as(position_mask))
    return tuple(torch.where(position_mask[:, 0])[0].tolist())


def _valid_kimodo_subsets() -> set[tuple[int, ...]]:
    expected_groups = ((7, 10), (8, 11), (20,), (21,))
    assert KIMODO_EE_GROUPS == expected_groups
    subsets: set[tuple[int, ...]] = set()
    for enabled in itertools.product((False, True), repeat=len(expected_groups)):
        if not any(enabled):
            continue
        subsets.add(
            tuple(sorted(
                joint_id
                for use_group, group in zip(enabled, expected_groups)
                if use_group
                for joint_id in group
            ))
        )
    return subsets


def test_kimodo_endpoint_all_masks_six_position_joints_and_root_reference() -> None:
    motion = torch.arange(12 * DIM_HY273, dtype=torch.float32).reshape(1, 12, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.tensor([12]),
        modes=("endpoints",),
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="all",
        min_keyframes=1,
        max_keyframes=1,
        generator=torch.Generator().manual_seed(7),
    )

    frames = torch.where(result.motion_mask[0].any(dim=-1))[0]
    assert frames.numel() == 1
    frame = int(frames.item())
    expected = torch.zeros(DIM_HY273, dtype=torch.bool)
    expected[ROOT_SLICE.start : HEADING_SLICE.stop] = True
    expected[joint_pos_slice_for(joint_id for group in KIMODO_EE_GROUPS for joint_id in group)] = True
    assert torch.equal(result.motion_mask[0, frame], expected)
    assert torch.equal(result.observed_motion[result.motion_mask], motion[result.motion_mask])
    assert torch.count_nonzero(result.observed_motion[~result.motion_mask]) == 0


def test_kimodo_random_endpoint_control_samples_only_nonempty_logical_groups() -> None:
    batch_size = 64
    motion = torch.randn(batch_size, 10, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.full((batch_size,), 10),
        modes=("endpoints",),
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="random_nonempty",
        min_keyframes=1,
        max_keyframes=1,
        generator=torch.Generator().manual_seed(11),
    )

    valid_subsets = _valid_kimodo_subsets()
    sampled: set[tuple[int, ...]] = set()
    for batch_idx in range(batch_size):
        frame = int(torch.where(result.motion_mask[batch_idx].any(dim=-1))[0].item())
        selected = _selected_joint_ids(result.motion_mask[batch_idx, frame])
        assert selected in valid_subsets
        sampled.add(selected)
        assert result.motion_mask[batch_idx, frame, ROOT_SLICE.start : HEADING_SLICE.stop].all()
    assert len(sampled) > 1


def test_endpoint_root_reference_can_only_be_disabled_explicitly() -> None:
    motion = torch.randn(1, 8, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.tensor([8]),
        modes=("endpoints",),
        endpoint_subset_mode="all",
        include_root_ref_for_endpoints=False,
        min_keyframes=1,
        max_keyframes=1,
        generator=torch.Generator().manual_seed(3),
    )
    frame = int(torch.where(result.motion_mask[0].any(dim=-1))[0].item())
    assert not result.motion_mask[0, frame, ROOT_SLICE.start : HEADING_SLICE.stop].any()
    assert _selected_joint_ids(result.motion_mask[0, frame])


def test_resume_contract_rejects_shape_preserving_text_semantic_change() -> None:
    requested = Namespace(
        data_root="/data",
        text_root="/text",
        max_frames=300,
        min_frames=16,
        prediction_type="x0",
        hidden_dim=1024,
        num_heads=8,
        depth_double=6,
        depth_single=12,
        mlp_ratio=2.0,
        dropout=0.0,
        text_encoder="hy_cache",
        max_text_tokens=128,
        hytext_cache_dir="/cache",
        hytext_ctxt_dim=4096,
        hytext_vtxt_dim=768,
        text_dropout_prob=0.1,
        random_first_heading=True,
        root_origin_shift=True,
        self_conditioning=False,
        self_cond_mode="add_proj",
        self_cond_scale=1.0,
    )
    saved = vars(requested).copy()
    validate_resume_contract(requested, saved, "checkpoint.pt")
    saved["max_text_tokens"] = 50
    with pytest.raises(RuntimeError, match="max_text_tokens"):
        validate_resume_contract(requested, saved, "checkpoint.pt")


def test_ema_contract_rejects_missing_parameter() -> None:
    model_state = {"a": torch.zeros(2), "b": torch.zeros(3)}
    with pytest.raises(RuntimeError, match="missing=.*b"):
        validate_ema_contract(model_state, {"a": torch.zeros(2)}, "checkpoint.pt")
