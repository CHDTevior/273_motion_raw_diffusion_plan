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
from sample_hy273_raw import resolve_endpoint_protocol
from train_hy273_raw_flow import (
    build_arg_parser,
    explicit_cli_destinations,
    merge_config,
    validate_ema_contract,
    validate_resume_contract,
)


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
    requested = build_arg_parser().parse_args([])
    requested.data_root = "/data"
    requested.text_root = "/text"
    requested.text_encoder = "hy_cache"
    requested.max_text_tokens = 128
    requested.hytext_cache_dir = "/cache"
    requested.random_first_heading = True
    requested.root_origin_shift = True
    saved = vars(requested).copy()
    validate_resume_contract(requested, saved, "checkpoint.pt")
    for field, changed in (
        ("max_text_tokens", 50),
        ("split", "val"),
        ("time_schedule", "uniform"),
        ("hytext_allow_cache_miss", True),
    ):
        incompatible = saved.copy()
        incompatible[field] = changed
        with pytest.raises(RuntimeError, match=field):
            validate_resume_contract(requested, incompatible, "checkpoint.pt")


def test_ema_contract_rejects_missing_parameter() -> None:
    model_state = {"a": torch.zeros(2), "b": torch.zeros(3)}
    with pytest.raises(RuntimeError, match="missing=.*b"):
        validate_ema_contract(model_state, {"a": torch.zeros(2)}, "checkpoint.pt")


def test_explicit_cli_default_value_overrides_conflicting_yaml() -> None:
    parser = build_arg_parser()
    argv = [
        "--control_modes",
        "none,root,endpoints,fullpose,mixed",
        "--endpoint_subset_mode",
        "random_nonempty",
    ]
    args = parser.parse_args(argv)
    cfg = {
        "control": {
            "modes": ["none"],
            "endpoint_subset_mode": "all",
        }
    }
    merged = merge_config(
        args,
        cfg,
        explicit_cli=explicit_cli_destinations(parser, argv),
    )
    assert merged.control_modes == "none,root,endpoints,fullpose,mixed"
    assert merged.endpoint_subset_mode == "random_nonempty"


def test_training_cli_rejects_abbreviated_options() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--endpoint_subset", "random_nonempty"])


def test_sampling_reuses_checkpoint_endpoint_protocol_and_allows_override() -> None:
    checkpoint_args = Namespace(
        endpoint_preset="five_point",
        endpoint_subset_mode="all",
        endpoint_root_ref_mode="none",
        max_control_keyframes=5,
    )
    inherited = resolve_endpoint_protocol(checkpoint_args)
    assert inherited == {
        "endpoint_preset": "five_point",
        "endpoint_subset_mode": "all",
        "endpoint_root_ref_mode": "none",
        "max_control_keyframes": 5,
        "include_root_ref_for_endpoints": False,
    }
    overridden = resolve_endpoint_protocol(
        checkpoint_args,
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="random_nonempty",
        endpoint_root_ref_mode="kimodo_hidden_root",
        max_control_keyframes=8,
    )
    assert overridden["endpoint_preset"] == "kimodo_ee"
    assert overridden["endpoint_subset_mode"] == "random_nonempty"
    assert overridden["include_root_ref_for_endpoints"] is True
