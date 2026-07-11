"""Lightweight control evaluation for HY273 raw-flow checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text
from models.raw_motion.hy273_constraints import (
    KimodoControlCurriculum,
    build_kimodo_control_curriculum_batch,
)
from models.raw_motion.hy273_normalizer import apply_kimodo_training_transform
from models.raw_motion.hy273_slices import (
    CONTACT_JOINTS,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)
from sample_hy273_raw import (
    ODESampleOutput,
    checkpoint_normalizer,
    checkpoint_weight_state,
    apply_checkpoint_path_override,
    resolve_endpoint_protocol,
    sample_ode,
    verify_checkpoint_assets,
)
from train_hy273_raw_flow import create_model


def masked_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    value = (pred - target).square().sum(dim=-1).sqrt()
    mask = mask.squeeze(-1).bool()
    if not mask.any():
        return value.new_tensor(0.0)
    return value[mask].mean()


def foot_skate_metric(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    joints = reconstruct_global_joints_from_features(features)
    contact = features[..., 269:273] > 0.5
    feet = joints[:, :, list(CONTACT_JOINTS)]
    vel = feet[:, 1:] - feet[:, :-1]
    contact_mid = contact[:, 1:] & contact[:, :-1]
    valid = torch.arange(features.shape[1] - 1, device=features.device)[None, :] < (lengths[:, None] - 1)
    mask = contact_mid & valid[..., None]
    if not mask.any():
        return features.new_tensor(0.0)
    return vel.norm(dim=-1)[mask].mean()


def phase2_distribution_conditioned_on_control(
    none_prob: float, mixed_prob: float
) -> float:
    controlled_mass = 1.0 - float(none_prob)
    if controlled_mass <= 0:
        raise ValueError("Checkpoint control distribution has no controlled examples")
    if not 0 <= float(mixed_prob) <= controlled_mass:
        raise ValueError("Invalid Phase-2 none/mixed probabilities")
    return float(mixed_prob) / controlled_mass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="")
    p.add_argument("--text_root", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=32)
    p.add_argument("--output", default="")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--cfg_scale", type=float, default=3.5)
    p.add_argument("--control_cfg_scale", type=float, default=2.0)
    p.add_argument("--curriculum_progress", type=float, default=1.0)
    p.add_argument("--weight_source", choices=["ema", "model", "auto"], default="ema")
    p.add_argument(
        "--text_encoder",
        choices=["clip", "hy_cache", "hytext_cache", "qwen_clip_cache", "none"],
        default="",
    )
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt.get("args", {}))
    apply_checkpoint_path_override(train_args, "data_root", args.data_root)
    apply_checkpoint_path_override(train_args, "text_root", args.text_root)
    if args.text_encoder:
        train_args.text_encoder = args.text_encoder
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    verify_checkpoint_assets(train_args)
    model = create_model(train_args).to(device)
    state_dict, weight_source = checkpoint_weight_state(
        ckpt, args.weight_source, args.checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    normalizer = checkpoint_normalizer(ckpt, train_args, device, args.checkpoint)
    endpoint_protocol = resolve_endpoint_protocol(train_args)
    train_none_prob = float(getattr(train_args, "control_none_prob", 0.10))
    train_mixed_prob = float(getattr(train_args, "control_mixed_prob", 0.25))
    controlled_mixed_prob = phase2_distribution_conditioned_on_control(
        train_none_prob, train_mixed_prob
    )
    dataset = Kimodo273TextDataset(
        train_args.data_root,
        split=args.split,
        text_root=train_args.text_root or None,
        max_frames=train_args.max_frames,
        random_crop=False,
        deterministic_text=True,
    )
    totals = {
        f"{prefix}_{metric}": 0.0
        for prefix in ("raw", "exact")
        for metric in (
            "endpoint_err_feature",
            "endpoint_err_fk",
            "fk_consistency_err",
            "root_xz_err_feature",
            "foot_skate",
        )
    }
    count = 0
    mode_counts: Counter[str] = Counter()
    for start in range(0, min(args.max_samples, len(dataset)), args.batch_size):
        samples = [dataset[i] for i in range(start, min(start + args.batch_size, args.max_samples, len(dataset)))]
        batch = collate_kimodo273_text(samples)
        gt = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        transform = apply_kimodo_training_transform(gt, random_heading=False, root_shift=True)
        gt = transform.motion
        controls = build_kimodo_control_curriculum_batch(
            gt,
            lengths,
            progress=float(args.curriculum_progress),
            config=KimodoControlCurriculum(
                none_prob=0.0,
                mixed_prob=controlled_mixed_prob,
                max_sparse_keyframes=int(endpoint_protocol["max_control_keyframes"]),
                dense_min_fraction=float(
                    getattr(train_args, "control_dense_min_fraction", 0.25)
                ),
                endpoint_preset=str(endpoint_protocol["endpoint_preset"]),
                endpoint_subset_mode=str(endpoint_protocol["endpoint_subset_mode"]),
                include_root_ref_for_endpoints=bool(
                    endpoint_protocol["include_root_ref_for_endpoints"]
                ),
                include_endpoint_rotations=False,
            ),
            generator=torch.Generator(device=device).manual_seed(int(args.seed) + start),
        )
        mode_counts.update(controls.mode_ids)
        sampled = sample_ode(
            model,
            normalizer,
            lengths,
            batch["texts"],
            controls.observed_motion,
            controls.motion_mask,
            transform.c_dir,
            num_steps=args.num_steps,
            self_conditioning=bool(getattr(train_args, "self_conditioning", False)),
            cfg_scale=float(args.cfg_scale),
            control_cfg_scale=float(args.control_cfg_scale),
            prediction_type=str(getattr(train_args, "prediction_type", "x0")),
            return_details=True,
        )
        assert isinstance(sampled, ODESampleOutput)
        gt_feat_joints = reconstruct_global_joints_from_features(gt)
        endpoint_mask = controls.motion_mask[..., 5:71].reshape(gt.shape[0], gt.shape[1], 22, 3).any(dim=-1)
        root_xz_mask = controls.motion_mask[..., [0, 2]].any(dim=-1)
        valid_joint = (torch.arange(gt.shape[1], device=device)[None, :] < lengths[:, None])[..., None].expand_as(endpoint_mask)
        for prefix, pred in (
            ("raw", sampled.raw_motion),
            ("exact", sampled.exact_clamped_motion),
        ):
            pred_feat_joints = reconstruct_global_joints_from_features(pred)
            pred_fk_joints = fk_positions_from_global_rot6d(pred)
            totals[f"{prefix}_endpoint_err_feature"] += float(
                masked_l2(pred_feat_joints, gt_feat_joints, endpoint_mask).item()
            )
            totals[f"{prefix}_endpoint_err_fk"] += float(
                masked_l2(pred_fk_joints, gt_feat_joints, endpoint_mask).item()
            )
            totals[f"{prefix}_fk_consistency_err"] += float(
                masked_l2(pred_fk_joints, pred_feat_joints, valid_joint).item()
            )
            totals[f"{prefix}_root_xz_err_feature"] += float(
                masked_l2(pred[..., [0, 2]], gt[..., [0, 2]], root_xz_mask).item()
            )
            totals[f"{prefix}_foot_skate"] += float(foot_skate_metric(pred, lengths).item())
        count += 1
    metrics = {k: v / max(count, 1) for k, v in totals.items()}
    metrics.update(
        {
            "checkpoint": args.checkpoint,
            "num_steps": args.num_steps,
            "seed": int(args.seed),
            "cfg_scale": float(args.cfg_scale),
            "control_cfg_scale": float(args.control_cfg_scale),
            "weight_source": weight_source,
            "batches": count,
            "endpoint_protocol": endpoint_protocol,
            "control_mode_counts": dict(sorted(mode_counts.items())),
            "distribution": {
                "protocol": "phase2_conditioned_on_controlled",
                "training_none_prob": train_none_prob,
                "training_mixed_prob": train_mixed_prob,
                "evaluation_none_prob": 0.0,
                "evaluation_mixed_prob": controlled_mixed_prob,
            },
        }
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
