"""Train HY273 raw-space rectified-flow model."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text
from models.raw_motion.flow_schedule import (
    bce_logits_masked,
    build_flow_state,
    clean_from_velocity,
    mse_masked,
    sample_timesteps,
    smooth_l1_masked,
)
from models.raw_motion.hy273_constraints import build_synthetic_control_batch
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_kimodo_training_transform
from models.raw_motion.hy273_slices import (
    CONTACT_JOINTS,
    CONTACT_SLICE,
    CONT_DIM,
    ROOT_DIM,
    ROOT_SLICE,
    VELOCITY_SLICE,
    reconstruct_global_joints_from_features,
)
from models.raw_motion.raw_flow_dit import HY273RawFlow


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return data


def flatten_get(cfg: dict[str, Any], dotted: str, default: Any) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> tuple[torch.device, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size(), local_rank
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, -1


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def seed_all(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank) * 100003
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/raw_flow_hy273.yaml")
    p.add_argument("--data_root", default="")
    p.add_argument("--text_root", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", default="checkpoints/t2m/hy273_raw_flow")
    p.add_argument("--name", default="hy273_raw_flow")
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--min_frames", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=4000)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_every", type=int, default=1)
    p.add_argument("--time_schedule", choices=["uniform", "logit_normal"], default="logit_normal")
    p.add_argument("--denoiser_p_mean", type=float, default=0.0)
    p.add_argument("--denoiser_p_std", type=float, default=1.0)
    p.add_argument("--prediction_type", choices=["velocity", "x0"], default="x0")
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--depth_double", type=int, default=6)
    p.add_argument("--depth_single", type=int, default=12)
    p.add_argument("--mlp_ratio", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--text_encoder", choices=["clip", "hy_cache", "hytext_cache", "qwen_clip_cache", "none"], default="clip")
    p.add_argument("--clip_path", default="checkpoints/clip/ViT-B-32.pt")
    p.add_argument("--clip_version", default="ViT-B/32")
    p.add_argument("--max_text_tokens", type=int, default=50)
    p.add_argument("--hytext_cache_dir", default="")
    p.add_argument("--hytext_ctxt_dim", type=int, default=4096)
    p.add_argument("--hytext_vtxt_dim", type=int, default=768)
    p.add_argument("--hytext_max_open_shards", type=int, default=8)
    p.add_argument("--hytext_allow_cache_miss", action="store_true")
    p.add_argument("--text_dropout_prob", type=float, default=0.1)
    p.add_argument("--random_first_heading", action="store_true")
    p.add_argument("--no_random_first_heading", action="store_true")
    p.add_argument("--root_origin_shift", action="store_true")
    p.add_argument("--no_root_origin_shift", action="store_true")
    p.add_argument("--control_modes", default="none,root,endpoints,fullpose,mixed")
    p.add_argument("--max_control_keyframes", type=int, default=8)
    p.add_argument("--flow_loss_weight", type=float, default=1.0)
    p.add_argument("--contact_loss_weight", type=float, default=0.1)
    p.add_argument("--control_cont_loss_weight", type=float, default=0.25)
    p.add_argument("--control_contact_loss_weight", type=float, default=0.05)
    p.add_argument("--clean_cont_loss_weight", type=float, default=0.0)
    p.add_argument("--clean_root_vel_loss_weight", type=float, default=0.0)
    p.add_argument("--clean_joint_vel_loss_weight", type=float, default=0.0)
    p.add_argument("--foot_lock_loss_weight", type=float, default=0.0)
    p.add_argument("--semantic_loss_fps", type=float, default=30.0)
    p.add_argument("--foot_lock_contact_threshold", type=float, default=0.5)
    p.add_argument("--root_heading_loss_weight", type=float, default=1.0)
    p.add_argument("--velocity_loss_weight", type=float, default=1.0)
    p.add_argument("--self_conditioning", action="store_true")
    p.add_argument("--self_cond_train_prob", type=float, default=0.5)
    p.add_argument("--self_cond_mode", default="add_proj")
    p.add_argument("--self_cond_scale", type=float, default=1.0)
    return p


def merge_config(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "data_root": "data.data_root",
        "text_root": "data.text_root",
        "max_frames": "data.max_frames",
        "min_frames": "data.min_frames",
        "batch_size": "train.batch_size",
        "num_workers": "train.num_workers",
        "max_epochs": "train.max_epochs",
        "max_steps": "train.max_steps",
        "lr": "train.lr",
        "weight_decay": "train.weight_decay",
        "prediction_type": "model.prediction_type",
        "hidden_dim": "model.hidden_dim",
        "num_heads": "model.num_heads",
        "depth_double": "model.depth_double",
        "depth_single": "model.depth_single",
        "mlp_ratio": "model.mlp_ratio",
        "dropout": "model.dropout",
        "text_encoder": "text.encoder",
        "clip_path": "text.clip_path",
        "clip_version": "text.clip_version",
        "max_text_tokens": "text.max_text_tokens",
        "hytext_cache_dir": "text.hytext_cache_dir",
        "hytext_ctxt_dim": "text.hytext_ctxt_dim",
        "hytext_vtxt_dim": "text.hytext_vtxt_dim",
        "hytext_max_open_shards": "text.hytext_max_open_shards",
        "text_dropout_prob": "text.dropout_prob",
        "amp_dtype": "train.amp_dtype",
        "ema_decay": "ema.decay",
        "ema_every": "ema.every",
        "flow_loss_weight": "loss.flow",
        "contact_loss_weight": "loss.contact",
        "control_cont_loss_weight": "loss.control_cont",
        "control_contact_loss_weight": "loss.control_contact",
        "clean_cont_loss_weight": "loss.clean_cont",
        "clean_root_vel_loss_weight": "loss.clean_root_vel",
        "clean_joint_vel_loss_weight": "loss.clean_joint_vel",
        "foot_lock_loss_weight": "loss.foot_lock",
        "semantic_loss_fps": "loss.semantic_fps",
        "foot_lock_contact_threshold": "loss.foot_lock_contact_threshold",
        "root_heading_loss_weight": "loss.root_heading",
        "velocity_loss_weight": "loss.velocity",
        "control_modes": "control.modes",
        "max_control_keyframes": "control.max_keyframes",
        "self_cond_train_prob": "self_conditioning.train_prob",
        "self_cond_mode": "self_conditioning.mode",
        "self_cond_scale": "self_conditioning.scale",
    }
    parser_defaults = vars(build_arg_parser().parse_args([]))
    for attr, path in defaults.items():
        if getattr(args, attr) == parser_defaults[attr]:
            setattr(args, attr, flatten_get(cfg, path, getattr(args, attr)))
    if not args.self_conditioning:
        args.self_conditioning = bool(flatten_get(cfg, "self_conditioning.enabled", False))
    if not args.hytext_allow_cache_miss:
        args.hytext_allow_cache_miss = bool(flatten_get(cfg, "text.hytext_allow_cache_miss", False))
    if not args.amp:
        args.amp = bool(flatten_get(cfg, "train.amp", False))
    if not args.ema and not args.no_ema:
        args.ema = bool(flatten_get(cfg, "ema.enabled", False))
    if isinstance(args.control_modes, (list, tuple)):
        args.control_modes = ",".join(str(item) for item in args.control_modes)
    if not args.random_first_heading and not args.no_random_first_heading:
        args.random_first_heading = bool(flatten_get(cfg, "transform.random_first_heading", True))
    if not args.root_origin_shift and not args.no_root_origin_shift:
        args.root_origin_shift = bool(flatten_get(cfg, "transform.root_origin_shift", True))
    if args.no_random_first_heading:
        args.random_first_heading = False
    if args.no_root_origin_shift:
        args.root_origin_shift = False
    if args.no_ema:
        args.ema = False
    return args


def create_model(args: argparse.Namespace) -> HY273RawFlow:
    return HY273RawFlow(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth_double=args.depth_double,
        depth_single=args.depth_single,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        text_encoder=args.text_encoder,
        clip_path=args.clip_path,
        clip_version=args.clip_version,
        max_text_tokens=args.max_text_tokens,
        hytext_cache_dir=getattr(args, "hytext_cache_dir", ""),
        hytext_ctxt_dim=getattr(args, "hytext_ctxt_dim", 4096),
        hytext_vtxt_dim=getattr(args, "hytext_vtxt_dim", 768),
        hytext_max_open_shards=getattr(args, "hytext_max_open_shards", 8),
        hytext_strict_cache=not bool(getattr(args, "hytext_allow_cache_miss", False)),
        self_conditioning=args.self_conditioning,
        self_cond_mode=args.self_cond_mode,
        self_cond_scale=args.self_cond_scale,
    )


def continuous_loss_weights(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.ones(1, 1, CONT_DIM, device=device, dtype=dtype)
    weights[..., :ROOT_DIM] = float(args.root_heading_loss_weight)
    weights[..., VELOCITY_SLICE] = float(args.velocity_loss_weight)
    return weights


def weighted_mse_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(-1)
    weight = weight.to(device=pred.device, dtype=pred.dtype)
    denom = (mask_f * weight).sum().clamp_min(1.0)
    return ((pred - target).square() * mask_f * weight).sum() / denom


def weighted_smooth_l1_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(-1)
    weight = weight.to(device=pred.device, dtype=pred.dtype)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    denom = (mask_f * weight).sum().clamp_min(1.0)
    return (loss * mask_f * weight).sum() / denom


def predict_clean_cont(
    z_cont_imp: torch.Tensor,
    t: torch.Tensor,
    pred_cont: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "x0":
        return pred_cont
    if prediction_type == "velocity":
        return clean_from_velocity(z_cont_imp, t, pred_cont)
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


def prediction_velocity_cont(
    z_cont_imp: torch.Tensor,
    t: torch.Tensor,
    pred_cont: torch.Tensor,
    x0_hat_cont: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "velocity":
        return pred_cont
    if prediction_type == "x0":
        t_view = t.view(-1, 1, 1).to(device=z_cont_imp.device, dtype=z_cont_imp.dtype)
        return (x0_hat_cont - z_cont_imp) / (1.0 - t_view).clamp_min(1e-4)
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


def entries_smooth_l1_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(-1)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    denom = mask_f.expand_as(loss).sum().clamp_min(1.0)
    return (loss * mask_f).sum() / denom


def compute_clean_semantic_losses(
    x0_hat_un: torch.Tensor,
    x0_un: torch.Tensor,
    valid: torch.Tensor,
    fps: float,
    contact_threshold: float,
) -> dict[str, torch.Tensor]:
    zero = x0_hat_un.sum() * 0.0
    if x0_hat_un.shape[1] < 2:
        return {"clean_root_vel": zero, "clean_joint_vel": zero, "foot_lock": zero}

    valid_pair = valid[:, 1:] & valid[:, :-1]
    fps_f = float(fps)
    pred_root_vel = (x0_hat_un[:, 1:, ROOT_SLICE] - x0_hat_un[:, :-1, ROOT_SLICE]) * fps_f
    target_root_vel = (x0_un[:, 1:, ROOT_SLICE] - x0_un[:, :-1, ROOT_SLICE]) * fps_f
    clean_root_vel = entries_smooth_l1_masked(pred_root_vel, target_root_vel, valid_pair)

    pred_joints = reconstruct_global_joints_from_features(x0_hat_un)
    target_joints = reconstruct_global_joints_from_features(x0_un)
    pred_joint_vel = (pred_joints[:, 1:] - pred_joints[:, :-1]) * fps_f
    target_joint_vel = (target_joints[:, 1:] - target_joints[:, :-1]) * fps_f
    clean_joint_vel = entries_smooth_l1_masked(pred_joint_vel, target_joint_vel, valid_pair)

    contact_gt = x0_un[..., CONTACT_SLICE] > float(contact_threshold)
    contact_pair = contact_gt[:, 1:] & contact_gt[:, :-1] & valid_pair[..., None]
    pred_foot_vel = pred_joint_vel[:, :, list(CONTACT_JOINTS)]
    foot_lock = entries_smooth_l1_masked(pred_foot_vel, torch.zeros_like(pred_foot_vel), contact_pair)
    return {
        "clean_root_vel": clean_root_vel,
        "clean_joint_vel": clean_joint_vel,
        "foot_lock": foot_lock,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "step": int(step),
    }
    if ema_state is not None:
        payload["ema"] = {key: value.detach().cpu() for key, value in ema_state.items()}
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


@torch.no_grad()
def init_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        key: value.detach().clone().float() if torch.is_floating_point(value) else value.detach().clone()
        for key, value in raw_model.state_dict().items()
    }


@torch.no_grad()
def update_ema_state(ema_state: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    for key, value in raw_model.state_dict().items():
        if key not in ema_state:
            ema_state[key] = value.detach().clone().float() if torch.is_floating_point(value) else value.detach().clone()
            continue
        target = ema_state[key]
        value = value.detach()
        if torch.is_floating_point(value):
            target.mul_(float(decay)).add_(value.to(device=target.device, dtype=target.dtype), alpha=1.0 - float(decay))
        else:
            target.copy_(value.to(device=target.device))


def compute_step_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    normalizer: HY273Normalizer,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    x0_un = batch["motion"].to(device=device, dtype=torch.float32)
    lengths = batch["lengths"].to(device=device)
    valid = batch["valid"].to(device=device)
    transform = apply_kimodo_training_transform(
        x0_un,
        random_heading=args.random_first_heading,
        root_shift=args.root_origin_shift,
    )
    x0_un = transform.motion
    c_dir = transform.c_dir
    controls = build_synthetic_control_batch(
        x0_un,
        lengths=lengths,
        modes=tuple(m.strip() for m in args.control_modes.split(",") if m.strip()),
        max_keyframes=args.max_control_keyframes,
        include_root_ref_for_endpoints=True,
    )
    obs_un = controls.observed_motion
    motion_mask = controls.motion_mask.to(device=device)
    x0 = normalizer.normalize(x0_un)
    obs = normalizer.normalize(obs_un)
    timesteps = sample_timesteps(
        x0.shape[0],
        device=device,
        schedule=args.time_schedule,
        p_mean=args.denoiser_p_mean,
        p_std=args.denoiser_p_std,
    ).to(dtype=x0.dtype)
    state = build_flow_state(x0, obs, motion_mask, timesteps)
    model_in = state["model_in"]
    x_self_cond = None

    use_sc = bool(args.self_conditioning) and float(args.self_cond_train_prob) > 0.0
    if use_sc and torch.rand((), device=device) < float(args.self_cond_train_prob):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                pred0 = model(
                    model_in,
                    t=timesteps,
                    c_dir=c_dir,
                    text=batch["texts"],
                    length_mask=valid,
                    x_self_cond=None,
                    text_drop_prob=0.0,
                )
            x0_hat0_cont = predict_clean_cont(
                state["z_cont_imp"],
                timesteps,
                pred0[..., :CONT_DIM],
                args.prediction_type,
            )
            x0_hat0 = torch.cat([x0_hat0_cont, torch.sigmoid(pred0[..., CONTACT_SLICE])], dim=-1)
            mask_f = motion_mask.to(dtype=x0.dtype)
            x_self_cond = (x0_hat0 * (1.0 - mask_f) + obs * mask_f).detach()

    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        pred = model(
            model_in,
            t=timesteps,
            c_dir=c_dir,
            text=batch["texts"],
            length_mask=valid,
            x_self_cond=x_self_cond,
            text_drop_prob=args.text_dropout_prob,
        )
        pred_cont = pred[..., :CONT_DIM]
        contact_logits = pred[..., CONTACT_SLICE]
        x0_hat_cont = predict_clean_cont(
            state["z_cont_imp"],
            timesteps,
            pred_cont,
            args.prediction_type,
        )
        v_pred_cont = prediction_velocity_cont(
            state["z_cont_imp"],
            timesteps,
            pred_cont,
            x0_hat_cont,
            args.prediction_type,
        )
        mask_cont = motion_mask[..., :CONT_DIM].bool()
        mask_contact = motion_mask[..., CONTACT_SLICE].bool()
        valid_cont_unmasked = valid[..., None] & (~mask_cont)
        valid_contact = valid[..., None].expand_as(contact_logits)
        cont_weights = continuous_loss_weights(args, v_pred_cont.device, v_pred_cont.dtype)
        if args.prediction_type == "x0":
            flow_loss = weighted_mse_masked(x0_hat_cont, state["x0_cont"], valid_cont_unmasked, cont_weights)
        else:
            flow_loss = weighted_mse_masked(v_pred_cont, state["v_target_cont"], valid_cont_unmasked, cont_weights)
        contact_loss = bce_logits_masked(contact_logits, state["x0_contact"], valid_contact)
        control_cont = smooth_l1_masked(x0_hat_cont, obs[..., :CONT_DIM], valid[..., None] & mask_cont)
        control_contact = bce_logits_masked(contact_logits, obs[..., CONTACT_SLICE], valid[..., None] & mask_contact)
        clean_cont = (
            weighted_smooth_l1_masked(x0_hat_cont, state["x0_cont"], valid[..., None], cont_weights)
            if args.clean_cont_loss_weight > 0
            else flow_loss.detach() * 0.0
        )
        if args.prediction_type == "x0":
            root_heading_primary = mse_masked(
                x0_hat_cont[..., :ROOT_DIM],
                state["x0_cont"][..., :ROOT_DIM],
                valid_cont_unmasked[..., :ROOT_DIM],
            )
            velocity_channel_primary = mse_masked(
                x0_hat_cont[..., VELOCITY_SLICE],
                state["x0_cont"][..., VELOCITY_SLICE],
                valid_cont_unmasked[..., VELOCITY_SLICE],
            )
        else:
            root_heading_primary = mse_masked(
                v_pred_cont[..., :ROOT_DIM],
                state["v_target_cont"][..., :ROOT_DIM],
                valid_cont_unmasked[..., :ROOT_DIM],
            )
            velocity_channel_primary = mse_masked(
                v_pred_cont[..., VELOCITY_SLICE],
                state["v_target_cont"][..., VELOCITY_SLICE],
                valid_cont_unmasked[..., VELOCITY_SLICE],
            )
        x0_hat = torch.cat([x0_hat_cont, torch.sigmoid(contact_logits)], dim=-1)
        x0_hat_un = normalizer.denormalize(x0_hat.float())
        semantic_losses = compute_clean_semantic_losses(
            x0_hat_un,
            x0_un.float(),
            valid,
            fps=float(args.semantic_loss_fps),
            contact_threshold=float(args.foot_lock_contact_threshold),
        )
        loss = (
            args.flow_loss_weight * flow_loss
            + args.contact_loss_weight * contact_loss
            + args.control_cont_loss_weight * control_cont
            + args.control_contact_loss_weight * control_contact
            + args.clean_cont_loss_weight * clean_cont
            + args.clean_root_vel_loss_weight * semantic_losses["clean_root_vel"]
            + args.clean_joint_vel_loss_weight * semantic_losses["clean_joint_vel"]
            + args.foot_lock_loss_weight * semantic_losses["foot_lock"]
        )
    metrics = {
        "loss": float(loss.detach().float().item()),
        "flow": float(flow_loss.detach().float().item()),
        "contact": float(contact_loss.detach().float().item()),
        "control_cont": float(control_cont.detach().float().item()),
        "control_contact": float(control_contact.detach().float().item()),
        "clean_cont": float(clean_cont.detach().float().item()),
        "clean_root_vel": float(semantic_losses["clean_root_vel"].detach().float().item()),
        "clean_joint_vel": float(semantic_losses["clean_joint_vel"].detach().float().item()),
        "foot_lock": float(semantic_losses["foot_lock"].detach().float().item()),
        "root_heading_primary": float(root_heading_primary.detach().float().item()),
        "velocity_channel_primary": float(velocity_channel_primary.detach().float().item()),
        "mask_frac": float(motion_mask.float().mean().detach().item()),
        "prediction_type_x0": float(args.prediction_type == "x0"),
        "self_cond": float(x_self_cond is not None),
    }
    return loss, metrics


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    args = merge_config(args, cfg)
    if not args.data_root:
        raise ValueError("--data_root is required")
    device, rank, world, local_rank = setup_distributed()
    seed_all(args.seed, rank)
    out_dir = Path(args.output_dir) / args.name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_resolved.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    dataset = Kimodo273TextDataset(
        args.data_root,
        split=args.split,
        text_root=args.text_root or None,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        random_crop=True,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_kimodo273_text,
    )
    model = create_model(args).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    global_step = 0
    ema_state: dict[str, torch.Tensor] | None = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        if bool(args.ema) and "ema" in ckpt:
            ema_state = {
                key: value.to(device=device) if torch.is_tensor(value) else value
                for key, value in ckpt["ema"].items()
            }
    if bool(args.ema) and ema_state is None:
        ema_state = init_ema_state(model)
    normalizer = HY273Normalizer.from_data_root(args.data_root).to(device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")

    try:
        for epoch in range(start_epoch, int(args.max_epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute_step_loss(
                    model,
                    batch,
                    normalizer,
                    args,
                    device,
                    amp_enabled=bool(args.amp and device.type == "cuda"),
                    amp_dtype=amp_dtype,
                )
                if not torch.isfinite(loss.detach()).all():
                    raise RuntimeError(f"Non-finite loss at step={global_step}: {metrics}")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                if ema_state is not None and int(args.ema_every) > 0 and global_step % int(args.ema_every) == 0:
                    update_ema_state(ema_state, model, float(args.ema_decay))
                global_step += 1
                if rank == 0 and (global_step == 1 or global_step % args.log_every == 0):
                    msg = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                    print(f"[train] epoch={epoch} step={global_step} {msg}", flush=True)
                if rank == 0 and args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        out_dir / "model" / f"step_{global_step:08d}.pt",
                        model,
                        optimizer,
                        args,
                        epoch,
                        global_step,
                        ema_state=ema_state,
                    )
                    save_checkpoint(
                        out_dir / "model" / "latest.pt",
                        model,
                        optimizer,
                        args,
                        epoch,
                        global_step,
                        ema_state=ema_state,
                    )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if rank == 0:
            save_checkpoint(out_dir / "model" / "latest.pt", model, optimizer, args, epoch + 1, global_step, ema_state=ema_state)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
