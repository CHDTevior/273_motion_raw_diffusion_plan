"""Sample HY273 raw-space flow checkpoints with step-wise source-domain clamping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text
from models.raw_motion.flow_schedule import make_ode_grid
from models.raw_motion.hy273_constraints import build_synthetic_control_batch
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_kimodo_training_transform
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM, HEADING_SLICE, ROOT_SLICE
from train_hy273_raw_flow import (
    create_model,
    predict_clean_cont,
    prediction_velocity_cont,
    resolve_representation_loss_space,
)


@torch.no_grad()
def sample_ode(
    model: torch.nn.Module,
    normalizer: HY273Normalizer,
    lengths: torch.Tensor,
    texts: list[str],
    observed_un: torch.Tensor,
    motion_mask: torch.Tensor,
    c_dir: torch.Tensor,
    num_steps: int = 32,
    self_conditioning: bool = False,
    cfg_scale: float = 1.0,
    contact_init: str = "random",
    contact_feedback: str = "blend",
    cfg_apply_contacts: bool = False,
    prediction_type: str = "velocity",
    velocity_t_eps: float = 1e-4,
) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    observed = normalizer.normalize(observed_un.to(device=device, dtype=torch.float32)).to(dtype=dtype)
    motion_mask = motion_mask.to(device=device, dtype=dtype)
    lengths = lengths.to(device=device)
    bsz, frames, _ = observed.shape
    valid = torch.arange(frames, device=device)[None, :] < lengths[:, None]
    z_cont = torch.randn(bsz, frames, CONT_DIM, device=device, dtype=dtype)
    if contact_init == "zeros":
        contact_noise = torch.zeros(bsz, frames, 4, device=device, dtype=dtype)
    elif contact_init == "half":
        contact_noise = torch.full((bsz, frames, 4), 0.5, device=device, dtype=dtype)
    elif contact_init == "random":
        contact_noise = torch.rand(bsz, frames, 4, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown contact_init: {contact_init}")
    if contact_feedback not in {"blend", "prob", "fixed"}:
        raise ValueError(f"Unknown contact_feedback: {contact_feedback}")
    contact_aux = contact_noise.clone()
    x_self_cond: Optional[torch.Tensor] = None
    grid = make_ode_grid(num_steps, device=device).to(dtype=dtype)
    for i in range(num_steps):
        t = grid[i].expand(bsz)
        dt = grid[i + 1] - grid[i]
        state = torch.cat([z_cont, contact_aux], dim=-1)
        state = state * (1.0 - motion_mask) + observed * motion_mask
        model_in = torch.cat([state, motion_mask], dim=-1)
        pred = model(
            model_in,
            t=t,
            c_dir=c_dir.to(device=device, dtype=dtype),
            text=texts,
            length_mask=valid,
            x_self_cond=x_self_cond,
            text_drop_prob=0.0,
        )
        if abs(float(cfg_scale) - 1.0) > 1e-6:
            pred_uncond = model(
                model_in,
                t=t,
                c_dir=c_dir.to(device=device, dtype=dtype),
                text=texts,
                length_mask=valid,
                x_self_cond=x_self_cond,
                text_drop_prob=0.0,
                force_drop_text=True,
            )
            pred_cfg = pred_uncond + float(cfg_scale) * (pred - pred_uncond)
            if cfg_apply_contacts:
                pred = pred_cfg
            else:
                pred = pred.clone()
                pred[..., :CONT_DIM] = pred_cfg[..., :CONT_DIM]
        pred_cont = pred[..., :CONT_DIM]
        contact_prob = torch.sigmoid(pred[..., CONTACT_SLICE])
        x0_hat_cont = predict_clean_cont(state[..., :CONT_DIM], t, pred_cont, prediction_type)
        v_cont = prediction_velocity_cont(
            state[..., :CONT_DIM],
            t,
            pred_cont,
            x0_hat_cont,
            prediction_type,
            velocity_t_eps=velocity_t_eps,
        )
        x0_hat = torch.cat([x0_hat_cont, contact_prob], dim=-1)
        x0_hat_clamped = x0_hat * (1.0 - motion_mask) + observed * motion_mask
        z_cont_next = state[..., :CONT_DIM] + dt * v_cont
        next_state = torch.cat([z_cont_next, contact_prob], dim=-1)
        next_state = next_state * (1.0 - motion_mask) + observed * motion_mask
        z_cont = next_state[..., :CONT_DIM]
        if contact_feedback == "blend":
            t_next = grid[i + 1].view(1, 1, 1).to(device=device, dtype=dtype)
            contact_aux = t_next * contact_prob + (1.0 - t_next) * contact_noise
        elif contact_feedback == "prob":
            contact_aux = next_state[..., CONTACT_SLICE]
        else:
            contact_aux = contact_noise
        x_self_cond = x0_hat_clamped.detach() if self_conditioning else None
    final = torch.cat([z_cont, contact_aux], dim=-1)
    final = final * (1.0 - motion_mask) + observed * motion_mask
    out = normalizer.denormalize(final.float())
    for batch_idx in range(bsz):
        length = int(lengths[batch_idx].clamp(min=1, max=frames).item())
        if length < frames:
            out[batch_idx, length:] = out[batch_idx, length - 1 : length]
    return out


def anchor_first_frame_convention(
    observed_un: torch.Tensor,
    motion_mask: torch.Tensor,
    c_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor the coordinate convention learned from root_origin_shift/c_dir."""
    observed_un = observed_un.clone()
    motion_mask = motion_mask.clone()
    observed_un[:, 0, ROOT_SLICE.start] = 0.0
    observed_un[:, 0, ROOT_SLICE.start + 2] = 0.0
    motion_mask[:, 0, ROOT_SLICE.start] = True
    motion_mask[:, 0, ROOT_SLICE.start + 2] = True
    observed_un[:, 0, HEADING_SLICE] = c_dir.to(device=observed_un.device, dtype=observed_un.dtype)
    motion_mask[:, 0, HEADING_SLICE] = True
    return observed_un, motion_mask


def resolve_endpoint_protocol(
    train_args: argparse.Namespace,
    endpoint_preset: str = "",
    endpoint_subset_mode: str = "",
    endpoint_root_ref_mode: str = "",
    max_control_keyframes: int = 0,
) -> dict[str, object]:
    root_ref_mode = endpoint_root_ref_mode or str(
        getattr(train_args, "endpoint_root_ref_mode", "kimodo_hidden_root")
    )
    return {
        "endpoint_preset": endpoint_preset
        or str(getattr(train_args, "endpoint_preset", "kimodo_ee")),
        "endpoint_subset_mode": endpoint_subset_mode
        or str(getattr(train_args, "endpoint_subset_mode", "random_nonempty")),
        "endpoint_root_ref_mode": root_ref_mode,
        "max_control_keyframes": int(
            max_control_keyframes
            or int(getattr(train_args, "max_control_keyframes", 8))
        ),
        "include_root_ref_for_endpoints": root_ref_mode == "kimodo_hidden_root",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--text_root", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="generation/hy273_raw_flow")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated dataset indices to sample. Defaults to the first --num_samples items.",
    )
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--contact_init", choices=["random", "zeros", "half"], default="random")
    parser.add_argument("--contact_feedback", choices=["blend", "prob", "fixed"], default="blend")
    parser.add_argument("--cfg_apply_contacts", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--weight_source", choices=["auto", "model", "ema"], default="auto")
    parser.add_argument("--control_modes", default="root,endpoints,fullpose,mixed")
    parser.add_argument("--endpoint_preset", choices=["kimodo_ee", "five_point"], default="")
    parser.add_argument(
        "--endpoint_subset_mode",
        choices=["all", "random_nonempty"],
        default="",
    )
    parser.add_argument(
        "--endpoint_root_ref_mode",
        choices=["kimodo_hidden_root", "none"],
        default="",
    )
    parser.add_argument("--max_control_keyframes", type=int, default=0)
    parser.add_argument("--c_dir_mode", choices=["dataset", "forward"], default="dataset")
    parser.add_argument("--anchor_first_frame", action="store_true")
    parser.add_argument("--text_encoder", choices=["clip", "hy_cache", "hytext_cache", "qwen_clip_cache", "none"], default="")
    parser.add_argument("--hytext_cache_dir", default="")
    parser.add_argument("--hytext_ctxt_dim", type=int, default=0)
    parser.add_argument("--hytext_vtxt_dim", type=int, default=0)
    parser.add_argument("--hytext_max_open_shards", type=int, default=0)
    parser.add_argument("--hytext_allow_cache_miss", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt.get("args", {}))
    if args.data_root:
        train_args.data_root = args.data_root
    if args.text_root:
        train_args.text_root = args.text_root
    if args.text_encoder:
        train_args.text_encoder = args.text_encoder
    if args.hytext_cache_dir:
        train_args.hytext_cache_dir = args.hytext_cache_dir
    if args.hytext_ctxt_dim > 0:
        train_args.hytext_ctxt_dim = args.hytext_ctxt_dim
    if args.hytext_vtxt_dim > 0:
        train_args.hytext_vtxt_dim = args.hytext_vtxt_dim
    if args.hytext_max_open_shards > 0:
        train_args.hytext_max_open_shards = args.hytext_max_open_shards
    if args.hytext_allow_cache_miss:
        train_args.hytext_allow_cache_miss = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(train_args).to(device)
    weight_source = args.weight_source
    if weight_source == "auto":
        weight_source = "ema" if "ema" in ckpt else "model"
    if weight_source == "ema" and "ema" not in ckpt:
        raise ValueError(f"--weight_source ema requested but checkpoint has no EMA: {args.checkpoint}")
    state_dict = ckpt["ema"] if weight_source == "ema" else ckpt["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    normalizer = HY273Normalizer.from_data_root(train_args.data_root).to(device)
    dataset = Kimodo273TextDataset(
        train_args.data_root,
        split=args.split,
        text_root=train_args.text_root or None,
        max_frames=train_args.max_frames,
        random_crop=False,
        deterministic_text=True,
    )
    if args.indices:
        sample_indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
        if not sample_indices:
            raise ValueError("--indices was provided but no valid indices were parsed")
        for idx in sample_indices:
            if idx < 0 or idx >= len(dataset):
                raise IndexError(f"Dataset index {idx} out of range for split {args.split} with {len(dataset)} items")
    else:
        sample_indices = list(range(min(args.num_samples, len(dataset))))
    samples = [dataset[i] for i in sample_indices]
    batch = collate_kimodo273_text(samples)
    motion = batch["motion"].to(device)
    lengths = batch["lengths"].to(device)
    transform = apply_kimodo_training_transform(motion, random_heading=False, root_shift=True)
    motion = transform.motion
    endpoint_protocol = resolve_endpoint_protocol(
        train_args,
        endpoint_preset=args.endpoint_preset,
        endpoint_subset_mode=args.endpoint_subset_mode,
        endpoint_root_ref_mode=args.endpoint_root_ref_mode,
        max_control_keyframes=args.max_control_keyframes,
    )
    controls = build_synthetic_control_batch(
        motion,
        lengths,
        modes=tuple(m.strip() for m in args.control_modes.split(",") if m.strip()),
        endpoint_preset=str(endpoint_protocol["endpoint_preset"]),
        endpoint_subset_mode=str(endpoint_protocol["endpoint_subset_mode"]),
        max_keyframes=int(endpoint_protocol["max_control_keyframes"]),
        include_root_ref_for_endpoints=bool(
            endpoint_protocol["include_root_ref_for_endpoints"]
        ),
    )
    if args.c_dir_mode == "forward":
        c_dir = torch.zeros(motion.shape[0], 2, device=device, dtype=motion.dtype)
        c_dir[:, 0] = 1.0
    else:
        c_dir = transform.c_dir.to(device)
    if args.anchor_first_frame:
        controls.observed_motion, controls.motion_mask = anchor_first_frame_convention(
            controls.observed_motion,
            controls.motion_mask,
            c_dir,
        )
    prediction_type = str(getattr(train_args, "prediction_type", "velocity"))
    representation_loss_space = resolve_representation_loss_space(
        prediction_type,
        str(getattr(train_args, "representation_loss_space", "auto")),
    )
    # The JiT epsilon caps the training loss weight only. ODE integration must
    # retain the rectified-flow vector field and uses only a numerical floor.
    sampling_velocity_t_eps = 1e-4
    pred = sample_ode(
        model,
        normalizer,
        lengths,
        batch["texts"],
        controls.observed_motion,
        controls.motion_mask,
        c_dir,
        num_steps=args.num_steps,
        self_conditioning=bool(getattr(train_args, "self_conditioning", False)),
        cfg_scale=float(args.cfg_scale),
        contact_init=args.contact_init,
        contact_feedback=args.contact_feedback,
        cfg_apply_contacts=bool(args.cfg_apply_contacts),
        prediction_type=prediction_type,
        velocity_t_eps=sampling_velocity_t_eps,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "samples.npy", pred.cpu().numpy())
    np.save(out_dir / "observed.npy", controls.observed_motion.cpu().numpy())
    np.save(out_dir / "mask.npy", controls.motion_mask.cpu().numpy())
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64, copy=False)
    np.save(out_dir / "lengths.npy", lengths_np)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "num_steps": args.num_steps,
                "cfg_scale": float(args.cfg_scale),
                "contact_init": args.contact_init,
                "contact_feedback": args.contact_feedback,
                "cfg_apply_contacts": bool(args.cfg_apply_contacts),
                "seed": int(args.seed),
                "weight_source": weight_source,
                "prediction_type": str(getattr(train_args, "prediction_type", "velocity")),
                "representation_loss_space": representation_loss_space,
                "velocity_loss_t_eps": float(
                    getattr(train_args, "velocity_loss_t_eps", 0.05)
                ),
                "sampling_velocity_t_eps": sampling_velocity_t_eps,
                "lengths": lengths_np.tolist(),
                "texts": batch["texts"],
                "rel_paths": batch["rel_paths"],
                "indices": sample_indices,
                "control_modes": controls.mode_ids,
                "endpoint_protocol": endpoint_protocol,
                "c_dir_mode": args.c_dir_mode,
                "anchor_first_frame": bool(args.anchor_first_frame),
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
