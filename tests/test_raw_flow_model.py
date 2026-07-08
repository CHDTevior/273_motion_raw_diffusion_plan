import json

import numpy as np
import torch

from models.raw_motion.flow_schedule import build_flow_state, clean_from_velocity
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_yaw_rotation, transform_equivariance_error
from models.raw_motion.hytext_cache import hytext_key
from models.raw_motion.raw_flow_dit import HY273RawFlow
from train_hy273_raw_flow import compute_clean_semantic_losses, predict_clean_cont


def _model(self_conditioning=False):
    return HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        mlp_ratio=2.0,
        text_encoder="none",
        self_conditioning=self_conditioning,
    )


def _write_fake_hytext_cache(tmp_path):
    cache_dir = tmp_path / "hytext_cache"
    shard_dir = cache_dir / "shards" / "shard_00000"
    shard_dir.mkdir(parents=True)
    texts = ["", "walk", "run"]
    ctxt = np.random.randn(len(texts), 5, 16).astype(np.float16)
    vtxt = np.random.randn(len(texts), 1, 8).astype(np.float16)
    ctxt_len = np.array([1, 4, 3], dtype=np.int16)
    np.save(shard_dir / "ctxt.npy", ctxt)
    np.save(shard_dir / "vtxt.npy", vtxt)
    np.save(shard_dir / "ctxt_len.npy", ctxt_len)
    index = {
        hytext_key(text): {"shard": "shard_00000", "row": row, "text": text}
        for row, text in enumerate(texts)
    }
    (cache_dir / "index.json").write_text(json.dumps(index))
    (cache_dir / "manifest.json").write_text(json.dumps({"format": "hytext_memmap_v1"}))
    return cache_dir


def test_forward_without_self_conditioning_shape_finite():
    model = _model(False)
    x = torch.randn(2, 8, 546)
    out = model(x, t=torch.rand(2), text=["walk", "run"], length_mask=torch.ones(2, 8, dtype=torch.bool))
    assert out.shape == (2, 8, 273)
    assert torch.isfinite(out).all()


def test_forward_with_self_conditioning_shape_finite():
    model = _model(True)
    x = torch.randn(2, 8, 546)
    sc = torch.randn(2, 8, 273)
    out = model(
        x,
        t=torch.rand(2),
        text=["walk", "run"],
        length_mask=torch.ones(2, 8, dtype=torch.bool),
        x_self_cond=sc,
    )
    assert out.shape == (2, 8, 273)
    assert torch.isfinite(out).all()


def test_forward_with_cached_hytext_shape_finite(tmp_path):
    cache_dir = _write_fake_hytext_cache(tmp_path)
    model = HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        mlp_ratio=2.0,
        text_encoder="hy_cache",
        max_text_tokens=5,
        hytext_cache_dir=str(cache_dir),
        hytext_ctxt_dim=16,
        hytext_vtxt_dim=8,
    )
    x = torch.randn(2, 8, 546)
    out = model(x, t=torch.rand(2), text=["walk", "run"], length_mask=torch.ones(2, 8, dtype=torch.bool))
    out_drop = model(
        x,
        t=torch.rand(2),
        text=["walk", "run"],
        length_mask=torch.ones(2, 8, dtype=torch.bool),
        force_drop_text=True,
    )
    assert out.shape == (2, 8, 273)
    assert out_drop.shape == (2, 8, 273)
    assert torch.isfinite(out).all()
    assert torch.isfinite(out_drop).all()


def test_flow_state_uses_imputed_clean_estimate_contract():
    x0 = torch.randn(2, 5, 273)
    x0[..., 269:273] = (torch.rand(2, 5, 4) > 0.5).float()
    obs = torch.zeros_like(x0)
    mask = torch.zeros_like(x0, dtype=torch.bool)
    mask[:, 0, :3] = True
    obs[mask] = x0[mask]
    t = torch.full((2,), 0.4)
    state = build_flow_state(x0, obs, mask, t)
    v = torch.randn(2, 5, 269)
    clean = clean_from_velocity(state["z_cont_imp"], t, v)
    assert clean.shape == (2, 5, 269)
    assert torch.allclose(state["z_imp"][:, 0, :3], obs[:, 0, :3])
    assert float(state["z_imp"][..., 269:273].min()) >= 0.0
    assert float(state["z_imp"][..., 269:273].max()) <= 1.0


def test_x0_prediction_contract_returns_clean_prediction():
    z = torch.randn(2, 5, 269)
    t = torch.rand(2)
    pred_x0 = torch.randn(2, 5, 269)
    clean = predict_clean_cont(z, t, pred_x0, "x0")
    assert torch.equal(clean, pred_x0)


def test_contacts_not_normalized():
    mean = torch.randn(273)
    std = torch.rand(273) + 0.1
    norm = HY273Normalizer(mean, std)
    x = torch.randn(2, 4, 273)
    x[..., 269:273] = (torch.rand(2, 4, 4) > 0.5).float()
    y = norm.normalize(x)
    z = norm.denormalize(y)
    assert torch.equal(y[..., 269:273], x[..., 269:273])
    assert torch.allclose(z, x, atol=1e-5)


def test_yaw_rotation_feature_global_positions_equivariant():
    x = torch.zeros(1, 6, 273)
    x[..., 3] = 1.0
    joints = torch.randn(1, 6, 22, 3)
    x[..., 5:71] = joints.reshape(1, 6, 66)
    angle = torch.tensor([0.5])
    err = transform_equivariance_error(x, angle)
    assert float(err) < 1e-5


def test_clean_velocity_losses_zero_for_matching_clean_motion():
    x = torch.randn(2, 6, 273)
    x[..., 3] = 1.0
    x[..., 4] = 0.0
    x[..., 269:273] = (torch.rand(2, 6, 4) > 0.5).float()
    valid = torch.ones(2, 6, dtype=torch.bool)
    losses = compute_clean_semantic_losses(x, x.clone(), valid, fps=30.0, contact_threshold=0.5)
    assert float(losses["clean_root_vel"]) == 0.0
    assert float(losses["clean_joint_vel"]) == 0.0
    assert torch.isfinite(losses["foot_lock"])
