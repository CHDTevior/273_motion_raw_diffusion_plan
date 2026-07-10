import torch

from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.raw_flow_dit import HY273RawFlow
from sample_hy273_raw import sample_ode


def test_sample_ode_clamps_observed_dims():
    model = HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        text_encoder="none",
        self_conditioning=True,
    )
    mean = torch.zeros(273)
    std = torch.ones(273)
    normalizer = HY273Normalizer(mean, std)
    obs = torch.zeros(2, 6, 273)
    mask = torch.zeros_like(obs, dtype=torch.bool)
    obs[:, 0, :5] = 3.0
    mask[:, 0, :5] = True
    lengths = torch.tensor([6, 6])
    out = sample_ode(
        model,
        normalizer,
        lengths,
        ["walk", "run"],
        obs,
        mask,
        c_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        num_steps=2,
        self_conditioning=True,
        prediction_type="x0",
    )
    assert out.shape == (2, 6, 273)
    assert torch.allclose(out[:, 0, :5].cpu(), obs[:, 0, :5])


def test_sample_ode_uses_numerical_velocity_epsilon():
    model = HY273RawFlow(
        hidden_dim=64,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        text_encoder="none",
    )
    normalizer = HY273Normalizer(torch.zeros(273), torch.ones(273))
    out = sample_ode(
        model,
        normalizer,
        torch.tensor([4]),
        ["walk"],
        torch.zeros(1, 4, 273),
        torch.zeros(1, 4, 273, dtype=torch.bool),
        c_dir=torch.tensor([[1.0, 0.0]]),
        num_steps=2,
        prediction_type="x0",
        velocity_t_eps=1e-4,
    )
    assert out.shape == (1, 4, 273)
    assert torch.isfinite(out).all()
