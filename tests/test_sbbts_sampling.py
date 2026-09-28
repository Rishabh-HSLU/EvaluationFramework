"""Sampling tests: the cached causal encoder matches encode_history; paths are well-formed."""

from __future__ import annotations

import torch

from sbbts.model import SBBTSDrift
from sbbts.sampling import CachedCausalEncoder, generate_paths

torch.manual_seed(0)


def test_cached_encoder_matches_full_causal_encoder() -> None:
    model = SBBTSDrift(state_dim=1, d_model=32, n_heads=4, n_layers=1).eval()
    batch, length = 5, 17
    y = torch.randn(batch, length, 1)
    times = torch.cumsum(torch.randint(1, 4, (batch, length)).float(), dim=-1)
    mask = torch.ones(batch, length, dtype=torch.bool)

    with torch.no_grad():
        full = model.encode_history(y, times, mask)  # (B, L, d_model), causal
    cache = CachedCausalEncoder(model.trajectory_encoder)
    cache.reset()
    stepwise = torch.stack([cache.step(y[:, i], times[:, i]) for i in range(length)], dim=1)

    assert torch.allclose(stepwise, full, atol=1e-5, rtol=1e-5)


def test_cached_encoder_ignores_future_like_the_causal_mask() -> None:
    """Position i's context must not change when later inputs change."""
    model = SBBTSDrift(state_dim=1, d_model=32, n_heads=4, n_layers=1).eval()
    y = torch.randn(3, 10, 1)
    times = torch.arange(1, 11).float().expand(3, 10)
    mask = torch.ones(3, 10, dtype=torch.bool)
    y_alt = y.clone()
    y_alt[:, 6:] += 5.0
    with torch.no_grad():
        a = model.encode_history(y, times, mask)[:, :6]
        b = model.encode_history(y_alt, times, mask)[:, :6]
    assert torch.allclose(a, b, atol=1e-6)


def test_generate_paths_shape_anchor_and_finiteness() -> None:
    model = SBBTSDrift(state_dim=1, d_model=16, n_heads=4, n_layers=1)
    gen = torch.Generator().manual_seed(1)
    paths = generate_paths(model, n_paths=6, n_bars=12, beta=2.0, n_euler=5, generator=gen)
    assert paths.shape == (6, 12, 1)
    assert torch.all(paths[:, 0] == 0.0)
    assert torch.isfinite(paths).all()
    assert not torch.allclose(paths[:, 1:], torch.zeros_like(paths[:, 1:]))

    gen2 = torch.Generator().manual_seed(1)
    again = generate_paths(model, n_paths=6, n_bars=12, beta=2.0, n_euler=5, generator=gen2)
    assert torch.equal(paths, again)  # deterministic given the generator

    other = generate_paths(
        model, 6, 12, 2.0, n_euler=5, context_on="x", generator=torch.Generator().manual_seed(1)
    )
    assert other.shape == paths.shape
