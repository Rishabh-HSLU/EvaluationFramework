"""SBBTS drift-network tests: shapes, causal masking, valid_mask masking.

Fixtures are directly-constructed `SessionExample` records fed through the
real `scripts.sbbts_batching.build_batches`, the same fixture idiom
test_sbbts_batching.py uses -- not a standalone synthetic corpus/bucketing
substitute (see scripts/reasoning.md for why that was deleted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sbbts.data import to_model_inputs
from sbbts.model import SBBTSDrift
from scripts.sbbts_batching import build_batches
from scripts.sbbts_panel import GAP_DTYPE, VALUE_DTYPE, SessionExample

torch.manual_seed(0)


def _example(index: int, length: int, gaps: np.ndarray | None = None) -> SessionExample:
    rng = np.random.default_rng(index)
    values = np.empty(length, dtype=VALUE_DTYPE)
    values[0] = 0.0
    values[1:] = np.cumsum(rng.normal(0.0, 1.0, size=length - 1))
    if gaps is None:
        gaps = rng.integers(1, 5, size=length).astype(GAP_DTYPE)
    return SessionExample(
        ticker=f"T{index:04d}",
        session=pd.Timestamp("2026-01-05", tz="UTC"),
        values=values,
        gaps=gaps,
    )


def _tiny_model() -> SBBTSDrift:
    # Small dims so tests run fast; architecture shape is otherwise identical
    # to the paper's config (same modules, same masking mechanism).
    return SBBTSDrift(state_dim=1, d_model=16, n_heads=4, n_layers=1)


def test_forward_pass_shape_from_bucketing_pipeline() -> None:
    examples = [_example(i, length) for i, length in enumerate([10, 12, 15, 20, 8, 9, 30, 25])]
    batches = build_batches(examples, batch_size=4, n_buckets=2)
    assert batches, "expected at least one batch from the real bucketing pipeline"

    model = _tiny_model()
    batch = batches[0]
    y, times, valid_mask = to_model_inputs(batch)
    b, seq_len, state_dim = y.shape
    assert state_dim == 1

    context_seq = model.encode_history(y, times, valid_mask)
    assert context_seq.shape == (b, seq_len, model.d_model)

    t_query = times[:, -1:]
    y_query = y[:, -1, :]
    out = model.forward(t_query, y_query, y, times, valid_mask)
    assert out.shape == (b, state_dim)
    assert torch.isfinite(out).all()


def test_causal_mask_blocks_future_positions() -> None:
    model = _tiny_model()
    model.eval()

    b, seq_len = 2, 6
    y = torch.randn(b, seq_len, 1)
    times = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
    valid_mask = torch.ones(b, seq_len, dtype=torch.bool)

    y_perturbed = y.clone()
    y_perturbed[:, seq_len - 1, :] += 100.0  # perturb only the last (future) position

    with torch.no_grad():
        out_original = model.encode_history(y, times, valid_mask)
        out_perturbed = model.encode_history(y_perturbed, times, valid_mask)

    # Every position strictly before the perturbed one must be unaffected:
    # the causal mask prevents it from attending to a future position.
    torch.testing.assert_close(out_original[:, :-1, :], out_perturbed[:, :-1, :])
    # The perturbed position itself must change (sanity: the model is
    # actually using its own input, not silently ignoring it).
    assert not torch.allclose(out_original[:, -1, :], out_perturbed[:, -1, :])


def test_valid_mask_is_respected_not_just_zeroing() -> None:
    """Perturbing a padded position must not change any real position's output."""
    model = _tiny_model()
    model.eval()

    real_len = 4
    padded_len = 7
    y = torch.zeros(1, padded_len, 1)
    y[0, :real_len, 0] = torch.randn(real_len)
    times = torch.zeros(1, padded_len)
    times[0, :real_len] = torch.arange(real_len, dtype=torch.float32)
    valid_mask = torch.zeros(1, padded_len, dtype=torch.bool)
    valid_mask[0, :real_len] = True

    y_perturbed = y.clone()
    # Perturb a padded position with a large, non-zero value -- if the
    # implementation merely zeroed padding instead of excluding it from
    # attention, this would leak into the real positions' outputs.
    y_perturbed[0, real_len:, 0] = 1e6

    with torch.no_grad():
        out_original = model.encode_history(y, times, valid_mask)
        out_perturbed = model.encode_history(y_perturbed, times, valid_mask)

    torch.testing.assert_close(out_original[:, :real_len, :], out_perturbed[:, :real_len, :])


def test_to_model_inputs_uses_actual_elapsed_time_from_gaps() -> None:
    """Time input must be real elapsed time (cumulative gaps), not a step index.

    gaps[0] is the first survivor's distance from the session open; gaps[i]
    for i >= 1 is the elapsed time since the previous surviving bar. The
    cumulative sum recovers each position's actual elapsed time since open.
    """
    gaps = np.array([5, 1, 1, 3, 1], dtype=GAP_DTYPE)  # a real gap (3) mid-sequence
    example = _example(0, length=5, gaps=gaps)
    batches = build_batches([example], batch_size=1, n_buckets=1)
    assert len(batches) == 1

    _, times, _ = to_model_inputs(batches[0])
    expected = np.cumsum(gaps)
    np.testing.assert_allclose(times[0].numpy(), expected)
    # Not a step index: consecutive positions are not all 1 apart.
    diffs = np.diff(times[0].numpy())
    assert (diffs == 1.0).sum() < len(diffs)
