"""Generate SBBTS sample paths from a trained drift network.

Implements the sampling procedure the paper gives after Algorithm 1
(Section 4) and the reference implementation's `generate_dsbm_batch`
(diffusion_dsbm.py), on this repository's elapsed-minute time grid:

    Y_{t0}     = X_{t0} - s(t0, X_{t0}, Phi(X_{t0})) / beta
    on [t_i, t_{i+1}):  dY = s(t, Y, c_i) dt + dW      (N_pi Euler-Maruyama steps)
    X_{t_{i+1}} = Y_{t_{i+1}} + s(t~_{i+1}, Y_{t_{i+1}}, c_i) / beta

with c_i the encoder's context for the path up to t_i. The paper writes the
generation-time context as Phi(Y_{t0:ti}) and the reference code feeds the
auxiliary Y path to the encoder while generating (training conditions on
the observed X path; the two coincide in the large-beta regime the method
assumes). `context_on` selects which path the encoder sees; "y" is the
reference behaviour.

The inverse map is evaluated at t~_{i+1} = t_{i+1} - safe_frac * Delta_t,
where `sbbts.training.endpoint_transform` evaluates the frozen transform
during training (the paper's t~ = t_{i+1} - xi); the Euler drift
evaluations stay inside [t_i, t_i + (1 - 1/N_pi) Delta_t], i.e. within the
range the regression covered.

Paths are generated on the full regular 1-minute session grid (every bar
present, Delta_t = 1), anchored at X = 0 on the first bar with elapsed
time 1, exactly as `scripts.sbbts_panel` anchors training examples. Real's
missing-bar mask is imposed afterwards by `assemble_price_panel`, the same
way the GBM and MSV controls are built.

The trajectory encoder is one causal post-norm TransformerEncoderLayer, so
its output at position i depends only on inputs 0..i; `CachedCausalEncoder`
evaluates it one position at a time with a key/value cache, which turns
the O(N^2) prefix re-encoding per bar into O(N) and is verified against
`SBBTSDrift.encode_history` in the tests.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import SBBTSDrift, TrajectoryEncoder


class CachedCausalEncoder:
    """Incremental evaluation of a one-layer causal TrajectoryEncoder (eval mode)."""

    def __init__(self, encoder: TrajectoryEncoder) -> None:
        layers = encoder.encoder.layers
        if len(layers) != 1:
            raise NotImplementedError("Cached evaluation is implemented for n_layers == 1 only.")
        layer = layers[0]
        if layer.norm_first:
            raise NotImplementedError("Expected the default post-norm TransformerEncoderLayer.")
        if encoder.encoder.norm is not None:
            raise NotImplementedError("Expected no final norm on the TransformerEncoder.")
        self.encoder = encoder
        self.layer = layer
        self.attn = layer.self_attn
        self.n_heads = self.attn.num_heads
        self.head_dim = self.attn.head_dim
        self.keys: list[Tensor] = []
        self.values: list[Tensor] = []

    def reset(self) -> None:
        self.keys.clear()
        self.values.clear()

    @torch.no_grad()
    def step(self, y_i: Tensor, t_i: Tensor) -> Tensor:
        """Append position i and return its context vector, (B, d_model).

        Args:
            y_i: (B, state_dim) value at position i.
            t_i: (B,) elapsed time at position i.
        """
        x = self.encoder.input_proj(y_i) + self.encoder.time_pos_embed(t_i.unsqueeze(-1))
        batch = x.shape[0]
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (batch, self.n_heads, 1, self.head_dim)
        q, k, v = (a.reshape(shape) for a in (q, k, v))
        self.keys.append(k)
        self.values.append(v)
        keys = torch.cat(self.keys, dim=2)  # (B, H, i+1, hd)
        values = torch.cat(self.values, dim=2)
        scores = (q @ keys.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, H, 1, i+1)
        attended = (torch.softmax(scores, dim=-1) @ values).reshape(batch, -1)  # (B, d_model)
        attended = self.attn.out_proj(attended)
        x = self.layer.norm1(x + attended)
        ff = self.layer.linear2(self.layer.activation(self.layer.linear1(x)))
        return self.layer.norm2(x + ff)


@torch.no_grad()
def generate_paths(
    model: SBBTSDrift,
    n_paths: int,
    n_bars: int,
    beta: float,
    n_euler: int = 50,
    context_on: str = "y",
    generator: torch.Generator | None = None,
    retransform_start: bool = False,
    invert_iters: int = 0,
    safe_frac: float = 0.01,
) -> Tensor:
    """Sample `n_paths` standardized cumulative paths of `n_bars` bars, (n_paths, n_bars, state_dim).

    Bar j sits at elapsed time j + 1 (minutes since the session open);
    bar 0 is the anchor X = 0.

    Two optional departures from the reference sampler make generation
    follow `compute_loss`'s definitions exactly instead of their large-beta
    approximation:

    - `retransform_start`: start interval i from
      Y_i = X_i - s(t_i, X_i, c_i)/beta (the training definition, with the
      interval's own context c_i) rather than carrying over the auxiliary
      endpoint of interval i-1.
    - `invert_iters` > 0: recover X_{i+1} by iterating the fixed point
      X = Y + s(t_{i+1}, X, c_i)/beta that many times (training evaluates
      the endpoint map at X, not at Y); 0 uses the reference's one-shot
      X = Y + s(t_{i+1}, Y, c_i)/beta.
    """
    if context_on not in ("y", "x"):
        raise ValueError("context_on must be 'y' (reference behaviour) or 'x'.")
    model.eval()
    device = next(model.parameters()).device
    state_dim = model.state_dim
    cache = CachedCausalEncoder(model.trajectory_encoder)
    cache.reset()

    out = torch.zeros(n_paths, n_bars, state_dim, device=device)
    x = torch.zeros(n_paths, state_dim, device=device)  # X_{t0} = 0
    t0 = torch.ones(n_paths, device=device)
    c0 = cache.step(x, t0)  # Phi(X_{t0}); the encoder's first input is X_{t0} either way
    y = x - model.drift(t0.unsqueeze(-1), x, c0) / beta  # Y_{t0}
    dt = 1.0 / n_euler
    sqrt_dt = math.sqrt(dt)

    context = c0
    for i in range(n_bars - 1):
        if i > 0:
            past_value = y if context_on == "y" else x
            context = cache.step(past_value, t0 + i)
        t_i = t0 + i
        if retransform_start and i > 0:
            y = x - model.drift(t_i.unsqueeze(-1), x, context) / beta
        for m in range(n_euler):
            tau = (t_i + m * dt).unsqueeze(-1)
            noise = torch.randn(y.shape, device=device, generator=generator)
            y = y + model.drift(tau, y, context) * dt + sqrt_dt * noise
        t_tilde = (t_i + 1.0 - safe_frac).unsqueeze(-1)  # t~_{i+1}, Delta_t = 1
        x = y + model.drift(t_tilde, y, context) / beta  # X_{t_{i+1}}
        for _ in range(invert_iters):
            x = y + model.drift(t_tilde, x, context) / beta
        out[:, i + 1] = x
    return out
