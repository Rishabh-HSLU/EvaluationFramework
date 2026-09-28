"""SBBTS drift network s_theta(t, Y_t, Phi_theta(Y_{t0:ti})).

Architecture per Appendix B of the SBBTS paper (arXiv:2604.07159):

    - Time t and current state Y_t are each mapped into R^d_model by an
      independent FFN: linear -> layer norm -> SiLU -> linear.
    - The past trajectory Y_{t0:ti} is embedded into the latent space by a
      linear layer and encoded by a one-layer encoder-only transformer
      (causal-masked, so it never sees future positions).
    - The three embedded vectors are concatenated and mapped back to the
      output dimension d by a similar FFN.

One deliberate adaptation for our data (documented in reasoning.md): the
paper's reference implementation adds an index-based sinusoidal positional
encoding to the trajectory encoder, appropriate for its regular observation
grids. Our data is on an irregular grid (missing minutes are dropped, not
imputed), so position i does not correspond to a fixed time offset. We
replace the index-based positional signal with an elapsed-time embedding
(the same FFN pattern applied to each position's actual elapsed time),
consistent with the task requirement that time inputs reflect real elapsed
time rather than a step index.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FeedForwardEmbed(nn.Module):
    """linear -> layer norm -> SiLU -> linear, mapping in_dim to out_dim."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TrajectoryEncoder(nn.Module):
    """Phi_theta: encodes Y_{t0:ti} into a context vector c_i per position.

    A single forward pass over the full (padded) sequence produces one
    context vector per position; causal masking guarantees c_i depends only
    on positions <= i, and the padding mask guarantees padded positions are
    excluded from attention entirely (not merely zeroed).
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int = 1,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(state_dim, d_model)
        self.time_pos_embed = FeedForwardEmbed(1, d_model)
        # Every argument is passed explicitly; see SBBTSModelConfig for why
        # the encoder-layer internals are pinned rather than inherited.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, y_seq: Tensor, times: Tensor, valid_mask: Tensor) -> Tensor:
        """
        Args:
            y_seq: (B, L, state_dim) trajectory values.
            times: (B, L) actual elapsed time of each position.
            valid_mask: (B, L) bool, True where the position is a real
                observation (not padding).

        Returns:
            (B, L, d_model) context vectors; position i is Phi_theta(Y_{t0:ti}).
        """
        seq_len = y_seq.shape[1]
        h = self.input_proj(y_seq) + self.time_pos_embed(times.unsqueeze(-1))

        # Bool causal mask (True = disallowed), matching src_key_padding_mask's
        # dtype: PyTorch deprecates mixing a float attn mask with a bool
        # padding mask.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=y_seq.device), diagonal=1
        )
        key_padding_mask = ~valid_mask

        return self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)


class SBBTSDrift(nn.Module):
    """Full drift network: s_theta(t, Y_t, c_i) -> R^state_dim."""

    def __init__(
        self,
        state_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 1,
        encoder_dropout: float = 0.1,
        encoder_dim_feedforward: int = 2048,
        encoder_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model

        self.time_embed = FeedForwardEmbed(1, d_model)
        self.state_embed = FeedForwardEmbed(state_dim, d_model)
        self.trajectory_encoder = TrajectoryEncoder(
            state_dim,
            d_model,
            n_heads,
            n_layers,
            dropout=encoder_dropout,
            dim_feedforward=encoder_dim_feedforward,
            activation=encoder_activation,
        )
        self.output_proj = FeedForwardEmbed(3 * d_model, state_dim, hidden_dim=d_model)

    def encode_history(self, y_seq: Tensor, times: Tensor, valid_mask: Tensor) -> Tensor:
        """Phi_theta(Y_{t0:ti}) for every position i in one pass. See TrajectoryEncoder."""
        return self.trajectory_encoder(y_seq, times, valid_mask)

    def drift(self, t: Tensor, y_t: Tensor, context: Tensor) -> Tensor:
        """s_theta(t, Y_t, context).

        Args:
            t: (..., 1) actual elapsed time.
            y_t: (..., state_dim) current state.
            context: (..., d_model) context vector, typically a slice of
                `encode_history`'s output.

        Returns:
            (..., state_dim) estimated drift.
        """
        t_emb = self.time_embed(t)
        y_emb = self.state_embed(y_t)
        combined = torch.cat([t_emb, y_emb, context], dim=-1)
        return self.output_proj(combined)

    def forward(
        self, t: Tensor, y_t: Tensor, y_seq: Tensor, times: Tensor, valid_mask: Tensor
    ) -> Tensor:
        """Convenience full forward pass: encode history, then predict drift at (t, y_t).

        Uses the last valid position's context, i.e. Phi_theta(Y_{t0:tN}) where
        N is the last real (non-padded) index of each row.
        """
        context_seq = self.encode_history(y_seq, times, valid_mask)
        last_valid_idx = valid_mask.sum(dim=1).clamp(min=1) - 1  # (B,)
        batch_idx = torch.arange(context_seq.shape[0], device=context_seq.device)
        context = context_seq[batch_idx, last_valid_idx]  # (B, d_model)
        return self.drift(t, y_t, context)
