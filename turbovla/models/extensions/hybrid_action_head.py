"""A's ACT head with signed-PLIF residual FFNs, not an all-spike head.

Attention, state projection, action queries and continuous regression keep A's
implementation. Only each decoder's FFN branch changes. T is local neuron time;
the H action positions remain parallel and do not carry state between calls.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..action_head import TurboVLAActionHead
from ..components.wta_operators import BipolarPLIF
from ..configuration import ActionHeadConfig


class HybridSpikeFFNDecoderLayer(nn.TransformerDecoderLayer):
    """Keep the native pre-norm attention/residual paths; replace only FFN."""

    def __init__(self, config: ActionHeadConfig, hidden_dim: int,
                 nheads: int, dim_feedforward: int) -> None:
        super().__init__(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=dim_feedforward,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.timesteps = int(config.spike_timesteps)
        self.ffn_input_spike = BipolarPLIF()
        self.ffn_hidden_norm = nn.LayerNorm(dim_feedforward)
        self.ffn_hidden_spike = BipolarPLIF()

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # The native decoder already applied norm3 to [B,H,C]. Both linears
        # consume unit signed events. No inverted dropout scales those events;
        # A's output dropout remains on the continuous temporal mean instead.
        input_membrane = hidden_membrane = None
        outputs = []
        for _ in range(self.timesteps):
            input_events, input_membrane = self.ffn_input_spike(x, input_membrane)
            hidden_current = self.ffn_hidden_norm(self.linear1(input_events))
            hidden_events, hidden_membrane = self.ffn_hidden_spike(hidden_current, hidden_membrane)
            outputs.append(self.linear2(hidden_events))
        return self.dropout3(torch.stack(outputs, dim=0).mean(dim=0))


class HybridSpikeFFNActionHead(TurboVLAActionHead):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int,
                 nheads: int, dim_feedforward: int) -> None:
        super().__init__(config, hidden_dim, nheads, dim_feedforward)
        self.gradient_checkpointing = bool(config.spike_gradient_checkpointing)
        self.decoder.decoder.layers = nn.ModuleList([
            HybridSpikeFFNDecoderLayer(config, hidden_dim, nheads, dim_feedforward)
            for _ in range(config.num_layers)
        ])

    def load_common(self, original: TurboVLAActionHead) -> None:
        """Copy the same newly constructed A head, never a trained policy."""
        self.state_projection.load_state_dict(original.state_projection.state_dict())
        self.decoder.action_queries.load_state_dict(original.decoder.action_queries.state_dict())
        self.decoder.action_projection.load_state_dict(original.decoder.action_projection.state_dict())
        for target, source in zip(self.decoder.decoder.layers, original.decoder.decoder.layers):
            for name in ("self_attn", "multihead_attn", "linear1", "linear2", "norm1", "norm2", "norm3"):
                getattr(target, name).load_state_dict(getattr(source, name).state_dict())

    def forward(self, vision_language_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # PLIF state is call-local and this head has no running-stat norms.
            return checkpoint(super().forward, vision_language_tokens, state, use_reentrant=False)
        return super().forward(vision_language_tokens, state)
