"""Continuous state MLP with a signed-PLIF/WTA action-head body.

This is our adaptation of the existing event operators, not the original AA
SDSA head. Both attention routes use binary multi-winner forward routing over
signed events; their softmax exists only in the routing surrogate backward.
Token-local LayerNorm replaces head temporal BN. All learned body projections
consume events; membrane residuals and the final temporal action readout remain
continuous. Neuron time is local to an observation, not physical history.
"""
from __future__ import annotations

import torch
from torch import nn

from .components.wta_operators import BipolarPLIF, MultiWinnerAttention
from .spike_action_head import (
    SpikeACTDecoder, SpikeAttention, SpikeDecoderLayer, SpikeTurboVLAActionHead,
)


class TokenCurrentNorm(nn.LayerNorm):
    """The event-head norm interface, without batch/time running statistics."""

    def forward(self, current, valid=None, time_index=0):
        output = super().forward(current)
        return output if valid is None else output.masked_fill(~valid, 0.0)


class WTAActionAttention(SpikeAttention):
    def __init__(self, hidden_dim, nheads, interaction, topk):
        super().__init__(hidden_dim, nheads, interaction)
        self.norms = nn.ModuleDict({
            name: TokenCurrentNorm(hidden_dim, eps=interaction.spike_norm_eps)
            for name in self.norms
        })
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in self.norms})
        self.routing = MultiWinnerAttention(topk, self.head_dim, interaction.wta_surrogate_temperature)

    def forward_step(self, query, memory, memory_valid, state, t):
        query_valid = torch.ones((*query.shape[:2], 1), dtype=torch.bool, device=query.device)

        def fire(name, current, valid):
            spikes, state[name] = self.spike_nodes[name](
                self.norms[name](current, valid, t), state.get(name), valid=valid,
            )
            return spikes

        sq = fire("input_query", query, query_valid)
        sm = fire("input_memory", memory, memory_valid)
        q = fire("query", self.query(sq), query_valid)
        k = fire("key", self.key(sm), memory_valid)
        v = fire("value", self.value(sm), memory_valid)
        batch, nq, channels = q.shape
        nk = k.shape[1]
        q = q.reshape(batch, nq, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        allowed = query_valid & memory_valid.transpose(-1, -2)
        route = self.routing(q @ k.transpose(-1, -2), allowed, q, k)
        context = (route @ v).transpose(1, 2).reshape(batch, nq, channels)
        live = query_valid & route.detach().ne(0).any(-1).any(1).unsqueeze(-1)
        return self.output(fire("context", context, live)).masked_fill(~live, 0.0)


class DeepSpikeDecoderLayer(SpikeDecoderLayer):
    def __init__(self, hidden_dim, nheads, dim_feedforward, interaction):
        super().__init__(hidden_dim, nheads, dim_feedforward, interaction)
        self.self_attn = WTAActionAttention(
            hidden_dim, nheads, interaction, topk=interaction.wta_text_topk,
        )
        self.cross_attn = WTAActionAttention(
            hidden_dim, nheads, interaction, topk=interaction.wta_vision_topk,
        )
        self.norms = nn.ModuleDict({
            "input": TokenCurrentNorm(hidden_dim, eps=interaction.spike_norm_eps),
            "hidden": TokenCurrentNorm(dim_feedforward, eps=interaction.spike_norm_eps),
        })


class DeepSpikeACTDecoder(SpikeACTDecoder):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction):
        super().__init__(config, hidden_dim, nheads, dim_feedforward, interaction)
        self.layers = nn.ModuleList([
            DeepSpikeDecoderLayer(hidden_dim, nheads, dim_feedforward, interaction)
            for _ in range(config.num_layers)
        ])
        self.action_projection.norms = nn.ModuleList([
            TokenCurrentNorm(dim, eps=interaction.spike_norm_eps)
            for dim in (hidden_dim, config.mlp_hidden_dim, config.mlp_hidden_dim)
        ])


class DeepSpikeActionHead(SpikeTurboVLAActionHead):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction):
        if not config.continuous_state:
            raise ValueError("Deep WTA head keeps A's continuous state MLP")
        super().__init__(config, hidden_dim, nheads, dim_feedforward, interaction)
        self.decoder = DeepSpikeACTDecoder(config, hidden_dim, nheads, dim_feedforward, interaction)
