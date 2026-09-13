"""Eight coordinate-local state tokens feeding the existing signed WTA head.

Both encoders emit signed unit events before the first learned projection.
The per-dimension arm uses a PLIF neuron per coordinate; the population arm
uses K staggered fixed thresholds per coordinate at each of T neuron steps.
Every step receives all eight standardized coordinates, never an 8/T slice.
Current norms, membranes, residuals and final robot actions remain continuous.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .components.spike_fusion import _ATanSpike
from .components.wta_operators import BipolarPLIF
from .deep_spike_action_head import DeepSpikeACTDecoder, TokenCurrentNorm
from .spike_action_head import SpikeTurboVLAActionHead


class CoordinateEventProjection(nn.Module):
    """Independent event-to-current weights for each state coordinate."""

    def __init__(self, coordinates, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(coordinates, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(coordinates, out_features))
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))

    def forward(self, events):
        return torch.einsum("bnk,ndk->bnd", events, self.weight) + self.bias


class CoordinatePopulationEncoder(nn.Module):
    """[B,8] to [B,8,K] events; T*K interleaved boundaries over [-limit,limit]."""

    def __init__(self, bins, timesteps, limit):
        super().__init__()
        grid = (torch.arange(bins * timesteps, dtype=torch.float32) + 0.5)
        grid = grid * (2 * limit / (bins * timesteps)) - limit
        self.register_buffer("centers", grid.reshape(bins, timesteps).T.contiguous())

    def forward(self, current, time_index):
        centers = self.centers[time_index]
        return _ATanSpike.apply(current[..., None] - centers) - (centers <= 0).to(current.dtype)


class CoordinateSpikeStateProjection(nn.Module):
    def __init__(self, config, hidden_dim, interaction):
        super().__init__()
        self.encoding = config.spike_state_encoding
        bins = config.spike_state_population if self.encoding == "multithreshold" else 1
        # Input projection shape differs across arms. Keep RNG consumption of
        # the common hidden/output layers identical despite that difference.
        with torch.random.fork_rng(devices=[]):
            self.input_projection = CoordinateEventProjection(config.state_dim, bins, config.state_hidden_dim)
        self.output_projection = nn.Linear(config.state_hidden_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, config.state_dim, hidden_dim))
        self.hidden_norm = TokenCurrentNorm(config.state_hidden_dim, eps=interaction.spike_norm_eps)
        self.output_norm = TokenCurrentNorm(hidden_dim, eps=interaction.spike_norm_eps)
        self.hidden_spike = BipolarPLIF()
        if self.encoding == "perdim_plif":
            self.input_spike = BipolarPLIF()
        else:
            self.population = CoordinatePopulationEncoder(
                bins, config.spike_timesteps, config.spike_state_population_range,
            )

    def forward_step(self, current, state, t):
        # The dataset has already applied fixed per-coordinate training stats.
        # Do not normalize a scalar with LayerNorm or mix coordinate statistics.
        if self.encoding == "perdim_plif":
            events, state["input"] = self.input_spike(current[..., None], state.get("input"))
        else:
            events = self.population(current, t)
        hidden = self.input_projection(events)
        spikes, state["hidden"] = self.hidden_spike(self.hidden_norm(hidden), state.get("hidden"))
        return self.output_norm(self.output_projection(spikes) + self.position)


class FullStateSpikeActionHead(SpikeTurboVLAActionHead):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction):
        nn.Module.__init__(self)
        self.config, self.state_dim = config, config.state_dim
        self.state_projection = CoordinateSpikeStateProjection(config, hidden_dim, interaction)
        self.decoder = DeepSpikeACTDecoder(config, hidden_dim, nheads, dim_feedforward, interaction)

    @torch.no_grad()
    def load_common(self, original):
        # The state branch is new and fresh. Preserve only the common fresh
        # decoder initialization, never weights from the stopped Deep policy.
        self.decoder.load_common(original.decoder)
