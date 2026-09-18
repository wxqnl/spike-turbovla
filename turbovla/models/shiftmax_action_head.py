"""BSA/Shiftmax action head with eight coordinate-local state tokens.

Adapted from Bipolar Self-attention for Spiking Transformers, NeurIPS 2025,
equations 13-19. Q/K are signed unit PLIF events; V retains normalized current
instead of being thresholded again. Forward attention weights are exact powers
of two with live row sums in (0.5, 1], not binary top-k or ordinary softmax.

Our adaptations: ACT self/cross attention, token-local LayerNorm, the existing
PLIF discretization, explicit padding masks, and a documented ceil-STE
normalization derivative. This is dense FP32 training, not a neuromorphic
kernel, an integer-only implementation, or a demonstrated energy reduction.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .components.wta_operators import BipolarPLIF
from .deep_spike_action_head import DeepSpikeACTDecoder, TokenCurrentNorm
from .full_state_action_head import CoordinateSpikeStateProjection, FullStateSpikeActionHead
from .spike_action_head import SpikeAttention, SpikeDecoderLayer


class _Shiftmax(torch.autograd.Function):
    """Exact dyadic forward; straight-through ceiling in the log normalizer.

For integer x, subtracting the integer row maximum is an exact stabilization
of 2**(x - ceil(log2(sum(2**x)))). Only the normalizer's ceiling uses an STE.
The backward is therefore ln(2) * w_i * (g_i - sum_j(p_j*g_j)), p=w/sum(w).
It includes cross-key competition; it is not the literal discontinuous
derivative of the forward quantization and is not an identity surrogate.
"""

    @staticmethod
    def forward(ctx, scores, allowed):
        live = allowed.any(-1, keepdim=True)
        masked = scores.masked_fill(~allowed, -torch.inf)
        maximum = masked.amax(-1, keepdim=True)
        maximum = torch.where(live, maximum, torch.zeros_like(maximum))
        numerator = torch.exp2(masked - maximum)
        # A live row contains a maximum with numerator 1; an empty row is 0.
        denominator_exponent = torch.log2(numerator.sum(-1, keepdim=True).clamp_min(1)).ceil()
        weights = numerator / torch.exp2(denominator_exponent)
        ctx.save_for_backward(weights)
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        (weights,) = ctx.saved_tensors
        probability = weights / weights.sum(-1, keepdim=True).clamp_min(0.5)
        centered = grad_output - (grad_output * probability).sum(-1, keepdim=True)
        return math.log(2.0) * weights * centered, None


class ShiftmaxRouting(nn.Module):
    """Normalize signed QK integer scores, masking padding rather than silence."""

    def forward(self, scores, allowed):
        if scores.ndim != 4 or allowed.dtype != torch.bool:
            raise ValueError("Shiftmax requires [B,H,Nq,Nk] scores and a boolean validity mask")
        if allowed.ndim == 3:
            allowed = allowed[:, None]
        return _Shiftmax.apply(scores, allowed.expand_as(scores))


class ShiftmaxActionAttention(SpikeAttention):
    def __init__(self, hidden_dim, nheads, interaction, *, continuous_inputs=False):
        super().__init__(hidden_dim, nheads, interaction)
        self.continuous_inputs = continuous_inputs
        self.norms = nn.ModuleDict({
            name: TokenCurrentNorm(hidden_dim, eps=interaction.spike_norm_eps)
            for name in self.norms
        })
        # V is the normalized projection current in BSA Eq.16, not TSN(V).
        # Continuous inputs keep their precision until the Q/K projections.
        # Do not retain unused neuron parameters in the DDP optimizer.
        continuous_nodes = {"value"}
        if continuous_inputs:
            continuous_nodes.update({"input_query", "input_memory"})
        self.spike_nodes = nn.ModuleDict({
            name: BipolarPLIF() for name in self.norms if name not in continuous_nodes
        })
        self.routing = ShiftmaxRouting()

    def forward_step(self, query, memory, memory_valid, state, t):
        query_valid = torch.ones((*query.shape[:2], 1), dtype=torch.bool, device=query.device)

        def fire(name, current, valid):
            spikes, state[name] = self.spike_nodes[name](
                self.norms[name](current, valid, t), state.get(name), valid=valid,
            )
            return spikes

        if self.continuous_inputs:
            query_input = self.norms["input_query"](query, query_valid, t)
            memory_input = self.norms["input_memory"](memory, memory_valid, t)
        else:
            query_input = fire("input_query", query, query_valid)
            memory_input = fire("input_memory", memory, memory_valid)
        q = fire("query", self.query(query_input), query_valid)
        k = fire("key", self.key(memory_input), memory_valid)
        v = self.norms["value"](self.value(memory_input), memory_valid, t)
        batch, nq, channels = q.shape
        nk = k.shape[1]
        q = q.reshape(batch, nq, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        allowed = query_valid & memory_valid.transpose(-1, -2)
        # No sqrt(d), data-dependent score std, or score rounding: QK is already
        # integer-valued because both operands contain only -1, 0, +1 events.
        # A valid silent key is not padding and retains the BSA zero-score path.
        route = self.routing(q @ k.transpose(-1, -2), allowed)
        context = (route @ v).transpose(1, 2).reshape(batch, nq, channels)
        live = query_valid & memory_valid.any(dim=1, keepdim=True)
        return self.output(fire("context", context, live)).masked_fill(~live, 0.0)


class ContinuousFFNDecoderLayer(SpikeDecoderLayer):
    """A's pre-norm continuous FFN inside the unchanged temporal decoder.

    Reuse the constructed common Linear/LayerNorm/attention modules, so this
    replacement consumes no initialization RNG. No FFN PLIF, hidden norm or
    membrane state remains. Each timestep keeps its own continuous FFN output.
    """

    def __init__(self, layer, dropout):
        nn.Module.__init__(self)
        self.self_attn = layer.self_attn
        self.cross_attn = layer.cross_attn
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.norms = nn.ModuleDict({"input": layer.norms["input"]})
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward_step(self, x, memory, memory_valid, state, t):
        query_valid = torch.ones((*x.shape[:2], 1), device=x.device, dtype=torch.bool)
        x = x + self.self_attn.forward_step(x, x, query_valid, state.setdefault("self", {}), t)
        x = x + self.cross_attn.forward_step(x, memory, memory_valid, state.setdefault("cross", {}), t)
        hidden = self.norms["input"](x, time_index=t)
        hidden = self.linear2(self.dropout(self.activation(self.linear1(hidden))))
        return x + self.dropout3(hidden)


class ShiftmaxACTDecoder(DeepSpikeACTDecoder):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction):
        super().__init__(config, hidden_dim, nheads, dim_feedforward, interaction)
        continuous_ffn = config.head_type in {
            "full_spike_state8_shiftmax_continuous_inputs_ann_ffn",
            "full_spike_state8_shiftmax_ann_ffn",
        }
        continuous_inputs = config.head_type in {
            "full_spike_state8_shiftmax_continuous_inputs",
            "full_spike_state8_shiftmax_continuous_inputs_ann_ffn",
        }
        for index, layer in enumerate(self.layers):
            layer.self_attn = ShiftmaxActionAttention(
                hidden_dim, nheads, interaction, continuous_inputs=continuous_inputs,
            )
            layer.cross_attn = ShiftmaxActionAttention(
                hidden_dim, nheads, interaction, continuous_inputs=continuous_inputs,
            )
            if continuous_ffn:
                self.layers[index] = ContinuousFFNDecoderLayer(layer, config.dropout)


class FullStateShiftmaxActionHead(FullStateSpikeActionHead):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction):
        nn.Module.__init__(self)
        self.config, self.state_dim = config, config.state_dim
        # Same constructor order as the existing per-coordinate WTA head keeps
        # the state-encoder initialization unchanged for a given seed.
        self.state_projection = CoordinateSpikeStateProjection(config, hidden_dim, interaction)
        self.decoder = ShiftmaxACTDecoder(config, hidden_dim, nheads, dim_feedforward, interaction)
