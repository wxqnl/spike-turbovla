"""WTA spike fusion v3: full temporal fusion, not an all-SNN encoder claim.

Our TurboVLA adaptation combines signed PLIF, binary multi-winner routing,
mask-aware time-indexed synchronized BN, and membrane-residual spike FFNs.
All weighted synapse inputs in this new scope are {-1, 0, +1} events.
The final action readout, membrane states and normalization remain continuous.
"""
from __future__ import annotations

import torch
from torch import nn

from .temporal_fusion import (
    SpikeTextProjection, SpikeVisionProjection, TemporalCrossAttention,
    TemporalSelfAttention, TemporalSpikeInteraction, fire,
)
from .wta_operators import (
    BipolarPLIF, MaskedTemporalBatchNorm, MultiWinnerAttention,
    SpikeDotProductAttention, checkpoint_contexts,
)


def current_norm(channels, config):
    return MaskedTemporalBatchNorm(
        channels, config.cross_timesteps, momentum=config.spike_norm_momentum,
        eps=config.spike_norm_eps, sync=config.spike_norm_sync,
    )


class WTATextProjection(SpikeTextProjection):
    def __init__(self, in_features, out_features, bias=True, *, config):
        super().__init__(in_features, out_features, bias=bias)
        self.input_norm = current_norm(in_features, config)
        self.output_norm = current_norm(out_features, config)
        self.spike_nodes = nn.ModuleDict({"input": BipolarPLIF()})

    def forward_step(self, hidden, state, valid, time_index=0):
        spikes = fire(self.spike_nodes, state, "input",
                      self.input_norm(hidden, valid, time_index), valid)
        current = self(spikes, from_temporal=True)
        return self.output_norm(current, valid, time_index).masked_fill(~valid, 0)


class WTAVisionProjection(SpikeVisionProjection):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.0, *, config):
        super().__init__(in_dim, out_dim, hidden_dim, dropout=dropout)
        self.input_norm = current_norm(in_dim, config)
        self.hidden_norm = current_norm(hidden_dim, config)
        self.output_norm = current_norm(out_dim, config)
        self.spike_nodes = nn.ModuleDict({"input": BipolarPLIF(), "hidden": BipolarPLIF()})

    def forward_step(self, tokens, state, time_index=0):
        spikes = fire(self.spike_nodes, state, "input",
                      self.input_norm(tokens, time_index=time_index))
        hidden = fire(self.spike_nodes, state, "hidden",
                      self.hidden_norm(self.linear1(spikes), time_index=time_index))
        return self.output_norm(self.skip(spikes) + self.linear2(hidden), time_index=time_index)


class WTACrossAttention(TemporalCrossAttention):
    def __init__(self, config):
        super().__init__(config.hidden_dim, config.enhancer_inner_dim, max(1, config.nheads // 2))
        sizes = {"input_v": config.hidden_dim, "input_l": config.hidden_dim,
                 "query_v": config.enhancer_inner_dim, "key_l": config.enhancer_inner_dim,
                 "value_v": config.enhancer_inner_dim, "value_l": config.enhancer_inner_dim,
                 "context_v": config.enhancer_inner_dim, "context_l": config.enhancer_inner_dim}
        self.norms = nn.ModuleDict({name: current_norm(dim, config) for name, dim in sizes.items()})
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in sizes})
        self.attend_text = MultiWinnerAttention(
            config.wta_text_topk, self.head_dim, config.wta_surrogate_temperature)
        self.attend_vision = MultiWinnerAttention(
            config.wta_vision_topk, self.head_dim, config.wta_surrogate_temperature)

    def forward_step(self, v, l, valid_v, valid_l, state, time_index=0):
        def encode(name, current, valid):
            return fire(self.spike_nodes, state, name,
                        self.norms[name](current, valid, time_index), valid)
        iv, il = encode("input_v", v, valid_v), encode("input_l", l, valid_l)
        q = self._heads(encode("query_v", self.v_proj(iv), valid_v))
        k = self._heads(encode("key_l", self.l_proj(il), valid_l))
        vv = self._heads(encode("value_v", self.values_v_proj(iv), valid_v))
        vl = self._heads(encode("value_l", self.values_l_proj(il), valid_l))
        score = q @ k.transpose(-1, -2)
        allowed = valid_v & valid_l.transpose(-1, -2)
        # A row-wise top-k map cannot be transposed and reused as reverse
        # routing: the two directions have different source sets and budgets.
        route_v = self.attend_text(score, allowed, q, k)
        route_l = self.attend_vision(score.transpose(-1, -2), allowed.transpose(-1, -2), k, q)
        cv = (route_v @ vl).transpose(1, 2).reshape(v.shape[0], v.shape[1], -1)
        cl = (route_l @ vv).transpose(1, 2).reshape(l.shape[0], l.shape[1], -1)
        # Suppress even output bias/norm offsets if a query has no live route.
        keep_v = valid_v & route_v.detach().ne(0).any(-1).any(1).unsqueeze(-1)
        keep_l = valid_l & route_l.detach().ne(0).any(-1).any(1).unsqueeze(-1)
        sv = encode("context_v", cv, keep_v)
        sl = encode("context_l", cl, keep_l)
        return self.out_v_proj(sv).masked_fill(~keep_v, 0), self.out_l_proj(sl).masked_fill(~keep_l, 0)


class WTACrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = WTACrossAttention(config)
        self.gamma_v = nn.Parameter(torch.full((config.hidden_dim,), 1e-4))
        self.gamma_l = nn.Parameter(torch.full((config.hidden_dim,), 1e-4))

    def forward_step(self, v, l, valid_v, valid_l, state, time_index=0):
        dv, dl = self.attn.forward_step(v, l, valid_v, valid_l, state, time_index)
        return v + self.gamma_v * dv, (l + self.gamma_l * dl).masked_fill(~valid_l, 0)


class WTASelfAttention(TemporalSelfAttention):
    def __init__(self, config):
        dim, heads = config.hidden_dim, max(1, config.nheads // 2)
        super().__init__(dim, heads)
        self.norms = nn.ModuleDict({name: current_norm(dim, config)
                                   for name in ("input", "q", "k", "v", "context")})
        self.spike_nodes = nn.ModuleDict({name: BipolarPLIF() for name in self.norms})
        if config.text_attention_routing == "wta":
            self.routing = MultiWinnerAttention(
                config.wta_text_topk, self.head_dim, config.wta_surrogate_temperature)
        else:
            self.routing = SpikeDotProductAttention()

    def forward_step(self, src, valid, allowed, state, time_index=0):
        def encode(name, current, mask=valid):
            return fire(self.spike_nodes, state, name,
                        self.norms[name](current, mask, time_index), mask)
        q, k, v = self.qkv(encode("input", src)).chunk(3, dim=-1)
        def heads(current):
            return current.reshape(current.shape[0], current.shape[1],
                                   self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(encode("q", q)), heads(encode("k", k)), heads(encode("v", v))
        route = self.routing(q @ k.transpose(-1, -2), allowed, q, k)
        context = (route @ v).transpose(1, 2).reshape(src.shape)
        keep = valid & route.detach().ne(0).any(-1).any(1).unsqueeze(-1)
        return self.out_proj(encode("context", context, keep)).masked_fill(~keep, 0)


class WTATextBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim
        self.self_attn = WTASelfAttention(config)
        self.linear1 = nn.Linear(dim, config.enhancer_inner_dim)
        self.linear2 = nn.Linear(config.enhancer_inner_dim, dim)
        self.norms = nn.ModuleDict({
            "input": current_norm(dim, config),
            "hidden": current_norm(config.enhancer_inner_dim, config),
            "output": current_norm(dim, config),
        })
        self.spike_nodes = nn.ModuleDict({"input": BipolarPLIF(), "hidden": BipolarPLIF()})

    def forward_step(self, src, valid, allowed, state, time_index=0):
        src = src + self.self_attn.forward_step(
            src, valid, allowed, state.setdefault("attention", {}), time_index)
        ff_state = state.setdefault("ffn", {})
        spikes = fire(self.spike_nodes, ff_state, "input",
                      self.norms["input"](src, valid, time_index), valid)
        hidden = fire(self.spike_nodes, ff_state, "hidden",
                      self.norms["hidden"](self.linear1(spikes), valid, time_index), valid)
        delta = self.norms["output"](self.linear2(hidden), valid, time_index)
        return (src + delta).masked_fill(~valid, 0)


class WTASpikeInteraction(TemporalSpikeInteraction):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.fusion_layers = nn.ModuleList([WTACrossBlock(config) for _ in range(config.num_layers)])
        self.text_layers = nn.ModuleList([WTATextBlock(config) for _ in range(config.num_layers)])

    @staticmethod
    def _run_layer(v, l, valid_v, valid_l, allowed, fusion, text):
        cross_state, text_state = {}, {}
        vs, ls = [], []
        for t in range(v.shape[0]):
            vt, lt = fusion.forward_step(v[t], l[t], valid_v, valid_l, cross_state, t)
            lt = text.forward_step(lt, valid_l, allowed, text_state, t)
            vs.append(vt)
            ls.append(lt)
        return torch.stack(vs), torch.stack(ls)

    @staticmethod
    def checkpoint_contexts():
        return checkpoint_contexts()

    def common_pretrained_parameter_names(self):
        return {name for name, _ in self.named_parameters()
                if ".spike_nodes." not in name and ".norms." not in name}
