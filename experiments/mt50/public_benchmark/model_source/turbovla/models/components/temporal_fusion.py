"""Temporal spike fusion v2; not a claim that the frozen encoders are all-SNN.

The six-layer fusion keeps T until the single action-head readout. Synaptic
Linear inputs and Q/K/V are binary; residuals are membrane currents. Static
channel affine calibration is explicit, saved, and never uses eval-batch stats.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .spike_fusion import FunctionalPLIF


class StaticCurrentNorm(nn.Module):
    """Training-data calibration, not LayerNorm or inference-time BatchNorm.

    The frozen channel affine can be absorbed into the preceding synapse.
    Input-boundary affines remain continuous and are explicitly disclosed.
    Calibration changes buffers only; regular forward has no side effects.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("center", torch.zeros(channels))
        self.register_buffer("inv_std", torch.ones(channels))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("sum_x", torch.zeros(channels, dtype=torch.float64))
        self.register_buffer("sum_x2", torch.zeros(channels, dtype=torch.float64))
        self.calibrating = False

    def begin(self):
        self.center.zero_()
        self.inv_std.fill_(1)
        self.sample_count.zero_()
        self.sum_x.zero_()
        self.sum_x2.zero_()
        self.calibrating = True

    def forward(self, current, valid=None):
        if self.calibrating:
            with torch.no_grad():
                flat = current.detach().reshape(-1, current.shape[-1]).double()
                if valid is not None:
                    flat = flat[valid.expand(*current.shape[:-1], 1).reshape(-1)]
                count = flat.new_tensor(float(len(flat)))
                sums = flat.sum(0)
                squares = flat.square().sum(0)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    for value in (count, sums, squares):
                        torch.distributed.all_reduce(value)
                if count.item():
                    self.sample_count.add_(count)
                    self.sum_x.add_(sums)
                    self.sum_x2.add_(squares)
                    mean = self.sum_x / self.sample_count
                    variance = (self.sum_x2 / self.sample_count - mean.square()).clamp_min(0.01)
                    self.center.copy_(mean.to(self.center))
                    self.inv_std.copy_(variance.rsqrt().to(self.inv_std))
        return (current - self.center) * self.inv_std


def fire(nodes, state, name, current, valid=None):
    spikes, state[name] = nodes[name](current, state.get(name), valid=valid)
    return spikes


class SpikeTextProjection(nn.Linear):
    """Retains the calibrated public weight/bias, replaces its input interface."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.input_norm = StaticCurrentNorm(in_features)
        self.output_norm = StaticCurrentNorm(out_features)
        self.spike_nodes = nn.ModuleDict({"input": FunctionalPLIF()})

    def forward_step(self, hidden, state, valid, time_index=0):
        spikes = fire(self.spike_nodes, state, "input", self.input_norm(hidden, valid), valid)
        return self.output_norm(self(spikes, from_temporal=True), valid).masked_fill(~valid, 0)

    def forward(self, spikes, *, from_temporal=False):
        if not from_temporal:
            raise RuntimeError("Temporal text projection requires forward_step and an explicit local state")
        return super().forward(spikes)

    def load_common_projection(self, state):
        if set(state) != {"weight", "bias"}:
            raise ValueError("Expected a weight/bias-only common projection")
        with torch.no_grad():
            for name, value in state.items():
                target = getattr(self, name)
                if target.shape != value.shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid common projection {name}")
                target.copy_(value)


class SpikeVisionProjection(nn.Module):
    """Binary-input synapses with a membrane-current residual projection."""

    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.input_norm = StaticCurrentNorm(in_dim)
        # Keep construction order of the original two-layer projection.
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.hidden_norm = StaticCurrentNorm(hidden_dim)
        self.output_norm = StaticCurrentNorm(out_dim)
        self.spike_nodes = nn.ModuleDict({
            "input": FunctionalPLIF(), "hidden": FunctionalPLIF(),
        })

    def forward_step(self, tokens, state, time_index=0):
        spikes = fire(self.spike_nodes, state, "input", self.input_norm(tokens))
        hidden = fire(self.spike_nodes, state, "hidden", self.hidden_norm(self.linear1(spikes)))
        return self.output_norm(self.skip(spikes) + self.linear2(hidden))

    def forward(self, tokens):
        raise RuntimeError("Temporal visual projection requires forward_step and an explicit local state")


class TemporalCrossAttention(nn.Module):
    def __init__(self, hidden, inner, heads):
        super().__init__()
        if inner % heads:
            raise ValueError("Cross inner dimension must be divisible by heads")
        self.num_heads, self.head_dim = heads, inner // heads
        self.v_proj = nn.Linear(hidden, inner)
        self.l_proj = nn.Linear(hidden, inner)
        self.values_v_proj = nn.Linear(hidden, inner)
        self.values_l_proj = nn.Linear(hidden, inner)
        self.out_v_proj = nn.Linear(inner, hidden)
        self.out_l_proj = nn.Linear(inner, hidden)
        sizes = {"input_v": hidden, "input_l": hidden,
                 "query_v": inner, "key_l": inner, "value_v": inner, "value_l": inner,
                 "context_v": inner, "context_l": inner}
        self.norms = nn.ModuleDict({name: StaticCurrentNorm(dim) for name, dim in sizes.items()})
        self.spike_nodes = nn.ModuleDict({name: FunctionalPLIF() for name in sizes})

    def _heads(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward_step(self, v, l, valid_v, valid_l, state):
        def encode(name, current, valid):
            return fire(self.spike_nodes, state, name, self.norms[name](current, valid), valid)
        iv, il = encode("input_v", v, valid_v), encode("input_l", l, valid_l)
        q = self._heads(encode("query_v", self.v_proj(iv), valid_v))
        k = self._heads(encode("key_l", self.l_proj(il), valid_l))
        vv = self._heads(encode("value_v", self.values_v_proj(iv), valid_v))
        vl = self._heads(encode("value_l", self.values_l_proj(il), valid_l))
        # Every product has a binary operand; no softmax, continuous QK, or scale.
        counts = q @ k.transpose(-1, -2)
        cv = (counts @ vl).transpose(1, 2).reshape(v.shape[0], v.shape[1], -1)
        cl = (counts.transpose(-1, -2) @ vv).transpose(1, 2).reshape(l.shape[0], l.shape[1], -1)
        sv = encode("context_v", cv, valid_v)
        sl = encode("context_l", cl, valid_l)
        keep_v = valid_v & valid_l.any(1, keepdim=True)
        keep_l = valid_l & valid_v.any(1, keepdim=True)
        return self.out_v_proj(sv).masked_fill(~keep_v, 0), self.out_l_proj(sl).masked_fill(~keep_l, 0)


class TemporalCrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = TemporalCrossAttention(
            config.hidden_dim, config.enhancer_inner_dim, max(1, config.nheads // 2)
        )
        self.gamma_v = nn.Parameter(torch.full((config.hidden_dim,), 1e-4))
        self.gamma_l = nn.Parameter(torch.full((config.hidden_dim,), 1e-4))

    def forward_step(self, v, l, valid_v, valid_l, state):
        dv, dl = self.attn.forward_step(v, l, valid_v, valid_l, state)
        # Membrane shortcut; binary events are regenerated before every synapse.
        return v + self.gamma_v * dv, (l + self.gamma_l * dl).masked_fill(~valid_l, 0)


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("Text hidden dimension must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norms = nn.ModuleDict({name: StaticCurrentNorm(dim) for name in ("input", "q", "k", "v", "context")})
        self.spike_nodes = nn.ModuleDict({name: FunctionalPLIF() for name in self.norms})

    def forward_step(self, src, valid, allowed, state):
        def encode(name, x):
            return fire(self.spike_nodes, state, name, self.norms[name](x, valid), valid)
        inputs = encode("input", src)
        q, k, v = self.qkv(inputs).chunk(3, dim=-1)
        def heads(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(encode("q", q)), heads(encode("k", k)), heads(encode("v", v))
        # Boolean pair mask stays [B,1,N,N]: no erroneous head/batch flattening.
        counts = (q @ k.transpose(-1, -2)).masked_fill(~allowed[:, None], 0)
        context = (counts @ v).transpose(1, 2).reshape(src.shape)
        return self.out_proj(encode("context", context)).masked_fill(~valid, 0)


class TemporalTextBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim
        self.self_attn = TemporalSelfAttention(dim, max(1, config.nheads // 2))
        self.linear1 = nn.Linear(dim, config.enhancer_inner_dim)
        self.linear2 = nn.Linear(config.enhancer_inner_dim, dim)
        self.norms = nn.ModuleDict({
            "input": StaticCurrentNorm(dim),
            "hidden": StaticCurrentNorm(config.enhancer_inner_dim),
            "output": StaticCurrentNorm(dim),
        })
        self.spike_nodes = nn.ModuleDict({"input": FunctionalPLIF(), "hidden": FunctionalPLIF()})

    def forward_step(self, src, valid, allowed, state):
        src = src + self.self_attn.forward_step(src, valid, allowed, state.setdefault("attention", {}))
        ff_state = state.setdefault("ffn", {})
        spikes = fire(self.spike_nodes, ff_state, "input", self.norms["input"](src, valid), valid)
        hidden = fire(self.spike_nodes, ff_state, "hidden", self.norms["hidden"](self.linear1(spikes), valid), valid)
        delta = self.norms["output"](self.linear2(hidden), valid)
        return (src + delta).masked_fill(~valid, 0)


class TemporalSpikeInteraction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion_layers = nn.ModuleList([TemporalCrossBlock(config) for _ in range(config.num_layers)])
        self.text_layers = nn.ModuleList([TemporalTextBlock(config) for _ in range(config.num_layers)])

    @staticmethod
    def checkpoint_contexts():
        return nullcontext(), nullcontext()

    @staticmethod
    def _run_layer(v, l, valid_v, valid_l, allowed, fusion, text):
        cross_state, text_state = {}, {}
        vs, ls = [], []
        for t in range(v.shape[0]):
            vt, lt = fusion.forward_step(v[t], l[t], valid_v, valid_l, cross_state)
            lt = text.forward_step(lt, valid_l, allowed, text_state)
            vs.append(vt)
            ls.append(lt)
        return torch.stack(vs), torch.stack(ls)

    def forward(self, visual_tokens, text_tokens, text_key_padding_mask, text_self_attention_masks):
        if visual_tokens.ndim != 4 or text_tokens.ndim != 4:
            raise ValueError("Temporal fusion requires [T,B,N,C] for both modalities")
        if visual_tokens.shape[:2] != text_tokens.shape[:2] or visual_tokens.shape[0] != self.config.cross_timesteps:
            raise ValueError("Mismatched modality time or batch axes")
        if text_key_padding_mask.dtype != torch.bool or text_key_padding_mask.shape != text_tokens.shape[1:3]:
            raise ValueError("Text padding mask must be bool [B,N], True means excluded")
        valid_l = ~text_key_padding_mask[..., None]
        valid_v = torch.ones((*visual_tokens.shape[1:3], 1), dtype=torch.bool, device=visual_tokens.device)
        allowed = valid_l & valid_l.transpose(-1, -2)
        if text_self_attention_masks is not None:
            if text_self_attention_masks.dtype != torch.bool or text_self_attention_masks.shape != allowed.shape:
                raise ValueError("Sentence mask must be bool [B,N,N], True means allowed")
            allowed = allowed & text_self_attention_masks
        text_tokens = text_tokens.masked_fill(~valid_l[None], 0)
        for fusion, text in zip(self.fusion_layers, self.text_layers):
            # Default arguments capture this layer, including during backward recomputation.
            def run(v, l, fusion=fusion, text=text):
                return self._run_layer(v, l, valid_v, valid_l, allowed, fusion, text)
            if self.training and torch.is_grad_enabled() and self.config.cross_gradient_checkpointing:
                visual_tokens, text_tokens = checkpoint(
                    run, visual_tokens, text_tokens, use_reentrant=False,
                    context_fn=self.checkpoint_contexts,
                )
            else:
                visual_tokens, text_tokens = run(visual_tokens, text_tokens)
        return visual_tokens, text_tokens

    def common_pretrained_parameter_names(self):
        return {name for name, _ in self.named_parameters() if ".spike_nodes." not in name}


def calibration_modules(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, StaticCurrentNorm)]


def begin_calibration(model):
    modules = calibration_modules(model)
    if not modules:
        raise ValueError("No temporal spike calibration modules")
    for _, module in modules:
        module.begin()


def finish_calibration(model):
    report = {}
    for name, module in calibration_modules(model):
        module.calibrating = False
        if (module.sample_count.item() < 2 or not torch.isfinite(module.inv_std).all()
                or not torch.isfinite(module.center).all()):
            raise RuntimeError(f"Insufficient or invalid training-data calibration at {name}")
        report[name] = {"observed_tokens": int(module.sample_count.item()),
                        "minimum_std": float(module.inv_std.reciprocal().min()),
                        "maximum_std": float(module.inv_std.reciprocal().max())}
    return report
