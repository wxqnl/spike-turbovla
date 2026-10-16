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
import torch.nn.functional as F
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
    def forward(ctx, scores, allowed, bits=0):
        live = allowed.any(-1, keepdim=True)
        masked = scores.masked_fill(~allowed, -torch.inf)
        maximum = masked.amax(-1, keepdim=True)
        maximum = torch.where(live, maximum, torch.zeros_like(maximum))
        numerator = torch.exp2(masked - maximum)
        # A live row contains a maximum with numerator 1; an empty row is 0. The exponent is rounded up to an integer
        # (bits=0, the original) or to 1/2**bits: then a row's weights jump by at most 2**(1/2**bits), not 2x, when
        # its sum crosses a rounding step (a small table of 2**(k/2**bits) times a shift).
        step = float(2 ** bits)
        denominator_exponent = (torch.log2(numerator.sum(-1, keepdim=True).clamp_min(1)) * step).ceil() / step
        weights = numerator / torch.exp2(denominator_exponent)
        ctx.save_for_backward(weights)
        return weights

    @staticmethod
    def backward(ctx, grad_output):
        (weights,) = ctx.saved_tensors
        probability = weights / weights.sum(-1, keepdim=True).clamp_min(0.5)
        centered = grad_output - (grad_output * probability).sum(-1, keepdim=True)
        return math.log(2.0) * weights * centered, None, None


class IntegerPLIF(BipolarPLIF):
    """BipolarPLIF's charging with integer events and a soft reset (I-LIF style).

    s = clamp(round(u / threshold), -levels, levels), then u <- u - s * threshold. The hard reset of BipolarPLIF
    drops how far the membrane overshot the threshold, so a context that changes little over the timesteps maps to a
    coarse staircase; the soft reset carries the residual over and the integer event keeps the overshoot. At inference
    an event of value s is |s| unit events of one sign (the next layer accumulates its weight column |s| times), so the
    layer stays addition-only. Gradient: straight through inside [-levels, levels]. Same parameters (and so the same
    initialization) as BipolarPLIF.
    """

    def __init__(self, levels, threshold=0.5):
        super().__init__(threshold)
        if levels < 1:
            raise ValueError("IntegerPLIF needs levels >= 1")
        self.levels = int(levels)

    def forward(self, current, membrane=None, threshold_scale=1.0, valid=None):
        if membrane is None:
            membrane = torch.zeros_like(current)
        voltage = membrane + self.decay_logit.sigmoid() * (current - membrane)
        threshold = self.log_threshold.exp() * threshold_scale
        scaled = (voltage / threshold).clamp(-self.levels, self.levels)
        spike = scaled + (torch.round(scaled) - scaled).detach()
        if valid is not None:
            spike = spike.masked_fill(~valid, 0.0)
            voltage = voltage.masked_fill(~valid, 0.0)
        return spike, voltage - spike.detach() * threshold


class ShiftmaxRouting(nn.Module):
    """Normalize signed QK integer scores, masking padding rather than silence."""

    def __init__(self, bits=0):
        super().__init__()
        self.bits = int(bits)  # 0: normalizer exponent rounded up to an integer (the original Shiftmax)

    def forward(self, scores, allowed):
        if scores.ndim != 4 or allowed.dtype != torch.bool:
            raise ValueError("Shiftmax requires [B,H,Nq,Nk] scores and a boolean validity mask")
        if allowed.ndim == 3:
            allowed = allowed[:, None]
        return _Shiftmax.apply(scores, allowed.expand_as(scores), self.bits)


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


class IFRCrossAttention(ShiftmaxActionAttention):
    """Cross-attention of the spiking action head with Integrate-and-Fire Routing (IFR) options.

    v2's Spatially Grounded Spike Attention (SGSA) is this module with integrate, shift 2, locate, norm_bits 4 (the
    fine normalizer) and sign_field (the sign field), together with the model's learned patch position embedding
    (recipe f7_sdtv3_kd_ifr_signfield). The other options are ablations.

    The Shiftmax attention routes every timestep on that timestep's spikes alone. A small change of the scene that
    flips a few key spikes then reshuffles the routing, and the target's position is averaged away when the head
    reads the image. IFR keeps the spikes, the values and the Shiftmax, and changes what the routing sees and what
    the read sums:

      integrate    each query-key pair integrates its spike match q_t . k_t over timesteps, as the membrane of an
                   integrate-and-fire neuron integrates its input, and routes on the running mean of that evidence
                   instead of on the current snapshot;
      shift        the scores are shifted right by `shift` bits before Shiftmax; with 32 channels per head, 2 bits make
                   base-2 Shiftmax as sharp as softmax with 1/sqrt(d) (2**(x/4) ~ e**(x/sqrt(32)));
      split_state  the robot-state tokens (the last `state_tokens` keys) get a Shiftmax of their own, so they no longer
                   take attention away from the image within one normalization;
      locate       attention reads what is at the attended tokens, not where they are, so a stable routing still
                   loses the target's position in the read. Each visual token's value gets a position value, a
                   learned linear map of fixed smooth coordinates of its patch (row, column and a few frequencies,
                   per camera). The read then also carries where the head looks: the centroid, spread and finer
                   moments of the attention over the image. The table is a parameter, constant at inference; the
                   Shiftmax weights are powers of two, so reading it is shifts and adds like the value read, and the
                   context still goes through the context neuron.
      spatial      before Shiftmax, each visual token's score is integrated over its k x k patch neighborhood
                   (binomial kernel [1,2,1] or [1,4,6,4,1], outer product: weights are powers of two over a power-of-two
                   sum, so shifts and adds), per camera, replicate padding. A sub-patch shift of the target flips key
                   spikes in many tokens with little spatial coherence, so the routing reshuffles by an amount
                   unrelated to the shift; integrating over the neighborhood, as `integrate` does over timesteps,
                   averages the flips out.
      norm_bits    Shiftmax divides by the next power of two above the row sum, so all weights of a row halve or
                   double when its sum crosses a power of two, whatever the target did. The exponent is rounded to
                   1/2**norm_bits instead (jumps <= 4.4% at 4): the fine normalizer.
      sign_field   a weighted sum of token contents cancels where they are: what the target leaves on one token it adds
                   to the next. Half of each head's visual value channels are multiplied by a fixed +-1 spatial pattern
                   of their token's patch (square waves along rows and columns, 4 scales x 2 phases), so the read also
                   carries where the attended contents are. Sign flips only; no parameters.
      context_levels
                   the context neuron fires integer events in [-levels, levels] with a soft reset (IntegerPLIF) instead
                   of BipolarPLIF's unit events with a hard reset (0 = BipolarPLIF), so the context spikes keep more of
                   the context current.

    Same constructor and parameters as the spiking module, hence the same initialization for a given seed (the
    position-value map starts at zero and draws no random numbers).
    """

    LOCATE_FREQUENCIES = (1, 2, 4, 8)  # finest: 4 cycles across the grid, a period of 3.5 patches at 14 x 14
    SPATIAL_KERNELS = {3: (1.0, 2.0, 1.0), 5: (1.0, 4.0, 6.0, 4.0, 1.0)}
    SIGN_SCALES = (1, 2, 4, 8)  # square waves with about 1, 2, 4, 8 sign changes across the grid

    @classmethod
    def sign_patterns(cls, views, grid, head_dim):
        """[views * grid^2, head_dim] in {-1, +1}: the first half of the channels +1, the second half square waves
        sign(cos(pi * s * coordinate + phase)) along rows and columns, phases pi/4 and 3pi/4, cycling."""
        axis = torch.arange(grid, dtype=torch.float64) / max(grid - 1, 1)
        rows, cols = torch.meshgrid(axis, axis, indexing="ij")
        waves = []
        for s in cls.SIGN_SCALES:
            for phase in (math.pi / 4, 3 * math.pi / 4):
                for coord in (rows.flatten(), cols.flatten()):
                    waves.append(torch.cos(math.pi * s * coord + phase))
        waves = torch.stack(waves, -1).sign()
        waves[waves == 0] = 1
        half = head_dim // 2
        modulated = waves.repeat(1, half // waves.shape[1] + 1)[:, :half]
        pattern = torch.cat((torch.ones(grid * grid, head_dim - half, dtype=torch.float64), modulated), -1)
        return pattern.repeat(views, 1).float()

    def __init__(self, hidden_dim, nheads, interaction, *, continuous_inputs, integrate, shift, state_tokens,
                 locate_views=0, locate_grid=0, spatial=0, views=0, grid=0, norm_bits=0, sign_field=False,
                 context_levels=0):
        super().__init__(hidden_dim, nheads, interaction, continuous_inputs=continuous_inputs)
        self.integrate, self.shift, self.state_tokens = bool(integrate), int(shift), int(state_tokens)
        self.context_levels = int(context_levels)
        if self.context_levels:
            self.spike_nodes["context"] = IntegerPLIF(self.context_levels)
        self.spatial, self.views, self.grid = int(spatial), int(views), int(grid)
        self.routing = ShiftmaxRouting(norm_bits)
        self.sign_field = bool(sign_field)
        if self.sign_field:
            if not (self.views and self.grid):
                raise ValueError("sign_field needs the visual layout")
            self.register_buffer("sign_pattern", self.sign_patterns(self.views, self.grid, self.head_dim),
                                 persistent=False)
        if self.spatial:
            if self.spatial not in self.SPATIAL_KERNELS or not (self.views and self.grid):
                raise ValueError(f"spatial integration needs a kernel in {sorted(self.SPATIAL_KERNELS)} and the layout")
            taps = torch.tensor(self.SPATIAL_KERNELS[self.spatial])
            self.register_buffer("spatial_kernel", (taps[:, None] * taps[None, :] / taps.sum() ** 2)[None, None],
                                 persistent=False)
        self.locate = locate_views > 0
        if self.locate:
            axis = torch.arange(locate_grid, dtype=torch.float32) / max(locate_grid - 1, 1)
            rows, cols = torch.meshgrid(axis, axis, indexing="ij")
            coords = torch.stack((rows.flatten(), cols.flatten()), -1)  # [grid^2, 2] in [0, 1], raster order
            angles = math.pi * coords[..., None] * torch.tensor(self.LOCATE_FREQUENCIES, dtype=torch.float32)
            per_view = torch.cat((torch.ones(len(coords), 1), coords, angles.sin().flatten(1), angles.cos().flatten(1)), -1)
            features = torch.block_diag(*[per_view] * locate_views)  # each camera its own block: [views * grid^2, F]
            self.register_buffer("locate_features", features, persistent=False)
            self.locate_value = nn.Parameter(torch.zeros(features.shape[1], hidden_dim))

    def read_values(self, v):
        """[B, H, Nk, head_dim] values -> what the read sums: sign field on the visual tokens, then position values."""
        if self.sign_field:
            nvis = self.sign_pattern.shape[0]
            if v.shape[2] < nvis:
                raise ValueError(f"sign_field expects {nvis} visual tokens first in the memory, got {v.shape[2]} tokens")
            v = torch.cat((v[:, :, :nvis] * self.sign_pattern.to(v.dtype), v[:, :, nvis:]), dim=2)
        return self.position_values(v)

    def position_values(self, v):
        """[B, H, Nk, head_dim] values -> the same with each visual token's position value added (locate)."""
        if not self.locate:
            return v
        nvis = self.locate_features.shape[0]
        if v.shape[2] < nvis:
            raise ValueError(f"locate expects {nvis} visual tokens first in the memory, got {v.shape[2]} tokens")
        position = (self.locate_features @ self.locate_value).to(v.dtype)  # [nvis, hidden], parameters only
        position = position.view(nvis, self.nheads, self.head_dim).transpose(0, 1)
        return torch.cat((v[:, :, :nvis] + position, v[:, :, nvis:]), dim=2)

    def spatial_scores(self, scores):
        """[B, H, Nq, Nk] scores -> the same with the visual scores integrated over each patch neighborhood."""
        if not self.spatial:
            return scores
        b, h, nq, nk = scores.shape
        nvis, pad = self.views * self.grid * self.grid, self.spatial // 2
        if nk < nvis:
            raise ValueError(f"spatial integration expects {nvis} visual tokens first in the memory, got {nk} tokens")
        vis = scores[..., :nvis].reshape(b * h * nq * self.views, 1, self.grid, self.grid)
        vis = F.conv2d(F.pad(vis, (pad, pad, pad, pad), mode="replicate"), self.spatial_kernel.to(vis.dtype))
        return torch.cat((vis.reshape(b, h, nq, nvis), scores[..., nvis:]), dim=-1)

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
        v = self.read_values(v.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2))
        allowed = query_valid & memory_valid.transpose(-1, -2)
        scores = q @ k.transpose(-1, -2)  # integer spike matches of this timestep
        if self.integrate:
            total = scores if state.get("match") is None else state["match"] + scores
            state["match"], state["steps"] = total, state.get("steps", 0) + 1
            scores = total / state["steps"]
        if self.shift:
            scores = scores * 2.0 ** -self.shift
        scores = self.spatial_scores(scores)
        if self.state_tokens:
            s = self.state_tokens
            route = torch.cat((self.routing(scores[..., :-s], allowed[..., :-s]),
                               self.routing(scores[..., -s:], allowed[..., -s:])), dim=-1)
        else:
            route = self.routing(scores, allowed)
        context = (route @ v).transpose(1, 2).reshape(batch, nq, channels)
        live = query_valid & memory_valid.any(dim=1, keepdim=True)
        return self.output(fire("context", context, live)).masked_fill(~live, 0.0)


class SoftmaxCrossAttention(ShiftmaxActionAttention):
    """Diagnostic control, not spike-driven: the continuous-input Shiftmax attention with its spikes removed.

    Same linears and token-local norms and the same constructor (so the same initialization for a given seed),
    but Q/K stay normalized currents instead of ternary PLIF events, routing is softmax(QK^T / sqrt(d)) instead
    of Shiftmax over integer scores, and the normalized context goes to the output projection without the
    context neuron. Used only as the head's cross-attention, to test whether spike routing loses the target's
    position (control C0).
    """

    def __init__(self, hidden_dim, nheads, interaction):
        super().__init__(hidden_dim, nheads, interaction, continuous_inputs=True)
        self.spike_nodes = nn.ModuleDict()  # no unused neuron parameters in the DDP optimizer

    def forward_step(self, query, memory, memory_valid, state, t):
        query_valid = torch.ones((*query.shape[:2], 1), dtype=torch.bool, device=query.device)
        query_input = self.norms["input_query"](query, query_valid, t)
        memory_input = self.norms["input_memory"](memory, memory_valid, t)
        q = self.norms["query"](self.query(query_input), query_valid, t)
        k = self.norms["key"](self.key(memory_input), memory_valid, t)
        v = self.norms["value"](self.value(memory_input), memory_valid, t)
        batch, nq, channels = q.shape
        nk = k.shape[1]
        q = q.reshape(batch, nq, self.nheads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, nk, self.nheads, self.head_dim).transpose(1, 2)
        allowed = (query_valid & memory_valid.transpose(-1, -2))[:, None]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        route = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
        context = (route @ v).transpose(1, 2).reshape(batch, nq, channels)
        live = query_valid & memory_valid.any(dim=1, keepdim=True)
        return self.output(self.norms["context"](context, live, t)).masked_fill(~live, 0.0)


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
            "full_spike_state8_shiftmax_continuous_inputs_softmax_cross",
        }
        softmax_cross = config.head_type == "full_spike_state8_shiftmax_continuous_inputs_softmax_cross"
        ifr = (config.ifr_integrate or config.ifr_shift > 0 or config.ifr_split_state or config.ifr_locate
               or config.ifr_spatial > 0 or config.ifr_norm_bits > 0 or config.ifr_sign_field
               or config.ifr_context_levels > 0)
        layout = config.ifr_locate or config.ifr_spatial or config.ifr_sign_field
        if layout and not (config.ifr_locate_views and config.ifr_locate_grid):
            raise ValueError("ifr_locate / ifr_spatial / ifr_sign_field need the visual layout "
                             "(set by the model before it builds the head)")
        for index, layer in enumerate(self.layers):
            layer.self_attn = ShiftmaxActionAttention(
                hidden_dim, nheads, interaction, continuous_inputs=continuous_inputs,
            )
            if ifr:
                layer.cross_attn = IFRCrossAttention(
                    hidden_dim, nheads, interaction, continuous_inputs=continuous_inputs,
                    integrate=config.ifr_integrate, shift=config.ifr_shift,
                    state_tokens=config.num_state_tokens if config.ifr_split_state else 0,
                    locate_views=config.ifr_locate_views if config.ifr_locate else 0,
                    locate_grid=config.ifr_locate_grid if config.ifr_locate else 0,
                    spatial=config.ifr_spatial, views=config.ifr_locate_views, grid=config.ifr_locate_grid,
                    norm_bits=config.ifr_norm_bits, sign_field=config.ifr_sign_field,
                    context_levels=config.ifr_context_levels,
                )
            else:
                layer.cross_attn = ShiftmaxActionAttention(
                    hidden_dim, nheads, interaction, continuous_inputs=continuous_inputs,
                )
            if softmax_cross:  # same initial weights and RNG stream as D0: copy from the spiking module just built
                with torch.random.fork_rng(devices=[]):
                    control = SoftmaxCrossAttention(hidden_dim, nheads, interaction)
                missing, unexpected = control.load_state_dict(layer.cross_attn.state_dict(), strict=False)
                if missing or any(not key.startswith("spike_nodes.") for key in unexpected):
                    raise RuntimeError(f"softmax control init mismatch: {missing}, {unexpected}")
                layer.cross_attn = control
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
