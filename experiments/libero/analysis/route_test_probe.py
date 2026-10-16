"""Inference-time routing interventions on a trained spiking head, measured with read_split_probe.py's taps.

read_split_probe.py splits the change of the head's visual read under a plate shift into a route term (routing
reshuffle) and a value term (contents seen through a fixed routing). In the spiking heads the route term is as large
as the value term (TurboVLA: a third of it), so threshold jitter of the key spikes buries the plate offset. Two
interventions on the routing of every cross-attention, without retraining:

  ROUTE_MODE=freeze   every render uses the routing of the unshifted render (row 0 of the probe batch), per
                      timestep; only the contents change. Does the rest of the head pass the value term to the actions?
  ROUTE_MODE=blur3 / blur5
                      before Shiftmax, the visual scores are averaged over each token's 3 x 3 (5 x 5) patch neighborhood
                      with a binomial kernel ([1,2,1] or [1,4,6,4,1], outer product, sum 16 or 256: shifts and adds),
                      per camera, replicate padding. A sub-patch shift flips key spikes in many tokens with little
                      spatial coherence, while the target's evidence spans several patches; spatial integration should
                      shrink the route term by about the kernel's effective sample count and keep the target.

  ROUTE_MODE=exactnorm
                      Shiftmax divides by the next power of two above the row sum (rows sum to (0.5, 1]), so a whole
                      row's weights halve or double when its sum crosses a power of two; exactnorm divides by the sum
                      itself (2**x / sum 2**x), same numerators.
  ROUTE_MODE=gain2 / exactnorm_gain2
                      scores shifted right by 2 more bits before routing (a 4x lower routing gain), with the original
                      or the exact normalizer.
  ROUTE_MODE=finenorm3 / finenorm4
                      the normalizer's exponent rounded up to 1/8 (1/16) instead of an integer (IFR norm_bits).
  ROUTE_MODE=finenorm4_sign
                      finenorm4 plus the IFR sign field switched on in the forward pass (I7's read with these weights):
                      the signed read then goes through the trained context neuron and the rest of the head, so ctxspk<i>
                      shows how much of the signed read survives the spike conversion. The context LayerNorm's per-channel
                      affine was trained on unsigned contexts, so later stages are indicative only.
  ROUTE_MODE=kcur     diagnostic, not a candidate: the key neuron passes its normalized current instead of spikes, so
                      the routing sees continuous keys (spike queries); is the key threshold the source of the jitter?

  ROUTE_MODE=asis     the model's own routing (with CTX_MODE below).

CTX_MODE (optional, default asis) swaps the context neuron of every cross-attention, keeping its trained decay and
threshold. Plate shifts flip 11-26% of the context spikes, yet the spikes carry less of the offset than the context
current does (IFR heads: ctx0 R^2 0.06-0.11 -> ctxspk0 0.01-0.03). The hard reset of BipolarPLIF drops how far the
membrane overshot the threshold, so over 4 timesteps a near-constant input (IFR integrates the routing) maps to a coarse
staircase:
  soft     reset by subtraction (membrane - spike * threshold): the residual carries over, a sigma-delta code
  int4     integer spikes clamp(round(v / threshold), -4, 4) with the soft reset (I-LIF style; 4 binary events per step
           at inference)
  cur      the normalized current itself (upper bound of what any spike code can keep)
Only ctxspk<i> of layer 0 is clean; later stages see a perturbed head.

FFN_MODE / READOUT_MODE (optional, same choices as CTX_MODE) swap the decoder layers' feed-forward neurons (input and
hidden) and the action readout's neurons the same way. I9 8k keeps half of the context current in its integer context
spikes (cross0 R^2 0.10), yet layer0 (residual + spiking feed-forward) is at 0: do the feed-forward / readout spikes bury
the cross-attention's position change, and does an integer code or the current keep it?

The trained weights were not adapted to the intervention, so absolute R^2 and actions are only indicative; the route
and value terms show how much routing noise each mode removes. Spiking heads only (Shiftmax / IFR).
Run like cross_probe.py: ROUTE_MODE=... SCRIPT=route_test_probe.py counterfactual_runs.sh <out> <trials> ...
"""
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from turbovla.models.shiftmax_action_head import IFRCrossAttention

sys.path.insert(0, str(Path(__file__).resolve().parent))
import counterfactual_rollout as cr  # noqa: E402
import read_split_probe  # noqa: E402

KERNELS = {"blur3": (1.0, 2.0, 1.0), "blur5": (1.0, 4.0, 6.0, 4.0, 1.0)}


def spatial_blur(scores, nvis, views, kernel):
    """[B, H, Q, K] scores -> the same with the first nvis (views x G x G, raster) scores binomially smoothed."""
    b, h, q, k = scores.shape
    side = int(round((nvis // views) ** 0.5))
    taps = torch.tensor(kernel, dtype=scores.dtype, device=scores.device)
    weight = (taps[:, None] * taps[None, :]) / taps.sum() ** 2
    pad = len(kernel) // 2
    vis = scores[..., :nvis].reshape(b * h * q * views, 1, side, side)
    vis = F.conv2d(F.pad(vis, (pad, pad, pad, pad), mode="replicate"), weight[None, None])
    return torch.cat((vis.reshape(b, h, q, nvis), scores[..., nvis:]), -1)


MODES = ("freeze", "blur3", "blur5", "exactnorm", "gain2", "exactnorm_gain2", "kcur", "finenorm3", "finenorm4", "finenorm4_sign", "asis")


def fine_shiftmax(scores, allowed, bits):
    """Shiftmax with the normalizer's exponent rounded up to 1/2**bits instead of to an integer (jumps <= 2**(1/2**bits))."""
    if allowed.ndim == 3:
        allowed = allowed[:, None]
    allowed = allowed.expand_as(scores)
    live = allowed.any(-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    maximum = torch.where(live, masked.amax(-1, keepdim=True), torch.zeros_like(scores[..., :1]))
    numerator = torch.exp2(masked - maximum)
    step = 2.0 ** bits
    exponent = torch.ceil(torch.log2(numerator.sum(-1, keepdim=True).clamp_min(1)) * step) / step
    return numerator / torch.exp2(exponent)


def exact_shiftmax(scores, allowed):
    """2**x / sum 2**x over the allowed keys (Shiftmax's numerators with the exact normalizer)."""
    if allowed.ndim == 3:
        allowed = allowed[:, None]
    allowed = allowed.expand_as(scores)
    w = torch.softmax(scores.masked_fill(~allowed, -torch.inf) * math.log(2.0), dim=-1)
    return torch.where(allowed.any(-1, keepdim=True), w, torch.zeros_like(w))


class RoutingIntervention(nn.Module):
    def __init__(self, routing, mode, nvis, views):
        super().__init__()
        self.routing, self.mode, self.nvis, self.views = routing, mode, nvis, views

    def forward(self, scores, allowed):
        if self.mode in KERNELS:
            scores = spatial_blur(scores, self.nvis, self.views, KERNELS[self.mode])
        if self.mode.endswith("gain2"):
            scores = scores * 0.25
        if self.mode.startswith("exactnorm"):
            route = exact_shiftmax(scores, allowed)
        elif self.mode.startswith("finenorm"):
            route = fine_shiftmax(scores, allowed, int(self.mode[len("finenorm")]))
        else:
            route = self.routing(scores, allowed)
        if self.mode == "asis":
            return self.routing(scores, allowed)
        if self.mode == "freeze":
            route = route[:1].expand_as(route)
        return route


class CurrentPassthrough(nn.Module):
    """Stands in for the key neuron: returns the normalized current (zero where invalid) and leaves the state as is."""

    def forward(self, current, state=None, valid=None):
        return (current if valid is None else current.masked_fill(~valid, 0.0)), state


class ContextNeuron(nn.Module):
    """BipolarPLIF's charging with a soft reset, binary or integer events (trained decay / threshold reused)."""

    def __init__(self, plif, levels):
        super().__init__()
        self.plif, self.levels = plif, levels

    def forward(self, current, membrane=None, threshold_scale=1.0, valid=None):
        if membrane is None:
            membrane = torch.zeros_like(current)
        voltage = membrane + self.plif.decay_logit.sigmoid() * (current - membrane)
        threshold = self.plif.log_threshold.exp() * threshold_scale
        if self.levels == 1:
            spike = (voltage >= threshold).float() - (voltage <= -threshold).float()
        else:
            spike = torch.clamp(torch.round(voltage / threshold), -self.levels, self.levels)
        if valid is not None:
            spike = spike.masked_fill(~valid, 0.0)
            voltage = voltage.masked_fill(~valid, 0.0)
        return spike, voltage - spike * threshold


CTX_MODES = {"asis": None, "soft": 1, "int2": 2, "int4": 4, "cur": 0}


def swapped(plif, mode):
    """The CTX_MODES replacement for one trained BipolarPLIF (mode != asis)."""
    return CurrentPassthrough() if mode == "cur" else ContextNeuron(plif, CTX_MODES[mode])


class RouteTestTaps(read_split_probe.SplitTaps):
    def __init__(self, model):
        super().__init__(model)
        decoder = model.action_head.decoder
        if hasattr(decoder, "decoder"):
            raise ValueError("route_test_probe.py intervenes on spiking heads only")
        mode = os.environ["ROUTE_MODE"]
        if mode not in MODES:
            raise ValueError(f"unknown ROUTE_MODE {mode}")
        views = int(model.num_views)
        nvis = views * int(model.vision_encoder.num_patches)  # visual tokens come first in the head's memory
        for layer in decoder.layers:
            attn = layer.cross_attn
            attn.routing = RoutingIntervention(attn.routing, mode, nvis, views)
            if mode == "kcur":
                attn.spike_nodes["key"] = CurrentPassthrough()
            ctx_mode = os.environ.get("CTX_MODE", "asis")
            if ctx_mode not in CTX_MODES:
                raise ValueError(f"unknown CTX_MODE {ctx_mode}")
            if ctx_mode == "cur":
                attn.spike_nodes["context"] = CurrentPassthrough()
            elif ctx_mode != "asis":
                attn.spike_nodes["context"] = ContextNeuron(attn.spike_nodes["context"], CTX_MODES[ctx_mode])
            if mode.endswith("_sign"):
                if not hasattr(attn, "read_values"):
                    raise ValueError("finenorm4_sign needs an IFR cross-attention")
                pattern = IFRCrossAttention.sign_patterns(views, int(round((nvis // views) ** 0.5)), attn.head_dim)
                attn.register_buffer("sign_pattern", pattern.to(next(attn.parameters()).device), persistent=False)
                attn.sign_field = True
        for env in ("FFN_MODE", "READOUT_MODE"):
            if os.environ.get(env, "asis") not in CTX_MODES:
                raise ValueError(f"unknown {env} {os.environ[env]}")
        ffn_mode, readout_mode = os.environ.get("FFN_MODE", "asis"), os.environ.get("READOUT_MODE", "asis")
        if ffn_mode != "asis":
            for layer in decoder.layers:
                for name in ("input", "hidden"):
                    layer.spike_nodes[name] = swapped(layer.spike_nodes[name], ffn_mode)
        if readout_mode != "asis":
            nodes = decoder.action_projection.spike_nodes
            for j in range(len(nodes)):
                nodes[j] = swapped(nodes[j], readout_mode)
        if mode.endswith("_sign"):
            self.signed = True  # values carry the sign field now: rterm / vterm / ctxvis are the signed read


def main():
    cr.SHIFTS, cr.PROBE_OFFSETS, cr.FRAMES, cr.Taps = [], 16, ("t0", "transport", "approach"), RouteTestTaps
    cr.main()


if __name__ == "__main__":
    main()
