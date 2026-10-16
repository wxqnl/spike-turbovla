"""Where inside the spiking head's cross-attention the plate position is lost.

head_probe.py: in the spiking base the plate offset is decodable from the fused tokens (R^2 0.81 at the first frame) and
not from the first cross-attention's output (0.03); TurboVLA keeps 0.53. Layer-0 queries are learned constants
(self-attention over the action queries only), so in layer 0 the plate can only enter through the keys and values
of the memory. This splits ShiftmaxActionAttention.forward_step (same computation, re-implemented with taps) into

  kvis<i>     key spikes of the visual tokens          (ternary PLIF events)
  vvis<i>     values of the visual tokens              (normalized current, not spiking)
  routevis<i> Shiftmax weights on the visual tokens
  ctxvis<i>   the visual tokens' share of the context  (routevis @ vvis)
  ctx<i>      full context before the context neuron  (route @ v, also over text and state tokens)
  ctxspk<i>   context spikes                           (ternary PLIF events, input of the output projection)
  cross<i>    output projection                        (as in head_probe.py)

plus, per render and layer, how much changes relative to the unshifted render of the same state: fraction of
visual key spikes flipped, L1 change of the visual Shiftmax row (summed over keys), fraction of context spikes
flipped. Everything else as head_probe.py; run with SCRIPT=cross_probe.py.

Also covers the IFR cross-attention (integrated, shifted, state split, locate: vvis then include the position values;
routevis is the image-side Shiftmax row) and
the C0 control (softmax over continuous Q/K; kvis are normalized key currents and ctxspk is the normalized context, so
their flip fractions are not meaningful there). Attention mass comes from the full routing row, renormalized.

  where<i>    where the routing looks: per head, query and camera, the centroid (row, column) of the image-side
              weights on the patch grid. Attention reads what is at the attended tokens, not where they are;
              if the plate offset is in where<i> but not in ctxvis<i>, a read of this centroid would carry it.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import counterfactual_rollout as cr  # noqa: E402
import head_probe  # noqa: E402
from turbovla.models.shiftmax_action_head import SoftmaxCrossAttention  # noqa: E402


class CrossTaps(head_probe.HeadTaps):
    def __init__(self, model):
        super().__init__(model)
        decoder = model.action_head.decoder
        self.raw = {}
        if not hasattr(decoder, "decoder"):  # spiking head only
            for hook in self.routing_hooks:
                hook.remove()
            for i, layer in enumerate(decoder.layers):
                layer.cross_attn.forward_step = self._instrumented(layer.cross_attn, i)

    def _instrumented(self, attn, i):
        softmax = isinstance(attn, SoftmaxCrossAttention)
        integrate, shift = getattr(attn, "integrate", False), getattr(attn, "shift", 0)
        split = getattr(attn, "state_tokens", 0)

        def forward_step(query, memory, memory_valid, state, t):  # forward_step of the three cross-attentions
            query_valid = torch.ones((*query.shape[:2], 1), dtype=torch.bool, device=query.device)

            def fire(name, current, valid):
                spikes, state[name] = attn.spike_nodes[name](
                    attn.norms[name](current, valid, t), state.get(name), valid=valid,
                )
                return spikes

            if attn.continuous_inputs:
                query_input = attn.norms["input_query"](query, query_valid, t)
                memory_input = attn.norms["input_memory"](memory, memory_valid, t)
            else:
                query_input = fire("input_query", query, query_valid)
                memory_input = fire("input_memory", memory, memory_valid)
            if softmax:
                q = attn.norms["query"](attn.query(query_input), query_valid, t)
                k = attn.norms["key"](attn.key(memory_input), memory_valid, t)
            else:
                q = fire("query", attn.query(query_input), query_valid)
                k = fire("key", attn.key(memory_input), memory_valid)
            v = attn.norms["value"](attn.value(memory_input), memory_valid, t)
            batch, nq, channels = q.shape
            nk = k.shape[1]
            qh = q.reshape(batch, nq, attn.nheads, attn.head_dim).transpose(1, 2)
            kh = k.reshape(batch, nk, attn.nheads, attn.head_dim).transpose(1, 2)
            vh = v.reshape(batch, nk, attn.nheads, attn.head_dim).transpose(1, 2)
            if hasattr(attn, "read_values"):  # IFR sign field and locate (each unchanged when off)
                vh = attn.read_values(vh)
            elif hasattr(attn, "position_values"):
                vh = attn.position_values(vh)
            allowed = query_valid & memory_valid.transpose(-1, -2)
            scores = qh @ kh.transpose(-1, -2)
            if softmax:
                scores = scores / math.sqrt(attn.head_dim)
                route = torch.softmax(scores.masked_fill(~allowed[:, None], -torch.inf), dim=-1)
            else:
                if integrate:
                    total = scores if state.get("match") is None else state["match"] + scores
                    state["match"], state["steps"] = total, state.get("steps", 0) + 1
                    scores = total / state["steps"]
                if shift:
                    scores = scores * 2.0 ** -shift
                if hasattr(attn, "spatial_scores"):
                    scores = attn.spatial_scores(scores)
                if split:
                    route = torch.cat((attn.routing(scores[..., :-split], allowed[..., :-split]),
                                       attn.routing(scores[..., -split:], allowed[..., -split:])), dim=-1)
                else:
                    route = attn.routing(scores, allowed)
            context = (route @ vh).transpose(1, 2).reshape(batch, nq, channels)
            live = query_valid & memory_valid.any(dim=1, keepdim=True)
            spikes = attn.norms["context"](context, live, t) if softmax else fire("context", context, live)
            out = attn.output(spikes).masked_fill(~live, 0.0)
            self._put(f"ctx{i}", context)
            self._put(f"ctxspk{i}", spikes)
            self._put(f"cross{i}", out)
            p = route.float() / route.float().sum(-1, keepdim=True).clamp_min(1e-12)
            self.mass_store.setdefault(i, []).append(p.mean((1, 2)))
            self.raw.setdefault(i, []).append((k.detach(), vh.detach(), route.detach()))
            return out
        return forward_step

    def run(self, policy, primaries, wrists, states, instruction):
        self.raw = {}
        actions, feats = super().run(policy, primaries, wrists, states, instruction)
        if not self.raw:
            return actions, feats
        n, nvis = len(primaries), self.shapes["projected"][1]
        flips = []
        for i in sorted(self.raw):
            kv, vv, rv, cv = [], [], [], []
            for k, vh, route in self.raw[i]:  # one entry per timestep
                kv.append(k[:, :nvis])
                vv.append(vh[:, :, :nvis])
                rv.append(route[..., :nvis])
                cv.append(route[..., :nvis] @ vh[:, :, :nvis])
            wv = []  # where the routing looks: per camera, the centroid of the image-side weights on the patch grid
            side = int(round((nvis // 2) ** 0.5))
            grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1)
            grid = grid.reshape(-1, 2).float().to(rv[0].device) / (side - 1)
            for r in rv:
                cams = [r[..., c * side * side:(c + 1) * side * side].float() for c in range(2)]
                wv.append(torch.stack([(w / w.sum(-1, keepdim=True).clamp_min(1e-12)) @ grid for w in cams], -2))
            for name, values in (("kvis", kv), ("vvis", vv), ("routevis", rv), ("where", wv), ("ctxvis", cv)):
                feats[f"{name}{i}"] = cr.project(torch.stack(values, 1).reshape(n, -1), f"{name}{i}")
            k_flip = torch.stack([(k != k[:1]).float().mean((1, 2)) for k in kv]).mean(0)
            route_change = torch.stack([(r - r[:1]).abs().sum(-1).mean((1, 2)) for r in rv]).mean(0)
            spk = self.head_store[f"ctxspk{i}"]  # still filled after HeadTaps.run
            spk_flip = torch.stack([(s != s[:1]).float().mean((1, 2)) for s in spk]).mean(0)
            flips.append(torch.stack([k_flip, route_change, spk_flip], -1))
        feats["flips"] = torch.stack(flips, 1).cpu().numpy()  # [B, layers, (key flip, route L1, ctx spike flip)]
        return actions, feats


def main():
    cr.SHIFTS, cr.PROBE_OFFSETS, cr.FRAMES, cr.Taps = [], 16, ("t0", "transport", "approach"), CrossTaps
    cr.main()


if __name__ == "__main__":
    main()
