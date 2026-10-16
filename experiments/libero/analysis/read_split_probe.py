"""Why the head's read of the image loses the plate offset: routing change versus content change.

attend_probe.py: the spiking heads look at the plate as much as TurboVLA does (best-head lift 2-3.5 in both, and
TurboVLA's averaged attention is just as diffuse), yet TurboVLA's read keeps the offset (R^2 0.39-0.53) and the
spiking read does not (0.06-0.16). With the visual read c = sum_j r_j v_j, a plate shift changes it by

  dc = sum_j dr_j v_j  +  sum_j r_j dv_j  +  sum_j dr_j dv_j
       route term         value term         cross term

The value term is the shift seen through a fixed routing: what the attended tokens' contents say about the plate.
The route term is the routing reshuffle (threshold jitter for spike keys). Recorded per decoder layer, relative to
the unshifted render, for the visual tokens only:

  rterm<i>   route term          vterm<i>   value term          (projected like every stage; the unshifted row is 0)
  uterm<i>   value term through a uniform routing: the change of each camera's mean value. Whether the plate's
             position survives a plain sum over the image tokens, i.e. whether the tokens' contents say where they are
             (a translation-equivariant code cancels: what the plate leaves on one token it adds to its neighbor)
  mterm<i>   value term with gain-field values: each visual value channel multiplied by a fixed spatial pattern of
             its token's patch (per head, channel pairs get cos / sin of pi * f * row or column, f in 1, 2, 4, 8),
             the same routing. A plain weighted sum of contents loses where they are when the contents do not encode
             it; position-modulated contents keep it (a gain field). Upper bound for a gain-field read.
  sterm<i>   value term with the IFR sign field (IFRCrossAttention.sign_patterns: half of each head's value channels
             multiplied by fixed +-1 square waves of the token's patch, the other half unchanged), the same routing
  usterm<i>  uterm with the sign field: does a plain sum over the image tokens keep the position once the signs are on
  sread<i>   the whole change of the sign-field read (route term included): what the I7 read would carry with these
             weights. The three sign terms are skipped for a head trained with the sign field (its values have it).
  f_<kind><i> / fu_<kind><i>
             with FIELDS=kind,... : the value term (same routing) and the uniform-sum term with other fixed position
             fields on the visual values, to compare field designs at equal weights (see field_table):
             sign_all (square waves on every channel), q5_half / q5_all (cosines rounded to {-1, -1/2, 0, 1/2, 1}:
             shifts only), q4_half / q4_all (levels {-1, -1/2, 1/2, 1}), signfine_half (square waves at scales 2-16), cos_all (unrounded, as mterm)
  split_norms [renders, layers, 7]: mean L2 norm (over heads, queries, timesteps) of the route term, the value term
              and the whole change of the visual read; then the route and value terms again with each routing row
              renormalized over the visual keys (p = r_vis / sum r_vis). The renormalized split drops what only
              rescales a whole row's image weights: Shiftmax's power-of-two normalizer (rows sum to (0.5, 1], so a
              row's weights all halve or double when its sum crosses a power of two) and weight moving to or from
              text / state keys; what remains is the reallocation among the image tokens. Last, the route and value
              terms with the sign-field values (0 when skipped).

The spiking heads use the routing exactly as the module does (rows not renormalized), so the three terms add up to
the change of ctxvis<i>. TurboVLA (ANN head, nn.MultiheadAttention) uses its per-head softmax weights and values
from its own input projection.
Run like cross_probe.py: SCRIPT=read_split_probe.py counterfactual_runs.sh <out> <trials> <parallel> name=ckpt:weights
"""
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import counterfactual_rollout as cr  # noqa: E402
import cross_probe  # noqa: E402
from turbovla.models.shiftmax_action_head import IFRCrossAttention  # noqa: E402


def gain_fields(nvis, views, head_dim, device):
    """[nvis, head_dim] fixed position patterns: channel pairs (cos, sin) of pi * f * row or column, cycling."""
    side = int(round((nvis // views) ** 0.5))
    axis = torch.arange(side, dtype=torch.float32, device=device) / max(side - 1, 1)
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((rows.flatten(), cols.flatten()), -1)  # [side^2, 2]
    patterns = []
    for f in (1, 2, 4, 8):
        for a in range(2):
            angle = math.pi * f * coords[:, a]
            patterns += [angle.cos(), angle.sin()]
    table = torch.stack(patterns, -1)  # [side^2, 16]
    table = table.repeat(1, head_dim // table.shape[1] + 1)[:, :head_dim]
    return table.repeat(views, 1)  # cameras share the patterns, each over its own grid


def split_terms(routes, values, nvis, renormalize=False):
    """routes [T][B, H, Q, K], values [T][B, H, K, d] -> route term, value term, total change, each [B, T, H, Q, d]."""
    rt, vt, dc = [], [], []
    for r, v in zip(routes, values):
        r, v = r[..., :nvis].float(), v[:, :, :nvis].float()
        if renormalize:
            r = r / r.sum(-1, keepdim=True).clamp_min(1e-12)
        dr, dv = r - r[:1], v - v[:1]
        rt.append(dr @ v[:1])
        vt.append(r[:1] @ dv)
        dc.append(r @ v - r[:1] @ v[:1])
    return torch.stack(rt, 1), torch.stack(vt, 1), torch.stack(dc, 1)


def field_table(kind, nvis, views, head_dim, device):
    """[nvis, head_dim] fixed position field: <pattern>_<half|all>, patterns over rows and columns with phases
    pi/4 and 3pi/4 at scales 1, 2, 4, 8 (signfine: 2, 4, 8, 16); half keeps the first half of the channels at 1."""
    pattern, extent = kind.rsplit("_", 1)
    side = int(round((nvis // views) ** 0.5))
    axis = torch.arange(side, dtype=torch.float64) / max(side - 1, 1)
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    scales = (2, 4, 8, 16) if pattern == "signfine" else (1, 2, 4, 8)
    waves = torch.stack([torch.cos(math.pi * sc * coord + phase) for sc in scales
                         for phase in (math.pi / 4, 3 * math.pi / 4) for coord in (rows.flatten(), cols.flatten())], -1)
    if pattern.startswith("sign"):
        waves = torch.where(waves >= 0, 1.0, -1.0).double()
    elif pattern == "q5":
        waves = torch.round(2 * waves) / 2
    elif pattern == "q4":  # {-1, -1/2, 1/2, 1}: no channel is silenced
        waves = torch.where(waves >= 0, 1.0, -1.0).double() * torch.where(waves.abs() > 0.5, 1.0, 0.5).double()
    width = head_dim // 2 if extent == "half" else head_dim
    table = waves.repeat(1, width // waves.shape[1] + 1)[:, :width]
    table = torch.cat((torch.ones(side * side, head_dim - width, dtype=torch.float64), table), -1)
    return table.repeat(views, 1).float().to(device)


class SplitTaps(cross_probe.CrossTaps):
    def __init__(self, model):
        self.ann = {}
        super().__init__(model)
        decoder = model.action_head.decoder
        self.signed = any(getattr(layer.cross_attn, "sign_field", False) for layer in getattr(decoder, "layers", []))
        if hasattr(decoder, "decoder"):  # ANN head: keep each cross-attention's value input
            for i, layer in enumerate(decoder.decoder.layers):
                layer.multihead_attn.register_forward_pre_hook(self._keep_value(i), with_kwargs=True)

    @staticmethod
    def _need_weights(module, args, kwargs):  # ANN head: per-head weights [B, H, Nq, Nk]
        kwargs["need_weights"], kwargs["average_attn_weights"] = True, False
        return args, kwargs

    def _keep_value(self, i):
        def hook(module, args, kwargs):
            value = args[2] if len(args) > 2 else kwargs["value"]
            if not module.batch_first:
                value = value.transpose(0, 1)
            e, h = module.embed_dim, module.num_heads
            w = module.in_proj_weight[2 * e:] if module.in_proj_weight is not None else module.v_proj_weight
            b = module.in_proj_bias[2 * e:] if module.in_proj_bias is not None else None
            v = F.linear(value, w, b)
            self.ann.setdefault(i, {})["v"] = v.reshape(v.shape[0], v.shape[1], h, e // h).transpose(1, 2)
            return args, kwargs
        return hook

    def _ann_attention(self, i):
        def hook(module, args, out):
            self._put(f"cross{i}", out[0])
            w = out[1].float()
            self.mass_store.setdefault(i, []).append(w.mean((1, 2)))  # [B, Nk], as head_probe.py
            self.ann.setdefault(i, {})["r"] = w
        return hook

    def run(self, policy, primaries, wrists, states, instruction):
        self.ann = {}
        actions, feats = super().run(policy, primaries, wrists, states, instruction)
        n, nvis = len(primaries), self.shapes["projected"][1]
        if self.raw:
            layers = {i: ([route for _, _, route in self.raw[i]], [vh for _, vh, _ in self.raw[i]]) for i in self.raw}
        else:
            layers = {i: ([d["r"]], [d["v"]]) for i, d in self.ann.items()}
        norms = []
        for i in sorted(layers):
            rt, vt, dc = split_terms(*layers[i], nvis)
            feats[f"rterm{i}"] = cr.project(rt.reshape(n, -1), f"rterm{i}")
            feats[f"vterm{i}"] = cr.project(vt.reshape(n, -1), f"vterm{i}")
            values = layers[i][1]
            field = gain_fields(nvis, self.model.num_views, values[0].shape[-1], values[0].device)
            mt = torch.stack([r[:1, ..., :nvis].float() @ ((v[:, :, :nvis].float() - v[:1, :, :nvis].float()) * field)
                              for r, v in zip(layers[i][0], values)], 1)
            feats[f"mterm{i}"] = cr.project(mt.reshape(n, -1), f"mterm{i}")
            per_cam = nvis // self.model.num_views
            ut = torch.stack([torch.stack([(v[:, :, c * per_cam:(c + 1) * per_cam].float().mean(2)
                                            - v[:1, :, c * per_cam:(c + 1) * per_cam].float().mean(2))
                                           for c in range(self.model.num_views)], 2) for v in values], 1)
            feats[f"uterm{i}"] = cr.project(ut.reshape(n, -1), f"uterm{i}")
            rn, vn, _ = split_terms(*layers[i], nvis, renormalize=True)
            split = [rt, vt, dc, rn, vn]
            if self.signed:
                split += [torch.zeros_like(rt)] * 2
            else:
                signs = IFRCrossAttention.sign_patterns(self.model.num_views, int(round(per_cam ** 0.5)),
                                                        values[0].shape[-1]).to(values[0].device)
                signed = [torch.cat((v[:, :, :nvis].float() * signs, v[:, :, nvis:].float()), 2) for v in values]
                rs, vs, ds = split_terms(layers[i][0], signed, nvis)
                us = torch.stack([torch.stack([(v[:, :, c * per_cam:(c + 1) * per_cam].mean(2)
                                                - v[:1, :, c * per_cam:(c + 1) * per_cam].mean(2))
                                               for c in range(self.model.num_views)], 2) for v in signed], 1)
                for name, x in (("sterm", vs), ("usterm", us), ("sread", ds)):
                    feats[f"{name}{i}"] = cr.project(x.reshape(n, -1), f"{name}{i}")
                split += [rs, vs]
                for kind in filter(None, os.environ.get("FIELDS", "").split(",")):
                    field = field_table(kind, nvis, self.model.num_views, values[0].shape[-1], values[0].device)
                    fv = [v[:, :, :nvis].float() * field for v in values]
                    ft = torch.stack([r[:1, ..., :nvis].float() @ (v - v[:1]) for r, v in zip(layers[i][0], fv)], 1)
                    fu = torch.stack([torch.stack([(v[:, :, c * per_cam:(c + 1) * per_cam].mean(2)
                                                    - v[:1, :, c * per_cam:(c + 1) * per_cam].mean(2))
                                                   for c in range(self.model.num_views)], 2) for v in fv], 1)
                    feats[f"f_{kind}{i}"] = cr.project(ft.reshape(n, -1), f"f_{kind}{i}")
                    feats[f"fu_{kind}{i}"] = cr.project(fu.reshape(n, -1), f"fu_{kind}{i}")
            norms.append(torch.stack([x.norm(dim=-1).mean((1, 2, 3)) for x in split], -1))
        # [B, layers, (route, value, total, route / value renormalized over image keys, route / value signed)]
        feats["split_norms"] = torch.stack(norms, 1).cpu().numpy()
        return actions, feats


def main():
    cr.SHIFTS, cr.PROBE_OFFSETS, cr.FRAMES, cr.Taps = [], 16, ("t0", "transport", "approach"), SplitTaps
    cr.main()


if __name__ == "__main__":
    main()
