"""Where does the action head look? Records the head's cross-attention over the image at the probe frames.

The read-out checks (cross_probe.py) show the plate offset lost when the head reads the image, and IFR / locate did
not bring it back. A read can only carry the plate's position if the routing puts weight on the plate's
tokens. This records the cross-attention weight on every visual token, at the same frames and renders as
cross_probe.py (t0 / transport / approach, unshifted render first):

  attn_map  [renders, layers, visual tokens]         averaged over heads, queries and timesteps
  attn_hq   [layers, heads, queries, visual tokens]  unshifted render only, averaged over timesteps

Spiking heads are renormalized per row as in head_probe.py (rows sum to 1 over all keys). Heads specialize, so the
averaged map can hide a head that tracks the plate; attend_analysis.py scores every head. It projects the plate,
the target bowl and the gripper into both cameras and measures how much weight falls on each.

Works for every head cross_probe.py covers (Shiftmax, IFR, C0 softmax) and for the ANN head of TurboVLA.
Run like cross_probe.py: SCRIPT=attend_probe.py counterfactual_runs.sh <out> <trials> <parallel> name=ckpt:weights
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import counterfactual_rollout as cr  # noqa: E402
import cross_probe  # noqa: E402


class AttendTaps(cross_probe.CrossTaps):
    def __init__(self, model):
        self.hq_store = {}
        super().__init__(model)

    @staticmethod
    def _need_weights(module, args, kwargs):  # ANN head: per-head weights [B, H, Nq, Nk]
        kwargs["need_weights"], kwargs["average_attn_weights"] = True, False
        return args, kwargs

    def _ann_attention(self, i):
        def hook(module, args, out):
            self._put(f"cross{i}", out[0])
            w = out[1].float()
            self.mass_store.setdefault(i, []).append(w.mean((1, 2)))  # [B, Nk], as head_probe.py
            self.hq_store.setdefault(i, []).append(w[:1])
        return hook

    def run(self, policy, primaries, wrists, states, instruction):
        self.hq_store = {}
        actions, feats = super().run(policy, primaries, wrists, states, instruction)
        nvis = self.shapes["projected"][1]
        maps = [torch.stack(self.mass_store[i]).mean(0)[:, :nvis] for i in sorted(self.mass_store)]
        feats["attn_map"] = torch.stack(maps, 1).half().cpu().numpy()  # [B, layers, visual tokens], camera-major
        if self.raw:  # spiking heads: routes of every timestep, renormalized per row
            per_layer = []
            for i in sorted(self.raw):
                rows = [route[:1].float() for _, _, route in self.raw[i]]
                per_layer.append(torch.stack([r / r.sum(-1, keepdim=True).clamp_min(1e-12) for r in rows]).mean(0))
        else:
            per_layer = [torch.stack(self.hq_store[i]).mean(0) for i in sorted(self.hq_store)]
        feats["attn_hq"] = torch.cat(per_layer)[..., :nvis].half().cpu().numpy()  # [layers, heads, queries, nvis]
        return actions, feats


def main():
    cr.SHIFTS, cr.PROBE_OFFSETS, cr.FRAMES, cr.Taps = [], 16, ("t0", "transport", "approach"), AttendTaps
    cr.main()


if __name__ == "__main__":
    main()
