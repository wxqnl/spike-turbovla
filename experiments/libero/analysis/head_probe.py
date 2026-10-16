"""Where inside the action head the plate position is lost.

counterfactual_rollout.py found the plate offset linearly decodable up to the fused visual tokens (the action head's input) and not
in the predicted actions. This reruns only the open-loop probes of counterfactual_rollout.py (no closed-loop
branches), 16 plate offsets per frame, at three frames (first frame, start of transport, approach: the first policy
query with the held bowl within 5 cm of the plate in xy), and additionally records inside the action head:

  cross<i>    cross-attention output of decoder layer i (before the residual)
  layer<i>    output of decoder layer i
  readout<j>  output of the j-th linear of the action readout MLP (the last one is the action current before tanh)
  attention_mass  per decoder layer, the cross-attention weight on visual tokens / text tokens / state tokens
                  (spiking head: Shiftmax weights renormalized per row; averaged over heads, queries and timesteps)

The spiking head runs per timestep; its stages keep all timesteps ([B, T, ...]) before the random projection.
Run exactly like counterfactual_rollout.py (counterfactual_runs.sh with SCRIPT=head_probe.py).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import counterfactual_rollout as cr  # noqa: E402


class HeadTaps(cr.Taps):
    def __init__(self, model):
        super().__init__(model)
        self.nstate = int(model.config.action.num_state_tokens)
        self.head_store, self.mass_store = {}, {}
        decoder = model.action_head.decoder
        if hasattr(decoder, "decoder"):  # ANN head: nn.TransformerDecoder, one call per layer
            for i, layer in enumerate(decoder.decoder.layers):
                attn = layer.multihead_attn
                attn.register_forward_pre_hook(self._need_weights, with_kwargs=True)
                attn.register_forward_hook(self._ann_attention(i))
                layer.register_forward_hook(lambda m, a, out, name=f"layer{i}": self._put(name, out))
            linears = [m for m in decoder.action_projection.modules() if isinstance(m, nn.Linear)]
            for j, linear in enumerate(linears):
                linear.register_forward_hook(lambda m, a, out, name=f"readout{j}": self._put(name, out))
        else:  # spiking head: forward_step once per timestep
            self.routing_hooks = []  # cross_probe.py replaces them (IFR routes twice per step, C0 not at all)
            for i, layer in enumerate(decoder.layers):
                self.routing_hooks.append(layer.cross_attn.routing.register_forward_hook(self._spike_routing(i)))
                self._wrap(layer.cross_attn, f"cross{i}")
                self._wrap(layer, f"layer{i}")
            readout = decoder.action_projection

            def readout_step(x, state, t):  # SpikeActionReadout.forward_step, recording each linear's output
                for j, (norm, spike, linear) in enumerate(zip(readout.norms, readout.spike_nodes, readout.layers)):
                    events, state[j] = spike(norm(x, time_index=t), state.get(j))
                    x = linear(events)
                    self._put(f"readout{j}", x)
                return x
            readout.forward_step = readout_step

    def _put(self, name, x):
        self.head_store.setdefault(name, []).append(x.detach())

    def _wrap(self, module, name):
        original = module.forward_step

        def forward_step(*args, **kwargs):
            out = original(*args, **kwargs)
            self._put(name, out)
            return out
        module.forward_step = forward_step

    @staticmethod
    def _need_weights(module, args, kwargs):
        kwargs["need_weights"], kwargs["average_attn_weights"] = True, True
        return args, kwargs

    def _ann_attention(self, i):
        def hook(module, args, out):
            self._put(f"cross{i}", out[0])
            self.mass_store.setdefault(i, []).append(out[1].float().mean(1))  # [B, Nq, Nk] -> [B, Nk]
        return hook

    def _spike_routing(self, i):
        def hook(module, args, weights):  # [B, H, Nq, Nk], rows sum to (0.5, 1]
            p = weights.float() / weights.float().sum(-1, keepdim=True).clamp_min(1e-12)
            self.mass_store.setdefault(i, []).append(p.mean((1, 2)))
        return hook

    def run(self, policy, primaries, wrists, states, instruction):
        self.head_store, self.mass_store = {}, {}
        actions, feats = super().run(policy, primaries, wrists, states, instruction)
        n = len(primaries)
        for name, values in self.head_store.items():
            x = torch.stack(values, 1) if len(values) > 1 else values[0]  # per-timestep calls -> [B, T, ...]
            feats[name] = cr.project(x.reshape(n, -1), name)
        nvis = self.shapes["projected"][1]
        mass = []
        for i in sorted(self.mass_store):
            p = torch.stack(self.mass_store[i]).mean(0)  # [B, Nk], averaged over timesteps
            mass.append(torch.stack([p[:, :nvis].sum(-1), p[:, nvis:-self.nstate].sum(-1),
                                     p[:, -self.nstate:].sum(-1)], -1))
        feats["attention_mass"] = torch.stack(mass, 1).cpu().numpy()  # [B, layers, (visual, text, state)]
        if "head_shapes" not in self.__dict__:
            self.head_shapes = {k: (len(v), tuple(v[0].shape)) for k, v in self.head_store.items()}
            print("[head_probe] head stages (calls, shape)", self.head_shapes, flush=True)
        return actions, feats


def main():
    cr.SHIFTS, cr.PROBE_OFFSETS, cr.FRAMES, cr.Taps = [], 16, ("t0", "transport", "approach"), HeadTaps
    cr.main()


if __name__ == "__main__":
    main()
