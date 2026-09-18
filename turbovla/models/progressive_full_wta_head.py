"""A-shaped, shared-weight ANN -> full signed-PLIF/WTA action head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .action_head import TurboVLAActionHead
from .components.wta_operators import BipolarPLIF, MultiWinnerAttention

class ProgressiveFullWTAHead(TurboVLAActionHead):
    def __init__(self, original, config, interaction):
        nn.Module.__init__(self)
        # The original A modules are the ONLY owners of all linear weights.
        self.state_projection = original.state_projection
        self.decoder = original.decoder
        self.config = config
        self.norms = nn.ModuleDict()
        self.neurons = nn.ModuleDict()
        self.routes = nn.ModuleDict()
        hidden = interaction.hidden_dim
        def site(name, dim, extra_norm=True):
            self.neurons[name] = BipolarPLIF()
            if extra_norm:
                self.norms[name] = nn.LayerNorm(dim, eps=interaction.spike_norm_eps)
        site("state_input", config.state_dim, False)
        site("state_hidden", config.state_hidden_dim)
        for i, layer in enumerate(self.decoder.decoder.layers):
            for kind, topk in (("self", interaction.wta_text_topk), ("cross", interaction.wta_vision_topk)):
                prefix = f"l{i}_{kind}"
                for suffix in ("iq", "im", "q", "k", "v", "ctx"):
                    site(prefix+"_"+suffix, hidden, suffix != "iq")
                self.routes[prefix] = MultiWinnerAttention(topk, hidden//interaction.nheads,
                                                          interaction.wta_surrogate_temperature)
            site(f"l{i}_ff_input", hidden, False)
            site(f"l{i}_ff_hidden", interaction.dim_feedforward)
        for i, linear in enumerate(self.decoder.action_projection.layers):
            site(f"readout{i}", linear.in_features)
        self.nheads = interaction.nheads
        self.register_buffer("transition_alpha", torch.tensor(0., dtype=torch.float64))
        self.register_buffer("transition_updates", torch.tensor(0, dtype=torch.int64))
        self.probe_alpha = None  # Never serialized; only the capacity entry sets this.

    def set_progress(self, completed_steps, total_steps):
        progress = completed_steps / total_steps
        u = max(0., min(1., (progress-self.config.ann_fraction)/self.config.transition_fraction))
        alpha = 0.5 - 0.5*math.cos(math.pi*u)
        if self.probe_alpha is not None:
            alpha = self.probe_alpha
        self.transition_alpha.fill_(alpha)
        self.transition_updates.fill_(completed_steps)
        return alpha

    def _fire(self, name, current, states, norm=None):
        normalized = (self.norms[name] if norm is None else norm)(current)
        events, states[name] = self.neurons[name](normalized, states.get(name))
        return events

    def _attention(self, prefix, module, query, memory, query_norm, states):
        # F.linear uses views of the original packed QKV Parameters, never copies.
        sq = self._fire(prefix+"_iq", query, states, query_norm)
        sm = self._fire(prefix+"_im", memory, states)
        qw, kw, vw = module.in_proj_weight.chunk(3, 0)
        qb, kb, vb = module.in_proj_bias.chunk(3, 0)
        q = self._fire(prefix+"_q", F.linear(sq, qw, qb), states)
        k = self._fire(prefix+"_k", F.linear(sm, kw, kb), states)
        v = self._fire(prefix+"_v", F.linear(sm, vw, vb), states)
        b, nq, c = q.shape
        nk = k.shape[1]
        q = q.reshape(b,nq,self.nheads,c//self.nheads).transpose(1,2)
        k = k.reshape(b,nk,self.nheads,c//self.nheads).transpose(1,2)
        v = v.reshape(b,nk,self.nheads,c//self.nheads).transpose(1,2)
        allowed = torch.ones((b,nq,nk), device=q.device, dtype=torch.bool)
        route = self.routes[prefix](q @ k.transpose(-1,-2), allowed, q, k)
        context = (route @ v).transpose(1,2).reshape(b,nq,c)
        live = route.detach().ne(0).any(-1).any(1).unsqueeze(-1)
        events = self._fire(prefix+"_ctx", context, states).masked_fill(~live,0.)
        return module.out_proj(events).masked_fill(~live,0.)

    def spike_forward(self, memory, state):
        if state.ndim == 3:
            state = state[:, -1]
        states = {}
        currents = []
        sp = self.state_projection
        for t in range(self.config.spike_timesteps):
            x = self._fire("state_input", state, states, sp.net[0])
            x = sp.net[1](x)
            x = self._fire("state_hidden", x, states)
            x = sp.net[4](x).reshape(state.shape[0],sp.num_tokens,sp.hidden_dim)
            tokens = sp.output_norm(x + sp.position)
            mem = torch.cat((memory,tokens),dim=1)
            x = self.decoder.action_queries.weight[None].expand(memory.shape[0],-1,-1)
            for i, layer in enumerate(self.decoder.decoder.layers):
                x = x + self._attention(f"l{i}_self", layer.self_attn, x, x, layer.norm1, states)
                x = x + self._attention(f"l{i}_cross", layer.multihead_attn, x, mem, layer.norm2, states)
                f = self._fire(f"l{i}_ff_input", x, states, layer.norm3)
                f = layer.linear1(f)
                f = self._fire(f"l{i}_ff_hidden", f, states)
                x = x + layer.linear2(f)
            for i, linear in enumerate(self.decoder.action_projection.layers):
                x = linear(self._fire(f"readout{i}", x, states))
            currents.append(x)
        return torch.tanh(torch.stack(currents).mean(0))

    def forward(self, memory, state):
        alpha = float(self.transition_alpha)
        if alpha == 0.:
            return super().forward(memory,state)
        if self.training and torch.is_grad_enabled() and self.config.spike_gradient_checkpointing:
            spike = checkpoint(self.spike_forward,memory,state,use_reentrant=False)
        else:
            spike = self.spike_forward(memory,state)
        if alpha == 1.:
            return spike
        return (1.-alpha)*super().forward(memory,state) + alpha*spike
