"""A's ANN action head with only cross-attention internals replaced by WTA."""
import torch
from torch import nn
from ..action_head import TurboVLAActionHead
from ..deep_spike_action_head import WTAActionAttention


class MinimalCrossWTADecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction, original):
        super().__init__(d_model=hidden_dim, nhead=nheads,
                         dim_feedforward=dim_feedforward, dropout=config.dropout,
                         batch_first=True, norm_first=True)
        # Preserve all ANN parameters, including the original Q/K/V initialization.
        self.load_state_dict(original.state_dict(), strict=True)
        cross = WTAActionAttention(hidden_dim, nheads, interaction,
                                   topk=interaction.wta_vision_topk)
        cross.load_common(self.multihead_attn, self.norm2)
        self.multihead_attn = cross
        self.timesteps = int(config.spike_timesteps)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        # A's ACT decoder supplies no masks. Never silently ignore a new mask.
        if attn_mask is not None or key_padding_mask is not None or is_causal:
            raise ValueError("Minimal cross-WTA requires A's unmasked ACT memory")
        valid = torch.ones((*mem.shape[:2], 1), dtype=torch.bool, device=mem.device)
        state = {}  # Local neuron time; no state crosses blocks or observations.
        currents = [self.multihead_attn.forward_step(x, mem, valid, state, t)
                    for t in range(self.timesteps)]
        return self.dropout2(torch.stack(currents, dim=0).mean(dim=0))


class MinimalCrossWTAActionHead(TurboVLAActionHead):
    def __init__(self, config, hidden_dim, nheads, dim_feedforward, interaction, original):
        nn.Module.__init__(self)
        self.state_projection = original.state_projection
        self.decoder = original.decoder
        for i, layer in enumerate(self.decoder.decoder.layers):
            self.decoder.decoder.layers[i] = MinimalCrossWTADecoderLayer(
                config, hidden_dim, nheads, dim_feedforward, interaction, layer)
        # Inherit A's forward, ACTDecoder, readout and continuous block residuals.
