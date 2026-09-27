"""Compatibility import for the hybrid-FFN extension."""

from .extensions.hybrid_action_head import HybridSpikeFFNActionHead, HybridSpikeFFNDecoderLayer

__all__ = ["HybridSpikeFFNActionHead", "HybridSpikeFFNDecoderLayer"]
