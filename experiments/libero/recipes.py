"""Named ablations; independent encoder/backend overrides remain available."""

PRESETS = {
    "ann": ("dinov3", "bert", "ann"),
    "vision": ("sdtv3_19m", "bert", "ann"),
    "vision_text": ("sdtv3_19m", "smoothspike_bert", "ann"),
    "vision_text_cross": ("sdtv3_19m", "smoothspike_bert", "spike_sdsa"),
}
VISION_CHOICES = ("dinov3", "sdtv3_19m")
TEXT_CHOICES = ("bert", "smoothspike_bert")
CROSS_CHOICES = ("ann", "spike_sdsa")
