"""Training-only RGB perturbation; labels and evaluation stay unchanged."""

import numpy as np
from PIL import Image, ImageEnhance


def light_augment_rgb(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected real uint8 HWC RGB frame")
    h, w = image.shape[:2]
    dx, dy = np.random.randint(-4, 5, size=2)
    padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode="edge")
    shifted = padded[4 + dy:4 + dy + h, 4 + dx:4 + dx + w]
    brightness, contrast = np.random.uniform(0.9, 1.1, size=2)
    out = ImageEnhance.Brightness(Image.fromarray(shifted)).enhance(float(brightness))
    out = ImageEnhance.Contrast(out).enhance(float(contrast))
    return np.array(out, dtype=np.uint8, copy=True)
