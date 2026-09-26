"""Per-pixel feature maps derived from the base map (cached)."""
import numpy as np
import cv2
from skimage.filters import sato
from .common import load_rgb, cached


def hsv():
    a = load_rgb()
    h = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32)
    return h[..., 0] * 2.0, h[..., 1] / 255.0, h[..., 2] / 255.0


def lab():
    a = load_rgb()
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def cream_score():
    """How much a pixel looks like pavement / gravel / dirt road surface."""
    H, S, V = hsv()
    hue_ok = np.maximum(smoothstep(0, 12, H) * (1 - smoothstep(58, 75, H)), 1 - smoothstep(0.10, 0.2, S))
    return (smoothstep(0.42, 0.62, V) * (1 - smoothstep(0.30, 0.50, S)) * hue_ok).astype(np.float32)


def ridge_bright(sigmas=(0.7, 1.0, 1.5)):
    def f():
        L = lab()[..., 0] / 255.0
        return sato(L, sigmas=list(sigmas), black_ridges=False).astype(np.float32)
    return cached('sato_bright_' + '_'.join(str(s) for s in sigmas), f)


def water_mask():
    a = load_rgb()
    H, S, V = hsv()
    r, g, b = [a[..., i].astype(int) for i in range(3)]
    return (H >= 180) & (H <= 230) & (S > 0.35) & (V > 0.22) & (b > r + 15)


def road_prob():
    def f():
        c = cream_score()
        s = ridge_bright()
        s = s / np.percentile(s, 99.5)
        s = np.clip(s, 0, 1)
        p = np.sqrt(np.clip(c, 0, 1)) * (0.25 + 0.75 * s)
        p[water_mask()] *= 0.1
        return p.astype(np.float32)
    return cached('road_prob_v1', f)
