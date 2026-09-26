"""Shared helpers for the world-extraction pipeline.

All geometry produced by the pipeline is expressed in SOURCE-IMAGE PIXEL
coordinates of assets/maps/source/base_map.webp (2000 x 667). Pixel centres
are at integer + 0.5 in continuous coordinates; x grows right, y grows down.
See docs/COORDINATES.md.
"""
import json
import os
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PNG = os.path.join(ROOT, 'assets/maps/processed/base_map.png')
CACHE = os.path.join(ROOT, 'tools/.cache')
W, H = 2000, 667
os.makedirs(CACHE, exist_ok=True)


def path(*p):
    return os.path.join(ROOT, *p)


def load_rgb():
    return np.asarray(Image.open(SRC_PNG).convert('RGB'))


def save_json(p, obj, indent=None):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(obj, f, indent=indent, separators=(',', ':') if indent is None else None)
        f.write('\n')


def load_json(p):
    with open(p) as f:
        return json.load(f)


def cached(name, fn):
    """Cache numpy arrays keyed by name in tools/.cache (delete to rebuild)."""
    p = os.path.join(CACHE, name + '.npy')
    if os.path.exists(p):
        return np.load(p)
    v = fn()
    np.save(p, v)
    return v


def rnd(v, n=2):
    return float(round(float(v), n))
