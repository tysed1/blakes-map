"""Livewire road tracer: snaps rough hand-placed waypoints to the true
centreline in the base map using a minimum-cost path through a road-likeness
cost field, then smooths the result into a spline-ready polyline.

The hand waypoints live in data/manual/*.json (the editable source of truth);
this module turns them into precise centrelines reproducibly.
"""
import numpy as np
import cv2
from skimage.graph import route_through_array
from .common import cached, W, H
from .features import road_prob, load_rgb, hsv, water_mask


def _norm(x):
    return np.clip(x / (np.percentile(x, 99.7) + 1e-6), 0, 1)


def pave_dt():
    def f():
        Hh, S, V = hsv()
        pave = ((S < 0.33) & (V > 0.40) & (V < 0.97) & ~water_mask()).astype(np.uint8)
        k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        cl = cv2.morphologyEx(pave, cv2.MORPH_CLOSE, k(3))
        op = cv2.morphologyEx(cl, cv2.MORPH_OPEN, k(5))
        return cv2.distanceTransform(op, cv2.DIST_L2, 5).astype(np.float32)
    return cached('pave_dt_v1', f)


_COST = {}


def cost_field(kind):
    """kind: 'local' (1-3px roads), 'major' (3-6px), 'highway' (6-11px)."""
    if kind in _COST:
        return _COST[kind]
    p = road_prob()
    if kind == 'local':
        s = cv2.GaussianBlur(p, (0, 0), 0.8)
    elif kind == 'major':
        s = _norm(cv2.GaussianBlur(p, (0, 0), 1.4))
        s = 0.7 * s + 0.3 * _norm(cv2.GaussianBlur(pave_dt(), (0, 0), 1.0))
    else:
        s = _norm(cv2.GaussianBlur(p, (0, 0), 2.4))
        s = 0.5 * s + 0.5 * _norm(cv2.GaussianBlur(pave_dt(), (0, 0), 1.2))
    s = np.clip(s, 0, 1)
    c = 1.0 / (0.03 + s) ** 1.6
    _COST[kind] = c.astype(np.float64)
    return _COST[kind]


def route(p0, p1, kind='local', margin=18, cost=None):
    cost = cost if cost is not None else cost_field(kind)
    x0, y0 = int(np.clip(round(p0[0] - 0.5), 0, W - 1)), int(np.clip(round(p0[1] - 0.5), 0, H - 1))
    x1, y1 = int(np.clip(round(p1[0] - 0.5), 0, W - 1)), int(np.clip(round(p1[1] - 0.5), 0, H - 1))
    bx0, bx1 = max(0, min(x0, x1) - margin), min(W, max(x0, x1) + margin + 1)
    by0, by1 = max(0, min(y0, y1) - margin), min(H, max(y0, y1) + margin + 1)
    sub = cost[by0:by1, bx0:bx1]
    idx, _ = route_through_array(sub, (y0 - by0, x0 - bx0), (y1 - by0, x1 - bx0), fully_connected=True, geometric=True)
    return [(c + bx0 + 0.5, r + by0 + 0.5) for r, c in idx]


def smooth_polyline(pts, sigma, fixed_ends=True, spacing=1.0):
    """Gaussian smoothing along arclength with pinned endpoints, then uniform resample."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    # resample to uniform spacing first
    pts = resample(pts, spacing)
    if sigma > 0 and len(pts) > 3:
        n = len(pts)
        pad = int(3 * sigma) + 1
        # reflect-pad around endpoints (odd reflection keeps endpoint & tangent)
        head = 2 * pts[0] - pts[1:pad + 1][::-1]
        tail = 2 * pts[-1] - pts[-pad - 1:-1][::-1]
        ext = np.vstack([head, pts, tail])
        k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / sigma) ** 2)
        k /= k.sum()
        xs = np.convolve(ext[:, 0], k, mode='same')[pad:pad + n]
        ys = np.convolve(ext[:, 1], k, mode='same')[pad:pad + n]
        out = np.stack([xs, ys], 1)
        if fixed_ends:
            out[0], out[-1] = pts[0], pts[-1]
        pts = out
    return pts


def resample(pts, spacing=1.0):
    pts = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] < 1e-9:
        return pts[:1]
    n = max(2, int(np.ceil(d[-1] / spacing)) + 1)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], 1)


def rdp(pts, eps):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        a, b = pts[i], pts[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = pts[i + 1:j]
        if len(seg) == 0:
            continue
        if L < 1e-9:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def trace(waypoints, kind='local', sigma=None, snap=True, margin=18):
    """Snap waypoints to centreline; returns dense smoothed polyline (Nx2)."""
    wp = [tuple(map(float, p)) for p in waypoints]
    if snap:
        path = [wp[0]]
        for a, b in zip(wp[:-1], wp[1:]):
            seg = route(a, b, kind, margin)
            path += seg[1:]
        path[0] = wp[0]
        path[-1] = wp[-1]
    else:
        path = wp
    if sigma is None:
        sigma = {'local': 2.0, 'major': 3.0, 'highway': 5.0}[kind]
    return smooth_polyline(path, sigma)
