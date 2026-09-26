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


def robust_smooth(pts, sigma, iters=4, c=1.5, spacing=1.0):
    """Gaussian smoothing along arclength with iteratively re-weighted points:
    short livewire detours (onto side roads, driveways) get down-weighted so
    the result follows the dominant road alignment. Endpoints stay pinned."""
    raw = resample(np.asarray(pts, float), spacing)
    n = len(raw)
    if n < 5 or sigma <= 0:
        return raw
    pad = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / sigma) ** 2)
    w = np.ones(n)
    head = 2 * raw[0] - raw[1:pad + 1][::-1]
    tail = 2 * raw[-1] - raw[-pad - 1:-1][::-1]
    ext = np.vstack([head, raw, tail])
    out = raw
    for _ in range(iters):
        we = np.r_[w[1:pad + 1][::-1], w, w[-pad - 1:-1][::-1]]
        num_x = np.convolve(ext[:, 0] * we, k, mode='same')[pad:pad + n]
        num_y = np.convolve(ext[:, 1] * we, k, mode='same')[pad:pad + n]
        den = np.convolve(we, k, mode='same')[pad:pad + n]
        out = np.stack([num_x / den, num_y / den], 1)
        out[0], out[-1] = raw[0], raw[-1]
        d = np.hypot(*(raw - out).T)
        w = 1.0 / (1.0 + (d / c) ** 2)
    return resample(out, spacing)


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
    return robust_smooth(path, sigma)


def catmull_rom(pts, spacing=1.0, alpha=0.5):
    """Centripetal Catmull-Rom spline through pts (no overshoot), resampled."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return resample(P, spacing)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = [P[1]]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.hypot(*(p1 - p0)), 1e-6) ** alpha
        t2 = t1 + max(np.hypot(*(p2 - p1)), 1e-6) ** alpha
        t3 = t2 + max(np.hypot(*(p3 - p2)), 1e-6) ** alpha
        n = max(2, int(np.hypot(*(p2 - p1)) / spacing * 2))
        for t in np.linspace(t1, t2, n)[1:]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return resample(np.array(out), spacing)


_CENTER = {}


def center_field(width):
    key = round(width, 1)
    if key not in _CENTER:
        Hh, S, V = hsv()
        pave = ((S < 0.36) & (V > 0.38) & (V < 0.98) & ~water_mask()).astype(np.float32)
        p = np.clip(road_prob() * 2, 0, 1)
        f = np.maximum(pave * 0.85, p)
        _CENTER[key] = cv2.GaussianBlur(f, (0, 0), max(0.8, width / 3.0))
    return _CENTER[key]


def _bilinear(F, x, y):
    x = np.clip(x - 0.5, 0, F.shape[1] - 1.001)
    y = np.clip(y - 0.5, 0, F.shape[0] - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = x - x0; fy = y - y0
    return (F[y0, x0] * (1 - fx) * (1 - fy) + F[y0, x0 + 1] * fx * (1 - fy)
            + F[y0 + 1, x0] * (1 - fx) * fy + F[y0 + 1, x0 + 1] * fx * fy)


def center_trace(waypoints, width=8.0, search=None, smooth=10.0, iters=3, pin_ends=True):
    """Spline through waypoints, then shifted along its normals to the centre of
    the pavement band, coarse-to-fine (wide search + heavy offset smoothing first,
    so systematic waypoint error is absorbed without jumping to side roads)."""
    curve = catmull_rom(waypoints, 1.0)
    base = max(3.0, width * 0.6) if search is None else search
    for R, sm in ((base * 2.0, smooth * 2.5), (base * 1.2, smooth * 1.4), (base, smooth), (base * 0.6, smooth * 0.7)):
        curve = _center_pass(curve, width, R, sm, pin_ends)
    return curve


def _center_pass(curve, width, R, smooth, pin_ends):
    F = center_field(width)
    offs = np.arange(-R, R + 0.01, 0.25)
    for _ in range(1):
        t = np.gradient(curve, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        X = curve[:, 0][:, None] + nrm[:, 0][:, None] * offs[None]
        Y = curve[:, 1][:, None] + nrm[:, 1][:, None] * offs[None]
        vals = _bilinear(F, X, Y) - 0.004 * np.abs(offs)[None]
        best = offs[np.argmax(vals, axis=1)]
        conf = vals.max(1) - vals.mean(1)
        best = np.where(conf > 0.03, best, 0.0)
        # robust smoothing of offsets
        from scipy.ndimage import gaussian_filter1d, median_filter
        best = median_filter(best, size=9, mode='nearest')
        best = gaussian_filter1d(best, smooth, mode='nearest')
        if pin_ends:
            ramp = np.minimum(1, np.minimum(np.arange(len(best)), np.arange(len(best))[::-1]) / 6.0)
            best *= ramp
        curve = curve + nrm * best[:, None]
        curve = resample(curve, 1.0)
    return curve
