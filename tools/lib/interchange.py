"""Parametric freeway interchange generator (diamond).

For a freeway F crossing a crossroad X at C, per travel direction:
  exit ramp     : diverges from the right edge of F `ramp_offset` before C
                  (taper), then curves to a terminal on X at `terminal_offset`
                  from F's centreline, arriving parallel to F (=> ~90 deg to X).
  entrance ramp : leaves the same terminal, merges into F's right edge
                  `ramp_offset` after C (taper).
The crossroad is carried over the freeway (bridge) with `clearance_m`.
Right-hand traffic (US). Image coords: y down, right of (dx,dy) is (-dy,dx).
"""
import math
import numpy as np
from shapely.geometry import LineString, Point
from .trace import resample


def _unit(v):
    n = math.hypot(v[0], v[1])
    return np.asarray(v, float) / max(n, 1e-9)


def _tangent(ls, s, h=3.0):
    a = np.asarray(ls.interpolate(max(0.0, s - h)).coords[0])
    b = np.asarray(ls.interpolate(min(ls.length, s + h)).coords[0])
    return _unit(b - a)


def _pt(ls, s):
    return np.asarray(ls.interpolate(min(max(s, 0.0), ls.length)).coords[0])


def bezier(p0, t0, p1, t1, k=0.42, n=None):
    d = math.hypot(*(p1 - p0))
    c0 = p0 + t0 * d * k
    c1 = p1 - t1 * d * k
    n = n or max(8, int(d * 2))
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c0 + 3 * (1 - t) * t * t * c1 + (t ** 3) * p1


def min_radius(pts):
    P = resample(np.asarray(pts), 1.0)
    if len(P) < 5:
        return float('inf')
    d1 = np.gradient(P, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.hypot(d1[:, 0], d1[:, 1]) ** 3
    k = num / np.maximum(den, 1e-9)
    return float(1.0 / max(k[2:-2].max(), 1e-9))


def compass(t):
    ang = math.degrees(math.atan2(-t[1], t[0])) % 360  # 0 = east, 90 = north
    return ['eastbound', 'northbound', 'westbound', 'southbound'][int(((ang + 45) % 360) // 90)]


def diamond(spec, fpts, xpts, hw_f, hw_r, defaults):
    F = LineString(fpts)
    X = LineString(xpts)
    ip = F.intersection(X)
    if ip.is_empty:
        raise ValueError(f"{spec['id']}: freeway and crossroad do not cross")
    if ip.geom_type != 'Point':
        ip = list(ip.geoms)[0]
    sF, sX = F.project(ip), X.project(ip)
    C = np.asarray(ip.coords[0])
    Lr = spec.get('ramp_offset_px', defaults['ramp_offset_px'])
    Dx = spec.get('terminal_offset_px', defaults['terminal_offset_px'])
    tp = spec.get('taper_px', defaults['taper_px'])
    tf = _tangent(F, sF)
    nR = np.array([-tf[1], tf[0]])
    # terminals: points on the crossroad at perpendicular distance Dx from the freeway
    term = {}
    for sgn in (-1, 1):
        s = sX
        for _ in range(400):
            s += sgn * 0.25
            P = _pt(X, s)
            if F.distance(Point(P)) >= Dx:
                break
        side = 1 if np.dot(P - C, nR) > 0 else -1
        term[side] = (P, F.project(Point(P)))
    ramps = []
    for d in (1, -1):
        side = d  # right side of travel: +nR for +F travel, -nR for -F travel
        n = nR * side
        P, sP = term[side]
        s_ex, s_en = sP - d * Lr, sP + d * Lr
        tdir = _tangent(F, sF) * d
        off_in = hw_f - 0.35 * hw_r
        off_out = hw_f + hw_r + 0.4
        # exit
        A = _pt(F, s_ex) + n * off_in
        A2 = _pt(F, s_ex + d * tp) + n * off_out
        t_a = _tangent(F, s_ex + d * tp) * d
        ex = np.vstack([A, bezier(A2, t_a, P, tdir)])
        ex = np.vstack([[A], resample(np.vstack([A, A2]), 1.0)[1:-1], bezier(A2, t_a, P, tdir)])
        # entrance
        B2 = _pt(F, s_en - d * tp) + n * off_out
        B = _pt(F, s_en) + n * off_in
        t_b = _tangent(F, s_en - d * tp) * d
        en = np.vstack([bezier(P, tdir, B2, t_b), resample(np.vstack([B2, B]), 1.0)[1:]])
        cd = compass(tdir)
        ramps.append(dict(pts=ex, kind='exit', direction=cd, terminal=P, freeway_station=s_ex, gore=A))
        ramps.append(dict(pts=en, kind='entrance', direction=cd, terminal=P, freeway_station=s_en, gore=B))
    for r in ramps:
        r['min_radius_px'] = min_radius(r['pts'])
    sep = dict(point=C.tolist(), upper='crossroad', clearance_m=spec.get('clearance_m', defaults['clearance_m']))
    return ramps, sep
