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


def _offset_run(F, s0, s1, n_side, o0, o1, step=1.0):
    """Points following the freeway between stations s0 -> s1 at a lateral offset that
    blends smoothly (smoothstep) from o0 to o1 on the side given by n_side (+1/-1 * right normal)."""
    m = max(2, int(abs(s1 - s0) / step) + 1)
    out = []
    sgn = 1 if np.dot(n_side, np.array([-_tangent(F, s0)[1], _tangent(F, s0)[0]])) > 0 else -1
    for i, s in enumerate(np.linspace(s0, s1, m)):
        t = i / (m - 1)
        o = o0 + (o1 - o0) * t * t * (3 - 2 * t)
        tf = _tangent(F, s)
        nr = np.array([-tf[1], tf[0]]) * sgn
        out.append(_pt(F, s) + nr * o)
    return np.asarray(out)


def ramp_radius_m(pts, trim_m=15.0):
    """Validator-style minimum radius (3-chord circumradius at 2 px spacing), metres."""
    P = resample(np.asarray(pts), 2.0)
    best = float('inf')
    k0 = int(trim_m / 5.0) + 3
    for i in range(max(3, k0), len(P) - max(3, k0)):
        a, b, c = P[i - 3], P[i], P[i + 3]
        ab, bc, ca = np.hypot(*(b - a)), np.hypot(*(c - b)), np.hypot(*(a - c))
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        best = min(best, ab * bc * ca / max(2 * area2, 1e-9) * 2.5)
    return best


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
        tdir = _tangent(F, sF) * d
        cd = compass(tdir)
        by = spec.get('ramp_offset_by', {})
        s_ex = sP - d * by.get(f'exit_{cd}', Lr)
        s_en = sP + d * by.get(f'entrance_{cd}', Lr)
        # arrival/departure direction at the terminal: square to the crossroad (a ramp
        # terminal is a stop/yield-controlled T), leaning toward the freeway direction
        # only as far as the skew requires (max ~20 deg off square)
        tx = _tangent(X, X.project(Point(P)))
        perp = np.array([-tx[1], tx[0]])
        if np.dot(perp, tdir) < 0:
            perp = -perp
        lean = spec.get('terminal_lean', defaults.get('terminal_lean', 0.35))
        tterm = _unit(perp * (1 - lean) + tdir * lean)
        off_in = hw_f - 0.35 * hw_r
        off_out = hw_f + hw_r + 0.6
        k0 = spec.get('curve_k', defaults.get('curve_k', 0.5))
        # exit: parallel lane-drop taper along the (possibly curved) freeway edge, then a
        # G1-continuous cubic to the terminal
        ex_par = _offset_run(F, s_ex, s_ex + d * tp, n, off_in, off_out)
        A2 = ex_par[-1]
        t_a = _tangent(F, s_ex + d * tp) * d
        ex = np.vstack([ex_par[:-1], bezier(A2, t_a, P, tterm, k=k0)])
        # entrance: cubic from the terminal, then a parallel acceleration lane + taper
        en_par = _offset_run(F, s_en - d * tp, s_en, n, off_out, off_in)
        B2 = en_par[0]
        t_b = _tangent(F, s_en - d * tp) * d
        en = np.vstack([bezier(P, tterm, B2, t_b, k=k0), en_par[1:]])
        A, B = ex[0], en[-1]
        ramps.append(dict(pts=ex, kind='exit', direction=cd, terminal=P, freeway_station=s_ex, gore=A))
        ramps.append(dict(pts=en, kind='entrance', direction=cd, terminal=P, freeway_station=s_en, gore=B))
    for r in ramps:
        r['min_radius_px'] = ramp_radius_m(r['pts']) / 2.5
    sep = dict(point=C.tolist(), upper='crossroad', clearance_m=spec.get('clearance_m', defaults['clearance_m']))
    return ramps, sep
