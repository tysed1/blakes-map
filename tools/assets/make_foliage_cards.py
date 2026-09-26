"""Procedurally painted foliage cluster cards (RGBA + normal, 1024^2) for leaf-card trees.

v2 (stylized realism, graphics ref.png): every card is a *dense leaf cluster* - hundreds of
small, correctly scaled leaves (card ~1.6 m wide -> leaves 6-14 cm) layered as a soft dome
with a ragged outline, instead of a few big leaves on radiating twigs. Leaves carry:
  * value / hue jitter per leaf (a crown never reads as one flat colour),
  * baked self-shadow (lower layers darker; lifted, never black),
  * a per-leaf normal (random tilt + cupping) in the normal map, so sun glints break up
    the crown into small facets like real foliage.
Cards are near-neutral (light grey-green); the per-instance tint sets species / autumn hue
(see lib_trees.foliage_material). Conifer cards: soft white-pine needle tufts, flat hemlock
sprays. Shrub cards: rhododendron / mountain-laurel rosettes and generic brush.

Outputs assets/foliage/card_<name>.png + card_<name>_nrm.png and _preview.jpg.
    python3 tools/assets/make_foliage_cards.py
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'assets/foliage')
os.makedirs(OUT, exist_ok=True)
SS = 2048          # supersampled canvas
RES = 1024
BASE = np.array([196, 204, 176], np.float32)   # neutral light grey-green (tint does the hue)


def leaf_poly(shape, L, Wd, n=22):
    """Leaf outline along +x from petiole (0,0) to tip (L,0)."""
    pts = []
    for i in range(n + 1):
        t = i / n
        s = math.sin(math.pi * t)
        if shape == 'ovate':        # dogwood / generic
            w = Wd * s ** 0.8 * (1.12 - 0.4 * t)
        elif shape == 'lance':      # hickory leaflet / laurel
            w = Wd * 0.6 * s ** 0.9
        elif shape == 'ellip':      # rhododendron
            w = Wd * 0.55 * s ** 0.6
        elif shape == 'oak':        # rounded lobes (white oak)
            w = Wd * (0.5 + 0.5 * abs(math.sin(t * math.pi * 3.5)) ** 0.7) * s ** 0.65
        elif shape == 'maple':      # palmate-ish silhouette in profile
            w = Wd * (0.35 + 0.65 * abs(math.sin(t * math.pi * 2.5)) ** 1.4) * s ** 0.45
        elif shape == 'tulip':      # tulip poplar: broad, squared/notched tip
            w = Wd * (0.95 * s ** 0.5) * (1.0 if t < 0.8 else 0.55 + 2.2 * (t - 0.8))
        else:
            w = Wd * s
        pts.append((L * t, w))
    return pts + [(x, -y) for x, y in reversed(pts)]


def xf(pts, ox, oy, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(ox + x * c - y * s, oy + x * s + y * c) for x, y in pts]


def _enc_n(n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return tuple(int(v) for v in np.clip((n * 0.5 + 0.5) * 255, 0, 255))


class Card:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.col = Image.new('RGBA', (SS, SS), (0, 0, 0, 0))
        self.nrm = Image.new('RGB', (SS, SS), (128, 128, 255))
        self.dc = ImageDraw.Draw(self.col)
        self.dn = ImageDraw.Draw(self.nrm)

    def twig(self, pts, w, col=(70, 56, 42)):
        self.dc.line(pts, fill=col + (255,), width=max(1, int(w)))

    def leaf(self, poly, shade, hue, nvec, rib=True, tip=None, base=None):
        c = BASE * shade * np.array(hue)
        c = tuple(int(v) for v in np.clip(c, 0, 255))
        self.dc.polygon(poly, fill=c + (255,), outline=tuple(int(v * 0.82) for v in c) + (255,))
        self.dn.polygon(poly, fill=_enc_n(nvec))
        if rib and tip is not None:
            self.dc.line([base, tip], fill=tuple(int(v * 0.8) for v in c) + (255,), width=max(1, int(SS * 0.0012)))

    def finish(self):
        im = self.col.resize((RES, RES), Image.LANCZOS)
        nm = self.nrm.resize((RES, RES), Image.LANCZOS)
        arr = np.asarray(im).astype(np.float32)
        al = arr[..., 3] > 8
        # bleed colour into transparent texels (no dark fringes when mip-mapped / filtered)
        _, (iy, ix) = ndi.distance_transform_edt(~al, return_indices=True)
        arr[..., :3] = arr[iy, ix, :3]
        n = np.asarray(nm).astype(np.float32)
        n = n[iy, ix]
        return Image.fromarray(arr.astype(np.uint8)), Image.fromarray(n.astype(np.uint8))


def _rand_tilt(rng, amt=0.55):
    return (rng.uniform(-amt, amt), rng.uniform(-amt, amt), 1.0)


def paint_broadleaf(shape, seed, n_leaves=520, leaf_len=(0.05, 0.075), aspect=(0.4, 0.55), radius=0.36,
                    compound=False, clump=7):
    """Dense dome of small leaves. leaf_len as fraction of the card width."""
    C = Card(seed); rng = C.rng
    cx = cy = SS / 2
    # overlapping leaf SPRAYS (twig + alternate leaves angled forward) criss-crossing the card:
    # reads as natural foliage, no radial rosette pattern
    leaves = []
    n_sprays = max(8, int(n_leaves / 12))
    for k in range(n_sprays):
        a0 = rng.uniform(0, 2 * math.pi); r0 = math.sqrt(rng.random()) * 0.3 * SS
        x0, y0 = cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)
        ang = rng.uniform(0, 2 * math.pi); bend = rng.normal(0, 0.5)
        TL = rng.uniform(0.09, 0.2) * SS
        sdepth = rng.random()
        pts = [(x0 + math.cos(ang + bend * t) * TL * t, y0 + math.sin(ang + bend * t) * TL * t) for t in np.linspace(0, 1, 14)]
        C.twig(pts, SS * 0.0035)
        nl = int(n_leaves / n_sprays)
        for j in range(nl):
            t = 0.12 + 0.88 * (j + rng.random()) / nl
            q = int(t * 13); px, py = pts[q]
            ta = ang + bend * t
            side = 1 if j % 2 else -1
            la = ta + side * rng.uniform(0.7, 1.4)
            L = rng.uniform(*leaf_len) * SS * (1.05 - 0.35 * t)
            tipx, tipy = px + math.cos(la) * L, py + math.sin(la) * L
            if math.hypot(tipx - cx, tipy - cy) > 0.47 * SS:
                continue
            depth = min(1, max(0, sdepth + rng.normal(0, 0.12)))
            leaves.append((depth, px, py, la, L, rng.uniform(*aspect)))
    leaves.sort(key=lambda t: -t[0])  # deep first
    for depth, px, py, ang, L, asp in leaves:
        shade = (1.0 - 0.42 * depth) * rng.uniform(0.84, 1.1)
        warm = rng.normal(0, 0.025)
        hue = (1 + warm, 1 + warm * 0.4, 1 - warm * 0.5)
        nv = _rand_tilt(rng)
        if compound:
            # pinnate compound leaf (hickory): rachis + 5 leaflets
            tip = (px + math.cos(ang) * L * 1.6, py + math.sin(ang) * L * 1.6)
            C.twig([(px, py), tip], SS * 0.0015, (110, 110, 80))
            for j, t in enumerate((0.35, 0.35, 0.65, 0.65, 1.0)):
                bx, by = px + math.cos(ang) * L * 1.6 * t * 0.95, py + math.sin(ang) * L * 1.6 * t * 0.95
                sa = ang + (0 if j == 4 else (1 if j % 2 else -1) * 0.9)
                ll = L * (0.75 if j < 4 else 0.9)
                poly = xf(leaf_poly('lance', ll, ll * asp), bx, by, sa)
                C.leaf(poly, shade * rng.uniform(0.95, 1.05), hue, nv, rib=False)
            continue
        poly = xf(leaf_poly(shape, L, L * asp), px, py, ang)
        C.leaf(poly, shade, hue, nv, True, (px + math.cos(ang) * L, py + math.sin(ang) * L), (px, py))
    return C.finish()


def paint_pine(seed, n_tufts=34, radius=0.36):
    """Eastern white pine: long soft needles in tufts at twig ends, blue-green, airy but full."""
    C = Card(seed); rng = C.rng
    cx = cy = SS / 2
    tufts = []
    for k in range(n_tufts):
        a = rng.uniform(0, 2 * math.pi); r = math.sqrt(rng.random()) * radius * 0.78 * SS
        tufts.append((cx + r * math.cos(a), cy + r * math.sin(a), rng.random()))
    for tx, ty, _ in tufts:
        C.twig([(cx, cy), ((cx + tx) / 2 + rng.normal(0, 10), (cy + ty) / 2 + rng.normal(0, 10)), (tx, ty)], SS * 0.004, (84, 66, 48))
    tufts.sort(key=lambda t: -t[2])
    for tx, ty, depth in tufts:
        base_ang = math.atan2(ty - cy, tx - cx)
        nl = int(rng.integers(55, 80))
        for j in range(nl):
            a = base_ang + rng.normal(0, 0.9)
            L = rng.uniform(0.07, 0.11) * SS   # ~ 10-15 cm needles on a 1.4 m card
            shade = (1 - 0.35 * depth) * rng.uniform(0.8, 1.12)
            c = tuple(int(v) for v in np.clip(BASE * shade * np.array((0.93, 1.0, 1.04)), 0, 255))
            bend = rng.normal(0, 0.25)
            p0 = (tx + rng.normal(0, 6), ty + rng.normal(0, 6))
            pm = (p0[0] + math.cos(a) * L * 0.5, p0[1] + math.sin(a) * L * 0.5)
            p1 = (pm[0] + math.cos(a + bend) * L * 0.5, pm[1] + math.sin(a + bend) * L * 0.5)
            C.dc.line([p0, pm, p1], fill=c + (255,), width=max(2, int(SS * 0.0022)))
            C.dn.line([p0, pm, p1], fill=_enc_n(_rand_tilt(rng, 0.6)), width=max(2, int(SS * 0.0022)))
    return C.finish()


def paint_hemlock(seed, n_sprays=16, radius=0.4):
    """Eastern hemlock: flat, lacy, drooping sprays of short needles (dense, dark)."""
    C = Card(seed); rng = C.rng
    cx = cy = SS / 2
    sprays = []
    for k in range(n_sprays):
        a = 2 * math.pi * k / n_sprays + rng.normal(0, 0.2)
        sprays.append((a, rng.uniform(0.25, 0.4) * SS, rng.random()))
    sprays.sort(key=lambda t: -t[2])
    for a, L, depth in sprays:
        bend = rng.normal(0, 0.25)
        pts = [(cx + math.cos(a + bend * t) * L * t, cy + math.sin(a + bend * t) * L * t) for t in np.linspace(0, 1, 24)]
        C.twig(pts, SS * 0.003, (80, 62, 46))
        for i in range(1, len(pts)):
            t = i / (len(pts) - 1)
            px, py = pts[i]
            # side branchlets
            if i % 3 == 0 and t < 0.85:
                for sgn in (-1, 1):
                    sa = a + bend * t + sgn * rng.uniform(0.6, 1.0)
                    sl = L * 0.3 * (1 - t)
                    sp = [(px + math.cos(sa) * sl * u, py + math.sin(sa) * sl * u) for u in np.linspace(0, 1, 8)]
                    C.twig(sp, SS * 0.0015, (80, 62, 46))
                    for q in sp[1:]:
                        _needles(C, rng, q, sa, depth, 0.022)
            _needles(C, rng, (px, py), a + bend * t, depth, 0.028 * (1.1 - 0.4 * t))
    return C.finish()


def _needles(C, rng, p, ang, depth, ln):
    for sgn in (-1, 1):
        a = ang + sgn * rng.uniform(1.1, 1.5)
        L = ln * SS * rng.uniform(0.8, 1.2)
        shade = (1 - 0.35 * depth) * rng.uniform(0.82, 1.1)
        c = tuple(int(v) for v in np.clip(BASE * shade * np.array((0.95, 1.0, 1.0)), 0, 255))
        q = (p[0] + math.cos(a) * L, p[1] + math.sin(a) * L)
        C.dc.line([p, q], fill=c + (255,), width=max(3, int(SS * 0.0038)))
        C.dn.line([p, q], fill=_enc_n(_rand_tilt(rng, 0.5)), width=max(3, int(SS * 0.0038)))


def paint_rhodo(seed, n_rosettes=15, radius=0.4):
    """Rhododendron / mountain laurel: glossy elongated leaves in whorled rosettes at twig tips."""
    C = Card(seed); rng = C.rng
    cx = cy = SS / 2
    ros = []
    for k in range(n_rosettes):
        a = rng.uniform(0, 2 * math.pi); r = math.sqrt(rng.random()) * radius * 0.7 * SS
        ros.append((cx + r * math.cos(a), cy + r * math.sin(a), rng.random()))
    for x, y, _ in ros:
        C.twig([(cx, cy), (x, y)], SS * 0.005, (74, 60, 46))
    ros.sort(key=lambda t: -t[2])
    for x, y, depth in ros:
        n = int(rng.integers(7, 11))
        for j in range(n):
            a = 2 * math.pi * j / n + rng.normal(0, 0.25)
            L = rng.uniform(0.09, 0.13) * SS
            poly = xf(leaf_poly('ellip', L, L * 0.42), x, y, a)
            shade = (1 - 0.35 * depth) * rng.uniform(0.8, 1.05)
            C.leaf(poly, shade, (0.95, 1.0, 1.02), _rand_tilt(rng, 0.4), True, (x + math.cos(a) * L, y + math.sin(a) * L), (x, y))
    return C.finish()


def main():
    cards = {
        # broadleaf crowns (card ~1.6 m: leaf fraction 0.055 -> ~9 cm leaves)
        'oak': paint_broadleaf('oak', 11, n_leaves=560, leaf_len=(0.05, 0.075), aspect=(0.45, 0.6)),
        'maple': paint_broadleaf('maple', 12, n_leaves=520, leaf_len=(0.045, 0.07), aspect=(0.55, 0.75)),
        'poplar': paint_broadleaf('tulip', 13, n_leaves=480, leaf_len=(0.05, 0.075), aspect=(0.55, 0.7)),
        'hickory': paint_broadleaf('lance', 14, n_leaves=190, leaf_len=(0.045, 0.06), aspect=(0.35, 0.45), compound=True),
        'dogwood': paint_broadleaf('ovate', 15, n_leaves=460, leaf_len=(0.045, 0.065), aspect=(0.45, 0.55)),
        'brush': paint_broadleaf('ovate', 16, n_leaves=700, leaf_len=(0.03, 0.05), aspect=(0.4, 0.55), clump=10),
        'pine': paint_pine(17),
        'hemlock': paint_hemlock(18),
        'rhodo': paint_rhodo(19),
    }
    for k, (im, nm) in cards.items():
        im.save(f'{OUT}/card_{k}.png', optimize=True)
        nm.save(f'{OUT}/card_{k}_nrm.png', optimize=True)
    prev = Image.new('RGBA', (len(cards) * 300, 300), (70, 100, 140, 255))
    for i, (im, _) in enumerate(cards.values()):
        prev.alpha_composite(im.resize((300, 300)), (i * 300, 0))
    prev.convert('RGB').save(f'{OUT}/_preview.jpg', quality=88)
    # coverage stats (alpha fill of the card) - guide for card density per crown
    for k, (im, _) in cards.items():
        a = np.asarray(im)[..., 3] > 128
        print(f'{k:8s} coverage {a.mean():.2f}')


if __name__ == '__main__':
    main()
