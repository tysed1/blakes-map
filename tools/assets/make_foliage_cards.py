"""Procedurally painted foliage cluster cards (RGBA, 1024^2) for leaf-card trees.

Hand-rolled painter (no external textures needed): twigs carrying leaves of a given
shape (ovate/oak-lobed/maple-palmate/lanceolate), or needle sprays for conifers.
Leaves get per-leaf value variation, a midrib, and baked self-shadowing (core darker,
rim/top lighter) so a crown built from cards reads as a volume at any distance.
Hardwood cards are near-neutral olive; the per-instance tint sets species/autumn hue.
Outputs assets/foliage/card_<name>.png + _nrm.png, and a preview sheet."""
import os, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'assets/foliage')
os.makedirs(OUT, exist_ok=True)
SS = 2048  # supersampled canvas


def leaf_poly(shape, L, Wd, n=28):
    """Leaf outline along +x from base (0,0) to tip (L,0)."""
    pts = []
    for i in range(n + 1):
        t = i / n
        if shape == 'ovate':
            w = Wd * math.sin(math.pi * t) ** 0.85 * (1.1 - 0.35 * t)
        elif shape == 'lance':
            w = Wd * 0.55 * math.sin(math.pi * t) ** 0.9
        elif shape == 'oak':
            w = Wd * (0.55 + 0.45 * abs(math.sin(t * math.pi * 3.5))) * math.sin(math.pi * t) ** 0.7
        elif shape == 'maple':
            w = Wd * (0.35 + 0.65 * abs(math.sin(t * math.pi * 2.5)) ** 1.5) * math.sin(math.pi * t) ** 0.5
        pts.append((L * t, w))
    return pts + [(x, -y) for x, y in reversed(pts)]


def xf(pts, ox, oy, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(ox + x * c - y * s, oy + x * s + y * c) for x, y in pts]


def paint_hardwood(shape, seed, n_twigs=17, leaf_len=(70, 110), density=9, base=(118, 128, 74)):
    rng = np.random.default_rng(seed)
    im = Image.new('RGBA', (SS, SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    C = SS / 2
    items = []
    for k in range(n_twigs):
        ang0 = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(0, 0.12) * SS
        x0, y0 = C + r0 * math.cos(ang0), C + r0 * math.sin(ang0)
        length = rng.uniform(0.2, 0.3) * SS
        ang = ang0 + rng.uniform(-0.4, 0.4)
        bend = rng.uniform(-0.5, 0.5)
        pts = []
        for i in range(12):
            t = i / 11
            a = ang + bend * t
            pts.append((x0 + math.cos(a) * length * t, y0 + math.sin(a) * length * t))
        items.append(('twig', pts, 0))
        for i in range(2, 12):
            for sgn in (-1, 1):
                if rng.random() > density / 10:
                    continue
                px, py = pts[i]
                a = ang + bend * i / 11 + sgn * rng.uniform(0.5, 1.2)
                L = rng.uniform(*leaf_len) * (1.15 - 0.3 * i / 11) * SS / 1024
                items.append(('leaf', xf(leaf_poly(shape, L, L * rng.uniform(0.32, 0.45)), px, py, a), (px, py, a, L)))
        # tip leaf
        px, py = pts[-1]
        L = leaf_len[1] * SS / 1024
        items.append(('leaf', xf(leaf_poly(shape, L, L * 0.4), px, py, ang + bend), (px, py, ang + bend, L)))
    # draw: twigs, then leaves ordered bottom->top so upper leaves overlap (light from above)
    for kind, pts, _ in items:
        if kind == 'twig':
            d.line(pts, fill=(62, 48, 34, 255), width=int(SS * 0.006))
    leaves = [it for it in items if it[0] == 'leaf']
    leaves.sort(key=lambda it: -it[2][1])
    for _, poly, (px, py, a, L) in leaves:
        cx = sum(p[0] for p in poly) / len(poly); cy = sum(p[1] for p in poly) / len(poly)
        rr = math.hypot(cx - C, cy - C) / (SS * 0.45)
        shade = (0.62 + 0.38 * min(1, rr)) * (0.9 + 0.2 * (1 - cy / SS)) * rng.uniform(0.82, 1.12)
        col = tuple(int(np.clip(c * shade, 0, 255)) for c in base)
        d.polygon(poly, fill=col + (255,))
        # midrib + subtle lighter half
        tip = (px + math.cos(a) * L, py + math.sin(a) * L)
        d.line([(px, py), tip], fill=tuple(int(c * 0.78) for c in col) + (255,), width=max(1, int(SS * 0.0018)))
    return finish(im)


def paint_conifer(seed, kind='pine', n_sprays=22):
    rng = np.random.default_rng(seed)
    im = Image.new('RGBA', (SS, SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    C = SS / 2
    base = (52, 84, 46) if kind == 'pine' else (44, 72, 44)
    for k in range(n_sprays):
        ang = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(0, 0.08) * SS
        x0, y0 = C + r0 * math.cos(ang), C + r0 * math.sin(ang)
        L = rng.uniform(0.24, 0.36) * SS
        pts = [(x0 + math.cos(ang) * L * t, y0 + math.sin(ang) * L * t) for t in np.linspace(0, 1, 30)]
        d.line(pts, fill=(70, 52, 36, 255), width=int(SS * 0.005))
        for i, (px, py) in enumerate(pts[2:], 2):
            t = i / 29
            nl = (0.11 if kind == 'pine' else 0.06) * SS * (1.1 - 0.4 * t)
            for sgn in (-1, 1):
                for j in range(5 if kind == 'pine' else 4):
                    a = ang + sgn * rng.uniform(0.35, 1.0 if kind == 'pine' else 1.35)
                    shade = rng.uniform(0.75, 1.25) * (0.7 + 0.3 * t)
                    col = tuple(int(np.clip(c * shade, 0, 255)) for c in base) + (255,)
                    d.line([(px, py), (px + math.cos(a) * nl, py + math.sin(a) * nl)], fill=col, width=max(3, int(SS * (0.0035 if kind == 'pine' else 0.0045))))
    return finish(im)


def finish(im):
    im = im.resize((1024, 1024), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32)
    al = arr[..., 3] > 8
    _, (iy, ix) = ndi.distance_transform_edt(~al, return_indices=True)
    arr[..., :3] = arr[iy, ix, :3]
    return Image.fromarray(arr.astype(np.uint8))


def normal_from(img, strength=3.0):
    a = np.asarray(img).astype(np.float32)
    h = (a[..., :3] @ np.array([0.3, 0.59, 0.11])) / 255 * (a[..., 3] / 255)
    h = ndi.gaussian_filter(h, 1.0)
    gy, gx = np.gradient(h)
    n = np.dstack([-gx * strength, gy * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))


def main():
    cards = {
        'oak': paint_hardwood('oak', 1, leaf_len=(80, 120)),
        'maple': paint_hardwood('maple', 2, leaf_len=(85, 125), density=8),
        'poplar': paint_hardwood('ovate', 3, leaf_len=(70, 100), density=9),
        'birch': paint_hardwood('lance', 4, leaf_len=(55, 80), n_twigs=16, density=9),
        'pine': paint_conifer(5, 'pine'),
        'hemlock': paint_conifer(6, 'hemlock', 26),
    }
    for k, im in cards.items():
        im.save(f'{OUT}/card_{k}.png')
        normal_from(im).save(f'{OUT}/card_{k}_nrm.png')
    prev = Image.new('RGBA', (len(cards) * 340, 340), (70, 100, 140, 255))
    for i, im in enumerate(cards.values()):
        prev.alpha_composite(im.resize((340, 340)), (i * 340, 0))
    prev.convert('RGB').save(f'{OUT}/_preview.jpg')


if __name__ == '__main__':
    main()
