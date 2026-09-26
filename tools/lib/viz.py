"""Debug renderers: draw pixel-space geometry over zoomed crops of the base map."""
from PIL import Image, ImageDraw
import numpy as np
from .common import SRC_PNG


def render(box, scale, lines=(), points=(), polys=(), out=None, grid=10, dim=1.0, labels=True, width=2):
    """lines: list of (pts, color[, label]); points: list of (x,y,color[,label]); polys: (pts, outline, fill)."""
    x0, y0, x1, y1 = box
    im = Image.open(SRC_PNG).convert('RGB').crop(box)
    im = im.resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.LANCZOS if scale < 6 else Image.NEAREST)
    if dim < 1:
        im = Image.fromarray((np.asarray(im) * dim).astype(np.uint8))
    im = im.convert('RGBA')
    ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    T = lambda p: ((p[0] - x0) * scale, (p[1] - y0) * scale)
    if grid:
        for gx in range(((x0 + grid - 1) // grid) * grid, x1, grid):
            X = (gx - x0) * scale
            d.line([(X, 0), (X, im.size[1])], fill=(255, 0, 0, 150) if gx % 50 == 0 else (255, 255, 255, 50))
            if gx % (grid * 2 if scale < 4 else grid) == 0:
                d.text((X + 2, 2), str(gx), fill=(255, 255, 0, 255))
        for gy in range(((y0 + grid - 1) // grid) * grid, y1, grid):
            Y = (gy - y0) * scale
            d.line([(0, Y), (im.size[0], Y)], fill=(255, 0, 0, 150) if gy % 50 == 0 else (255, 255, 255, 50))
            if gy % (grid * 2 if scale < 4 else grid) == 0:
                d.text((2, Y + 2), str(gy), fill=(255, 255, 0, 255))
    for pl in polys:
        pts, outline, fill = pl[0], pl[1], pl[2]
        if len(pts) >= 3:
            d.polygon([T(p) for p in pts], outline=outline, fill=fill)
    for ln in lines:
        pts, col = ln[0], ln[1]
        w = ln[3] if len(ln) > 3 else width
        if len(pts) >= 2:
            d.line([T(p) for p in pts], fill=col, width=w)
        if labels and len(ln) > 2 and ln[2]:
            m = pts[len(pts) // 2]
            X, Y = T(m)
            d.text((X + 3, Y - 10), str(ln[2]), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for pt in points:
        X, Y = T(pt[:2])
        r = 3
        d.ellipse([X - r, Y - r, X + r, Y + r], outline=pt[2], fill=pt[2])
        if labels and len(pt) > 3 and pt[3]:
            d.text((X + 4, Y + 2), str(pt[3]), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    im.alpha_composite(ov)
    im = im.convert('RGB')
    if out:
        im.save(out)
    return im
