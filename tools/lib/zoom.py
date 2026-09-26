"""Debug helper: render a zoomed crop of the base map with a labelled pixel grid.

Usage: python3 tools/lib/zoom.py x0 y0 x1 y1 scale out.png [overlay.png]
Grid: thin lines every 5 source px, labelled lines every 10/50 px.
"""
import sys
from PIL import Image, ImageDraw

ROOT = __file__.rsplit('/tools/', 1)[0]


def zoom(x0, y0, x1, y1, s, out, src=None, grid=True, overlay=None, alpha=0.55):
    im = Image.open(src or f'{ROOT}/assets/maps/processed/base_map.png').convert('RGB')
    c = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * s, (y1 - y0) * s), Image.NEAREST if s >= 6 else Image.LANCZOS)
    if overlay is not None:
        ov = Image.open(overlay).convert('RGBA').crop((x0, y0, x1, y1)).resize(c.size, Image.NEAREST)
        c = c.convert('RGBA')
        c.alpha_composite(ov)
        c = c.convert('RGB')
    if grid:
        d = ImageDraw.Draw(c)
        step = 5 if s >= 6 else 10
        for gx in range(((x0 + step - 1) // step) * step, x1, step):
            X = (gx - x0) * s
            major = gx % 50 == 0
            d.line([(X, 0), (X, c.size[1])], fill=(255, 0, 0) if major else ((255, 255, 255) if gx % 10 == 0 else (120, 120, 120)), width=1)
            if gx % 10 == 0:
                d.text((X + 2, 2), str(gx), fill=(255, 255, 0))
        for gy in range(((y0 + step - 1) // step) * step, y1, step):
            Y = (gy - y0) * s
            major = gy % 50 == 0
            d.line([(0, Y), (c.size[0], Y)], fill=(255, 0, 0) if major else ((255, 255, 255) if gy % 10 == 0 else (120, 120, 120)), width=1)
            if gy % 10 == 0:
                d.text((2, Y + 2), str(gy), fill=(255, 255, 0))
    c.save(out)


if __name__ == '__main__':
    a = sys.argv[1:]
    zoom(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), a[5], overlay=a[6] if len(a) > 6 else None)
