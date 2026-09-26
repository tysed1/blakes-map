"""Enhanced reading crops: base map + magenta tint where the road-likeness field is high,
with a labelled 10 px grid. usage: enhanced.py x0 y0 x1 y1 scale out.png"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from PIL import Image, ImageDraw
from tools.lib.common import load_rgb
from tools.lib.features import road_prob


def enhanced(x0, y0, x1, y1, s, out, extra_lines=None):
    a = load_rgb()[y0:y1, x0:x1].astype(np.float32)
    p = road_prob()[y0:y1, x0:x1]
    t = np.clip((p - 0.12) / 0.3, 0, 1)[..., None] * 0.55
    a = a * (1 - t) + np.array([255, 0, 255], np.float32) * t
    im = Image.fromarray(a.astype(np.uint8)).resize(((x1 - x0) * s, (y1 - y0) * s), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for gx in range((x0 // 10) * 10, x1, 10):
        if gx < x0: continue
        X = (gx - x0) * s
        d.line([(X, 0), (X, im.size[1])], fill=(255, 0, 0) if gx % 50 == 0 else (90, 90, 90))
        d.text((X + 2, 2), str(gx), fill=(255, 255, 0))
    for gy in range((y0 // 10) * 10, y1, 10):
        if gy < y0: continue
        Y = (gy - y0) * s
        d.line([(0, Y), (im.size[0], Y)], fill=(255, 0, 0) if gy % 50 == 0 else (90, 90, 90))
        d.text((2, Y + 2), str(gy), fill=(255, 255, 0))
    if extra_lines:
        for pts, col in extra_lines:
            d.line([((x - x0) * s, (y - y0) * s) for x, y in pts], fill=col, width=2)
    im.save(out)


if __name__ == '__main__':
    a = sys.argv[1:]
    enhanced(*map(int, a[:5]), a[5])
