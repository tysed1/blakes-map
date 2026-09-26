"""Side-by-side validation vs the art reference + simple look metrics.
usage: python3 tools/qa/compare_ref.py render.jpg out.jpg"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ref = Image.open(os.path.join(ROOT, 'graphics ref.png')).convert('RGB')
ren = Image.open(sys.argv[1]).convert('RGB')
W = 800
r1 = ref.resize((W, int(ref.size[1] * W / ref.size[0])))
r2 = ren.resize((W, int(ren.size[1] * W / ren.size[0])))
H = max(r1.size[1], r2.size[1])
c = Image.new('RGB', (2 * W, H + 60), (20, 20, 20))
c.paste(r1, (0, 0)); c.paste(r2, (W, 0))
def stats(im):
    a = np.asarray(im).astype(np.float32) / 255
    hsv = np.asarray(im.convert('HSV')).astype(np.float32) / 255
    h = a.shape[0]
    top, bot = slice(0, h // 5), slice(h // 3, h)
    return dict(lum=a.mean(), sat=hsv[..., 1].mean(), sky_lum=a[top].mean(), land_lum=a[bot].mean(), land_sat=hsv[bot, :, 1].mean(),
                warm=(a[bot, :, 0] - a[bot, :, 2]).mean(), green=(a[bot, :, 1] - (a[bot, :, 0] + a[bot, :, 2]) / 2).mean(),
                contrast=a[bot].std())
s1, s2 = stats(r1), stats(r2)
d = ImageDraw.Draw(c)
d.text((10, H + 8), 'REFERENCE   ' + '  '.join(f'{k}={v:.3f}' for k, v in s1.items()), fill=(230, 230, 230))
d.text((10, H + 30), 'RENDER      ' + '  '.join(f'{k}={v:.3f}' for k, v in s2.items()), fill=(255, 210, 120))
c.save(sys.argv[2], quality=90)
for k in s1:
    print(f'{k:9s} ref {s1[k]:.3f}  render {s2[k]:.3f}  diff {s2[k] - s1[k]:+.3f}')
