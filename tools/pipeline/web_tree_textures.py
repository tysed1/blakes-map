"""Downsized leaf-card / bark textures for the web trees (public/world/trees/)."""
import os
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'public/world/trees'); os.makedirs(OUT, exist_ok=True)
for c in ('oak', 'maple', 'poplar', 'hickory', 'dogwood', 'pine', 'hemlock', 'rhodo', 'brush'):
    im = Image.open(os.path.join(ROOT, f'assets/foliage/card_{c}.png')).convert('RGBA').resize((512, 512), Image.LANCZOS)
    # bleed colour into transparent texels so mipmaps don't fringe dark
    rgb = im.convert('RGB'); a = im.getchannel('A')
    blur = rgb.resize((64, 64), Image.BILINEAR).resize((512, 512), Image.BILINEAR)
    rgb = Image.composite(rgb, blur, a.point(lambda v: 255 if v > 8 else 0))
    rgb.putalpha(a)
    rgb.save(os.path.join(OUT, f'card_{c}.png'), optimize=True)
    Image.open(os.path.join(ROOT, f'assets/foliage/card_{c}_nrm.png')).convert('RGB').resize((512, 512), Image.LANCZOS).save(os.path.join(OUT, f'card_{c}_n.jpg'), quality=90)
ph = os.path.join(ROOT, 'assets/external/polyhaven/_cache1k')
for n, f in (('brown', 'bark_brown_02_diff_2k.jpg'), ('pine', 'pine_bark_diff_2k.jpg')):
    Image.open(os.path.join(ph, f)).convert('RGB').resize((512, 512), Image.LANCZOS).save(os.path.join(OUT, f'bark_{n}.jpg'), quality=86)
print('ok')
