"""Terrain close-up detail for the web viewer (public/world/terrain/).

  mask.png            RGBA 2000x667: R rock exposure (+talus), G canopy, B field type * 40, A moisture
  detail_<k>.webp     1024^2 tiling Poly Haven texture: RGB albedo, A displacement (bump)
                      grass (leafy_grass), forest (leaves_forest_ground), rock (lichen_rock), soil (farm_soil)
The baked albedo keeps the large-scale colour; the detail maps only modulate it up close.
"""
import os
import numpy as np
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'public/world/terrain'); os.makedirs(OUT, exist_ok=True)
PH = os.path.join(ROOT, 'assets/external/polyhaven/_cache1k')
eco = np.fromfile(os.path.join(ROOT, 'public/world/eco_u8.bin'), np.uint8).reshape(667, 2000, -1)
rock = np.maximum(eco[..., 10], (eco[..., 7].astype(np.int32) * 0.6).astype(np.uint8))
m = np.dstack([rock, eco[..., 2], np.minimum(eco[..., 3].astype(np.int32) * 40, 255).astype(np.uint8), eco[..., 0]])
Image.fromarray(m, 'RGBA').save(os.path.join(OUT, 'mask.png'), optimize=True)


def find(prefix):
    for f in sorted(os.listdir(PH)):
        if f.startswith(prefix):
            return os.path.join(PH, f)
    raise FileNotFoundError(prefix)


for k, name in (('grass', 'leafy_grass'), ('forest', 'leaves_forest_ground'), ('rock', 'lichen_rock'), ('soil', 'farm_soil')):
    d = Image.open(find(name + '_diff')).convert('RGB').resize((1024, 1024), Image.LANCZOS)
    h = Image.open(find(name + '_disp')).convert('L').resize((1024, 1024), Image.LANCZOS)
    d.putalpha(h)
    d.save(os.path.join(OUT, f'detail_{k}.webp'), quality=88, method=6)
print('ok', os.listdir(OUT))
