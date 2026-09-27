"""Contact sheets for QA shots.
  python3 tools/qa/contact_sheet.py OUT.jpg DIR                 # grid of DIR/*.png
  python3 tools/qa/contact_sheet.py OUT.jpg BEFORE_DIR AFTER_DIR # before | after pairs per shot
"""
import os, sys
from PIL import Image, ImageDraw
out, dirs = sys.argv[1], sys.argv[2:]
names = sorted({os.path.splitext(f)[0] for f in os.listdir(dirs[-1]) if f.endswith('.png')})
TW, TH = 480, 270
cols = 1 if len(dirs) == 2 else 4
cells = [(n, d) for n in names for d in dirs] if len(dirs) == 2 else [(n, dirs[0]) for n in names]
ncol = 2 if len(dirs) == 2 else cols
rows = (len(cells) + ncol - 1) // ncol
sheet = Image.new('RGB', (ncol * TW, rows * (TH + 18)), (18, 20, 22))
dr = ImageDraw.Draw(sheet)
for i, (n, d) in enumerate(cells):
    p = os.path.join(d, n + '.png')
    x, y = (i % ncol) * TW, (i // ncol) * (TH + 18)
    if os.path.exists(p):
        sheet.paste(Image.open(p).convert('RGB').resize((TW, TH)), (x, y + 18))
    dr.text((x + 4, y + 3), f'{n}  [{os.path.basename(os.path.normpath(d))}]', fill=(230, 220, 200))
sheet.save(out, quality=85)
print(out, sheet.size)
