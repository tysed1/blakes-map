"""Turn the Vite build (dist/) into an Artifact page + files map.

The Artifact host wraps the page in its own <html>/<head>/<body>, so the page file keeps only
the <title>, the stylesheet/script tags and the body content. Supporting files keep their
relative paths (./assets/*, ./world/*). eco_u8.bin is Blender/QA-only and is left out.
The host serves only web media types, so *.bin rasters ship as <name>.gz.b64.txt (gzip + base64),
which the app decodes when built with VITE_PACKED_BIN=1.
usage: VITE_PACKED_BIN=1 npm run build && python3 tools/deploy/prepare_artifact.py  -> dist/artifact.html, dist/files.json
"""
import base64, gzip, json, os, re

D = 'dist'
html = open(os.path.join(D, 'index.html')).read()
head = re.search(r'<head>(.*?)</head>', html, re.S).group(1)
body = re.search(r'<body>(.*?)</body>', html, re.S).group(1)
keep = [l for l in head.splitlines() if re.search(r'<title>|<script|<link rel="stylesheet"', l)]
page = '\n'.join(k.strip() for k in keep) + '\n<style>:root{color-scheme:dark}html,body{height:100%}</style>\n' + body.strip() + '\n'
open(os.path.join(D, 'artifact.html'), 'w').write(page)
files, tot = {}, 0
for sub in ('assets', 'world'):
    for f in sorted(os.listdir(os.path.join(D, sub))):
        if f == 'eco_u8.bin':
            continue
        p = f'{sub}/{f}'
        if f.endswith('.bin'):
            src = os.path.join(D, p)
            p += '.gz.b64.txt'
            open(os.path.join(D, p), 'wb').write(base64.b64encode(gzip.compress(open(src, 'rb').read(), 9)))
        elif f.endswith('.gz.b64.txt'):
            continue
        files[p] = os.path.join(D, p)
        tot += os.path.getsize(files[p])
json.dump(files, open(os.path.join(D, 'files.json'), 'w'), indent=1)
print(len(files), 'files', round(tot / 1e6, 1), 'MB; largest',
      max(((os.path.getsize(v), k) for k, v in files.items()))[1])
