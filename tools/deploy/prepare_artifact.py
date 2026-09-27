"""Turn the Vite build (dist/) into an Artifact page + files map.

The Artifact host wraps the page in its own <html>/<head>/<body>, so the page file keeps only
the <title>, the stylesheet/script tags and the body content. Supporting files keep their
relative paths (./assets/*, ./world/*). eco_u8.bin is Blender/QA-only and is left out.
The host serves only web media types, so every *.bin / *.ktx2 ships as <name>.wasm: a valid WebAssembly
module whose one custom section holds the gzip of the file after the best of a few lossless filters
(byte-plane shuffle, 16/32-bit delta + shuffle); world/pack.json maps names -> file + filter and
src/core/data.ts bin() reverses it when built with VITE_PACKED_BIN=1 (no base64: -25 % transfer).
usage: VITE_PACKED_BIN=1 npm run build && python3 tools/deploy/prepare_artifact.py  -> dist/artifact.html, dist/files.json
"""
import base64, gzip, json, os, re
import numpy as np

D = 'dist'
html = open(os.path.join(D, 'index.html')).read()
head = re.search(r'<head>(.*?)</head>', html, re.S).group(1)
body = re.search(r'<body>(.*?)</body>', html, re.S).group(1)
keep = [l for l in head.splitlines() if re.search(r'<title>|<script|<link rel="stylesheet"', l)]
page = '\n'.join(k.strip() for k in keep) + '\n<style>:root{color-scheme:dark}html,body{height:100%}</style>\n' + body.strip() + '\n'
open(os.path.join(D, 'artifact.html'), 'w').write(page)


def leb(n):
    out = bytearray()
    while True:
        b = n & 0x7f
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def wasm_container(payload, name=b'bm'):
    body = leb(len(name)) + name + payload
    return b'\0asm\x01\0\0\0' + b'\x00' + leb(len(body)) + body


def best_filter(raw):
    a = np.frombuffer(raw, np.uint8)
    cands = {'raw': raw}
    if len(raw) % 2 == 0:
        cands['sh2'] = a.reshape(-1, 2).T.tobytes()
        u = np.frombuffer(raw, np.uint16)
        cands['d16sh2'] = np.diff(u, prepend=np.uint16(0)).astype(np.uint16).view(np.uint8).reshape(-1, 2).T.tobytes()
    if len(raw) % 4 == 0:
        cands['sh4'] = a.reshape(-1, 4).T.tobytes()
        u = np.frombuffer(raw, np.uint32)
        cands['d32sh4'] = np.diff(u, prepend=np.uint32(0)).astype(np.uint32).view(np.uint8).reshape(-1, 4).T.tobytes()
    best = None
    for k, v in cands.items():
        z = gzip.compress(v, 9, mtime=0)
        if best is None or len(z) < len(best[1]):
            best = (k, z)
    return best


files, tot, pack = {}, 0, {}
SKIP = {'eco_u8.bin', 'trees.bin'}  # pipeline-only / legacy
for sub in ('assets', 'world'):
    for dirpath, _, names in os.walk(os.path.join(D, sub)):
        for f in sorted(names):
            if f in SKIP or f.endswith('.gz.b64.txt') or f.endswith('.bin.wasm') or f == 'pack.json':
                continue
            p = os.path.relpath(os.path.join(dirpath, f), D).replace(os.sep, '/')
            if f.endswith('.bin') or f.endswith('.ktx2'):
                src = os.path.join(D, p)
                filt, z = best_filter(open(src, 'rb').read())
                rel = os.path.relpath(src, os.path.join(D, 'world')).replace(os.sep, '/')
                p += '.wasm'
                open(os.path.join(D, p), 'wb').write(wasm_container(z))
                pack[rel] = {'file': rel + '.wasm', 'filter': filt, 'size': os.path.getsize(src)}
            files[p] = os.path.join(D, p)
            tot += os.path.getsize(files[p])
json.dump(pack, open(os.path.join(D, 'world', 'pack.json'), 'w'), indent=1)
files['world/pack.json'] = os.path.join(D, 'world', 'pack.json')
json.dump(files, open(os.path.join(D, 'files.json'), 'w'), indent=1)
print(len(files), 'files', round(tot / 1e6, 1), 'MB; largest',
      max(((os.path.getsize(v), k) for k, v in files.items()))[1])
