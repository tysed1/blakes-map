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
import base64, gzip, json, os, re, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'dist'
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


# float tables whose precision far exceeds their use: rounded to a grid (so the float bit patterns
# compress) and stored column-major; filter 'c<N>:<f>' tells bin() to transpose back
LOSSY = {
    # x, y, z (source px / m), scale, species, seed  -> 4 cm, 4 cm, 1.6 cm, 0.2 %, exact, 1e-4
    'vegetation_f32.bin': (6, [1 / 64, 1 / 64, 1 / 64, 1 / 512, 0, 1 / 4096]),
}


# u16 rasters: quantized to q units, then a 2D (left + up - upleft) predictor; filter 'p<W>:<f>'
U16_2D = {
    'backdrop_u16.bin': (1900, 16),   # distant ridges, 10 m cells: 0.29 m height steps
}


def round_tree_positions(raw):
    """trees/geo.bin: LOD vertex positions rounded to 2 mm (layout unchanged, float bits compress)."""
    g = json.load(open(os.path.join(D, 'world', 'trees', 'geo.json')))
    b = bytearray(raw)
    for sp in g['species']:
        for l in sp['lods']:
            p = np.frombuffer(b, np.float32, l['vcount'] * 3, l['pos']).copy()
            b[l['pos']:l['pos'] + p.nbytes] = (np.round(p * 512) / 512).astype(np.float32).tobytes()
    return bytes(b)


def lossy(rel, raw):
    """-> (filter prefix or None, payload)"""
    if rel == 'trees/geo.bin':
        return None, round_tree_positions(raw)
    if rel in U16_2D:
        W, q = U16_2D[rel]
        a = (np.round(np.frombuffer(raw, np.uint16) / q) * q).clip(0, 65535).astype(np.int64).reshape(-1, W)
        pred = np.zeros_like(a)
        pred[1:, 1:] = a[1:, :-1] + a[:-1, 1:] - a[:-1, :-1]
        pred[0, 1:] = a[0, :-1]
        pred[1:, 0] = a[:-1, 0]
        return f'p{W}', ((a - pred) % 65536).astype(np.uint16).tobytes()
    if rel not in LOSSY:
        return None, raw
    n, steps = LOSSY[rel]
    a = np.frombuffer(raw, np.float32).reshape(-1, n).copy()
    for c, st in enumerate(steps):
        if st:
            a[:, c] = np.round(a[:, c] / st) * st
    return f'c{n}', a.T.copy().tobytes()


# ---- section-aware packing for binaries described by a JSON index of {f, o, n, t} sections (infra):
# u16 sections are delta coded per component plane (xyz positions de-interleaved), 32-bit sections
# byte-shuffled, then split into low/high byte planes; bin() reverses it from pack.json 'segs'
SZ = {'u1': 1, 'i1': 1, 'u2': 2, 'i2': 2, 'f4': 4, 'u4': 4}


def sections(index_json, fname):
    out = []

    def walk(x, key=None):
        if isinstance(x, dict):
            if {'f', 'o', 'n', 't'} <= set(x):
                if x['f'] == fname:
                    out.append((x['o'], x['n'], x['t'], key))
                return
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
    walk(json.load(open(index_json)))
    return sorted(out)


def segment_pack(raw, secs):
    """-> (segs [[offset, bytes, mode, comps]], payload)"""
    segs, out, pos = [], [], 0
    for o, n, t, key in secs:
        nb = n * SZ[t]
        if o < pos:
            continue   # overlapping / duplicate reference
        if o > pos:
            segs.append([pos, o - pos, 'r', 1]); out.append(raw[pos:o])
        if SZ[t] == 2:
            c = 3 if key == 'pos' and n % 3 == 0 else 1
            a = np.frombuffer(raw, np.uint16, n, o).reshape(-1, c).T
            d = np.ascontiguousarray(np.diff(a.astype(np.int32), axis=1, prepend=0).astype(np.uint16))
            segs.append([o, nb, 'd', c]); out.append(d.view(np.uint8).reshape(-1, 2).T.tobytes())
        elif SZ[t] == 4:
            segs.append([o, nb, 's4', 1]); out.append(np.frombuffer(raw, np.uint8, nb, o).reshape(-1, 4).T.tobytes())
        else:
            segs.append([o, nb, 'r', 1]); out.append(raw[o:o + nb])
        pos = o + nb
    if pos < len(raw):
        segs.append([pos, len(raw) - pos, 'r', 1]); out.append(raw[pos:])
    return segs, b''.join(out)


SEGMENTED = {'infra/infra_roads.bin': 'infra/infra.json', 'infra/infra_struct.bin': 'infra/infra.json', 'infra/infra_water.bin': 'infra/infra.json'}

files, tot, pack = {}, 0, {}
SKIP = {'eco_u8.bin', 'trees.bin', 'road_qa.jpg', 'landuse.png'}  # pipeline-only / legacy / QA-only
# world JSON loaded through src/core/data.ts json() (which reads pack.json): packed like *.bin when large
PACK_JSON = {'roads.json', 'nodes.json', 'bridges.json', 'rail.json', 'water.json', 'landuse.json', 'regions.json', 'settlements.json', 'landmarks.json', 'qa.json'}
# a live KTX2 texture (world/ktx2.json: name -> source image) ships only as KTX2; other .ktx2 files
# (baked but not yet loaded by the viewer) are left out
ktx = json.load(open(os.path.join(D, 'world', 'ktx2.json'))) if os.path.exists(os.path.join(D, 'world', 'ktx2.json')) else {}
SKIP_PATHS = {'world/' + src for src in ktx.values()}

for sub in ('assets', 'world', 'basis'):
    for dirpath, _, names in os.walk(os.path.join(D, sub)):
        for f in sorted(names):
            if f in SKIP or f.endswith('.gz.b64.txt') or (sub == 'world' and f.endswith('.wasm')) or f == 'pack.json':
                continue
            p = os.path.relpath(os.path.join(dirpath, f), D).replace(os.sep, '/')
            if p in SKIP_PATHS or (f.endswith('.ktx2') and p[len('world/'):-len('.ktx2')] not in ktx):
                continue
            if f.endswith('.bin') or f.endswith('.ktx2') or (sub == 'world' and p[len('world/'):] in PACK_JSON and os.path.getsize(os.path.join(D, p)) > 20000):
                src = os.path.join(D, p)
                rel = os.path.relpath(src, os.path.join(D, 'world')).replace(os.sep, '/')
                segs = None
                if rel in SEGMENTED and os.path.exists(os.path.join(D, 'world', SEGMENTED[rel])):
                    segs, data = segment_pack(open(src, 'rb').read(), sections(os.path.join(D, 'world', SEGMENTED[rel]), os.path.basename(rel)))
                    filt, z = 'seg', gzip.compress(data, 9, mtime=0)
                else:
                    pre, data = lossy(rel, open(src, 'rb').read())
                    filt, z = best_filter(data)
                    if pre:
                        filt = f'{pre}:{filt}'

                p += '.wasm'
                open(os.path.join(D, p), 'wb').write(wasm_container(z))
                pack[rel] = {'file': rel + '.wasm', 'filter': filt, 'size': os.path.getsize(src)}
                if segs:
                    pack[rel]['segs'] = segs
            files[p] = os.path.join(D, p)
            tot += os.path.getsize(files[p])
json.dump(pack, open(os.path.join(D, 'world', 'pack.json'), 'w'), indent=1)
files['world/pack.json'] = os.path.join(D, 'world', 'pack.json')
json.dump(files, open(os.path.join(D, 'files.json'), 'w'), indent=1)
print(len(files), 'files', round(tot / 1e6, 1), 'MB; largest',
      max(((os.path.getsize(v), k) for k, v in files.items()))[1])
