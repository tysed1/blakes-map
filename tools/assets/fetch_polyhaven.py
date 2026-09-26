"""Fetch CC0 Poly Haven assets into assets/external/polyhaven/<id>/ (gitignored; re-fetchable).
usage: python3 tools/assets/fetch_polyhaven.py [--res 1k] id [id ...]
Models/textures: downloads the .blend (+ its included texture files); HDRIs: .hdr.
Writes assets/external/polyhaven/manifest.json (id, type, license CC0, author, source url)."""
import json, os, sys, urllib.request, concurrent.futures as cf
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'assets/external/polyhaven')
API = 'https://api.polyhaven.com'
UA = {'User-Agent': 'blakes-map-asset-fetch/1.0 (world planning tools)'}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, 'wb') as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    return path


def fetch(aid, res):
    info = json.loads(get(f'{API}/info/{aid}'))
    files = json.loads(get(f'{API}/files/{aid}'))
    d = os.path.join(OUT, aid)
    jobs = []
    if info['type'] == 0:  # hdri
        f = files['hdri'][res]['hdr']
        jobs.append((f['url'], os.path.join(d, f'{aid}_{res}.hdr')))
    else:
        b = files['blend'][res]['blend']
        jobs.append((b['url'], os.path.join(d, f'{aid}_{res}.blend')))
        for rel, f in b.get('include', {}).items():
            jobs.append((f['url'], os.path.join(d, rel)))
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: dl(*j), jobs))
    return {'id': aid, 'type': ['hdri', 'texture', 'model'][info['type']], 'name': info['name'], 'license': 'CC0',
            'authors': list(info.get('authors', {}).keys()), 'source': f'https://polyhaven.com/a/{aid}', 'res': res}


if __name__ == '__main__':
    a = sys.argv[1:]
    res = '1k'
    if '--res' in a:
        res = a[a.index('--res') + 1]; a = [x for x in a if x not in ('--res', res)]
    mp = os.path.join(OUT, 'manifest.json')
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    for aid in a:
        try:
            man[aid] = fetch(aid, res)
            print('ok', aid)
        except Exception as e:
            print('FAILED', aid, e)
    os.makedirs(OUT, exist_ok=True)
    json.dump(man, open(mp, 'w'), indent=1)
