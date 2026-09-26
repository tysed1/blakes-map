"""Build the Blender world scene from the canonical world data.

    blender -b --python tools/blender/build_world.py -- [--no-trees] [--out exports/blender/world.blend] [--glb]

Everything is placed through the canonical transform (data/world/world.json):
    Blender (Z-up):  bx = (px - 1000) * 2.5,  by = -(py - 333.5) * 2.5,  bz = elevation
(the glTF exporter converts to Y-up, which matches the web viewer's X/Y/Z exactly).

Object naming = world IDs, so primitives can be swapped for real assets one by one:
    TERRAIN_Cxx_yy, ROAD <road id>, BRIDGE <bridge id>, RAIL <rail id>, WATER_Cxx_yy,
    VEG_points (Geometry Nodes instancer), BACKDROP, CAM_*.
"""
import bpy, bmesh, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_trees as LT
import lib_materials as LM
import lib_groundcover as GC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OPT = lambda k: k in ARGS
def ARG(k, d=None):
    return ARGS[ARGS.index(k) + 1] if k in ARGS else d

W, H, MPP, OX, OY = 2000, 667, 2.5, 1000.0, 333.5
CHUNK = 100


def P(*p):
    return os.path.join(ROOT, *p)


def load(p):
    with open(P(p)) as f:
        return json.load(f)


def px2b(x, y, z):
    return ((np.asarray(x) - OX) * MPP, -(np.asarray(y) - OY) * MPP, np.asarray(z))


class HF:
    def __init__(self, a):
        self.a = a

    def at(self, x, y):
        a = self.a
        x = np.clip(np.asarray(x, float) - 0.5, 0, a.shape[1] - 1.001)
        y = np.clip(np.asarray(y, float) - 0.5, 0, a.shape[0] - 1.001)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        fx = x - x0; fy = y - y0
        return a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy


T = HF(np.fromfile(P('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W))
WLraw = np.fromfile(P('data/terrain/water_level_f32.bin'), np.float32).reshape(H, W)


def collection(name, parent=None):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in (parent or bpy.context.scene.collection).children:
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def mesh_obj(name, verts, faces, coll, uvs=None, mat=None, smooth=False, attrs=None):
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, np.float32).reshape(-1, 3)
    faces = np.asarray(faces, np.int32)
    me.vertices.add(len(verts))
    me.vertices.foreach_set('co', verts.ravel())
    nf, k = faces.shape
    me.loops.add(nf * k)
    me.loops.foreach_set('vertex_index', faces.ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set('loop_start', np.arange(0, nf * k, k, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(nf, k, np.int32))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    if uvs is not None:
        uv = me.uv_layers.new(name='UVMap')
        uv.data.foreach_set('uv', np.asarray(uvs, np.float32)[faces.ravel()].ravel())
    if smooth:
        me.polygons.foreach_set('use_smooth', np.ones(nf, bool))
    if attrs:
        for an, (dom, typ, data) in attrs.items():
            a = me.attributes.new(an, typ, dom)
            a.data.foreach_set('value' if typ in ('FLOAT', 'INT') else 'color', np.asarray(data, np.float32 if typ != 'INT' else np.int32).ravel())
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


# ------------------------------------------------------------------ terrain
def _blur(a, r):
    # separable box blur x3 ~ gaussian (numpy only; Blender python has no scipy)
    for _ in range(3):
        k = 2 * r + 1
        c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode='edge'), 1); a = (c[:, k:] - c[:, :-k]) / k
        c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode='edge'), 0); a = (c[k:] - c[:-k]) / k
    return a


def landuse_weights():
    lu = np.fromfile(P('public/world/landuse_u8.bin'), np.uint8).reshape(H, W)
    rm = np.zeros((H, W), np.float32)
    for f in load('data/roads/roads.geojson')['features']:
        c = np.asarray(f['geometry']['coordinates'])[:, :2]
        for a, b in zip(c[:-1], c[1:]):
            n = int(max(2, np.hypot(*(b - a)) * 2))
            xs = np.clip(np.linspace(a[0], b[0], n).astype(int), 0, W - 1); ys = np.clip(np.linspace(a[1], b[1], n).astype(int), 0, H - 1)
            rm[ys, xs] = 1
    wet = (lu == 1).astype(np.float32)
    layers = [np.isin(lu, [4]), np.isin(lu, [5]), np.isin(lu, [7, 8, 9, 10, 11]), np.isin(lu, [6]), np.isin(lu, [2, 3])]
    out = [_blur(l.astype(np.float32), 1) for l in layers]
    out.append(np.clip(_blur(wet, 2) * 2.2, 0, 1) * (1 - wet))
    out.append(np.clip(_blur(rm, 1) * 1.6, 0, 1))
    return np.stack(out).astype(np.float32)


LUW = None


def build_terrain(coll, mat):
    global LUW
    LUW = landuse_weights()
    objs = []
    for cy in range(0, math.ceil(H / CHUNK)):
        for cx in range(0, math.ceil(W / CHUNK)):
            x0, y0 = cx * CHUNK, cy * CHUNK
            x1, y1 = min(W, x0 + CHUNK), min(H, y0 + CHUNK)
            xs = np.arange(x0, x1 + 1, dtype=np.float64)
            ys = np.arange(y0, y1 + 1, dtype=np.float64)
            gx, gy = np.meshgrid(xs, ys)
            z = T.at(gx, gy)
            bx, by, bz = px2b(gx, gy, z)
            verts = np.stack([bx, by, bz], -1).reshape(-1, 3)
            nx, ny = len(xs), len(ys)
            i = np.arange(nx - 1); j = np.arange(ny - 1)
            ii, jj = np.meshgrid(i, j)
            a = (jj * nx + ii).ravel()
            faces = np.stack([a, a + nx, a + nx + 1, a + 1], 1)
            uvs = np.stack([gx.ravel() / W, 1 - gy.ravel() / H], 1)
            xi = np.clip(gx.ravel().astype(int), 0, W - 1); yi = np.clip(gy.ravel().astype(int), 0, H - 1)
            A = LUW[:, yi, xi].T  # field, meadow, developed, rock, forest, bank, shoulder
            lu_a = np.c_[A[:, 0], A[:, 1], A[:, 2], A[:, 3]]
            lu_b = np.c_[A[:, 4], A[:, 5], A[:, 6], np.ones(len(A))]
            ob = mesh_obj(f'TERRAIN_C{cx:02d}_{cy:02d}', verts, faces, coll, uvs=uvs, mat=mat, smooth=True,
                          attrs={'lu_a': ('POINT', 'FLOAT_COLOR', lu_a), 'lu_b': ('POINT', 'FLOAT_COLOR', lu_b)})
            objs.append(ob)
    return objs


def build_water(coll, mat):
    # raw buffers from the web export (Blender's Python has no PIL/scipy)
    lu = np.fromfile(P('public/world/landuse_u8.bin'), np.uint8).reshape(H, W)
    man = load('public/world/manifest.json')['terrain']
    wl = man['min_m'] + np.fromfile(P('public/world/water_u16.bin'), '<u2').reshape(H, W).astype(np.float32) * (man['max_m'] - man['min_m']) / 65535
    WL = HF(wl)  # nearest-filled water surface
    wet = lu == 1
    wet = wet | np.roll(wet, 1, 0) | np.roll(wet, -1, 0) | np.roll(wet, 1, 1) | np.roll(wet, -1, 1)
    for cy in range(0, math.ceil(H / CHUNK)):
        for cx in range(0, math.ceil(W / CHUNK)):
            sub = wet[cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK]
            if not sub.any():
                continue
            ys, xs = np.nonzero(sub)
            xs = xs + cx * CHUNK; ys = ys + cy * CHUNK
            vid = {}
            verts, faces = [], []
            def v(x, y):
                k = (x, y)
                if k not in vid:
                    vid[k] = len(verts)
                    verts.append((x, y))
                return vid[k]
            for x, y in zip(xs, ys):
                faces.append((v(x, y), v(x, y + 1), v(x + 1, y + 1), v(x + 1, y)))
            V = np.asarray(verts, float)
            bx, by, bz = px2b(V[:, 0], V[:, 1], WL.at(V[:, 0], V[:, 1]) + 0.05)
            mesh_obj(f'WATER_C{cx:02d}_{cy:02d}', np.stack([bx, by, bz], 1), faces, coll, mat=mat, smooth=True)


# ------------------------------------------------------------------ roads
SURF = {'freeway': 'hwy', 'highway': 'hwy', 'ramp': 'hwy', 'arterial': 'city', 'main_street': 'city', 'collector': 'city', 'urban_street': 'city',
        'residential': 'local', 'rural': 'chip', 'gravel': 'gravel', 'driveway': 'gravel', 'dirt': 'dirt'}
LIFT = 0.12


def densify(P3, spacing=2.0):
    out = []
    for a, b in zip(P3[:-1], P3[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(L / spacing)))
        for k in range(n):
            t = k / n
            out.append(a + (b - a) * t)
    out.append(P3[-1])
    return np.asarray(out)


def ribbon(Pb, o0, o1, lift, trim0=0.0, trim1=0.0, ground=True):
    """Pb: Nx3 blender coords along the road. Returns verts, quads."""
    P3 = Pb.copy()
    if ground:
        gx = P3[:, 0] / MPP + OX; gy = -P3[:, 1] / MPP + OY
        P3[:, 2] = np.maximum(P3[:, 2], T.at(gx, gy) + 0.05)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P3[:, :2], axis=0).T))]
    L = s[-1]
    if L - trim0 - trim1 < 0.5:
        return None, None
    keep = (s >= trim0) & (s <= L - trim1)
    t = np.gradient(P3[:, :2], axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nrm = np.stack([t[:, 1], -t[:, 0]], 1)  # right-hand normal (x east, y north)
    P3, nrm = P3[keep], nrm[keep]
    if len(P3) < 2:
        return None, None
    left = np.c_[P3[:, :2] + nrm * o0, P3[:, 2] + lift]
    right = np.c_[P3[:, :2] + nrm * o1, P3[:, 2] + lift]
    V = np.empty((len(P3) * 2, 3)); V[0::2] = left; V[1::2] = right
    k = np.arange(len(P3) - 1) * 2
    F = np.stack([k, k + 1, k + 3, k + 2], 1)  # CCW seen from above (+z)
    return V, F


def road_pts(r):
    c = np.asarray(r['geometry']['coordinates'], float)
    bx, by, bz = px2b(c[:, 0], c[:, 1], c[:, 2])
    return densify(np.stack([bx, by, bz], 1))


def build_roads(coll, mats, types):
    feats = load('data/roads/roads.geojson')['features']
    nodes = {f['properties']['id']: f for f in load('data/roads/road_nodes.geojson')['features']}
    nodeR = {}
    for f in feats:
        p = f['properties']
        if p.get('virtual'):
            continue
        for n in (p['from'], p['to']):
            nodeR[n] = max(nodeR.get(n, 0), p['width_m'] / 2)
    mark_coll = collection('ROAD_MARKINGS', coll)
    marks = {'yellow': ([], []), 'white': ([], [])}
    for f in feats:
        p = f['properties']
        if p.get('virtual'):
            continue
        t = p['type']
        Pb = road_pts(f)
        hw = p['width_m'] / 2
        lift = LIFT + types[t]['z'] * 0.004
        V, F = ribbon(Pb, -hw, hw, lift)
        if V is None:
            continue
        uv = np.zeros((len(V), 2)); s = np.r_[0, np.cumsum(np.hypot(*np.diff(V[0::2, :2], axis=0).T))]
        uv[0::2, 0] = 0; uv[1::2, 0] = 1; uv[0::2, 1] = s / 8; uv[1::2, 1] = s / 8
        ob = mesh_obj(f"ROAD {p['id']}", V, F, coll, uvs=uv, mat=mats[SURF[t]])
        for k in ('type', 'name', 'route', 'material', 'surface', 'lanes', 'width_m', 'speed_mph'):
            if p.get(k) is not None:
                ob[k] = p[k]
        ob['world_id'] = p['id']
        # markings
        jn = lambda n: nodes.get(n, {}).get('properties', {}).get('degree', 1) >= 3
        tr0 = nodeR.get(p['from'], 0) + 1 if jn(p['from']) else 0
        tr1 = nodeR.get(p['to'], 0) + 1 if jn(p['to']) else 0
        ml = lift + 0.02
        specs = []
        if t == 'freeway':
            specs = [('yellow', -0.6, 0.08), ('yellow', 0.6, 0.08), ('white', -hw + 1.2, 0.1), ('white', hw - 1.2, 0.1)]
        elif t in ('highway', 'arterial', 'main_street'):
            specs = [('yellow', -0.12, 0.06), ('yellow', 0.12, 0.06)] + ([('white', -hw + 0.9, 0.08), ('white', hw - 0.9, 0.08)] if t == 'highway' else [])
        elif t == 'ramp':
            specs = [('white', -hw + 0.5, 0.08), ('white', hw - 0.5, 0.08)]
        elif t in ('collector', 'urban_street', 'rural'):
            specs = [('yellow', 0.0, 0.06)]
        for col, o, w in specs:
            V2, F2 = ribbon(Pb, o - w, o + w, ml, tr0, tr1)
            if V2 is None:
                continue
            vs, fs = marks[col]
            off = sum(len(x) for x in vs)
            vs.append(V2); fs.append(F2 + off)
    for col, (vs, fs) in marks.items():
        if vs:
            mesh_obj(f'MARKINGS_{col}', np.vstack(vs), np.vstack(fs), mark_coll, mat=mats[col])
    # junction pads
    pad_coll = collection('JUNCTIONS', coll)
    rank = lambda t: types[t]['z']
    legs = {}
    for f in feats:
        p = f['properties']
        if p.get('virtual'):
            continue
        for n, zz in ((p['from'], f['geometry']['coordinates'][0][2]), (p['to'], f['geometry']['coordinates'][-1][2])):
            legs.setdefault(n, []).append((p['type'], zz))
    by_surf = {}
    for nid, n in nodes.items():
        pr = n['properties']
        if pr['degree'] < 3 or pr['kind'] == 'merge' or nid not in legs:
            continue
        R = nodeR.get(nid, 3) * 1.18 + 0.6
        x, y = n['geometry']['coordinates'][:2]
        zc = float(np.mean([z for _, z in legs[nid]]))
        bx, by, _ = px2b(x, y, 0)
        surf = SURF[max(legs[nid], key=lambda l: rank(l[0]))[0]]
        seg = 20
        ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        V = np.r_[[[bx, by, zc + LIFT + 0.05]], np.c_[bx + np.cos(ang) * R, by + np.sin(ang) * R, np.full(seg, zc + LIFT + 0.03)]]
        F = [(0, 1 + k, 1 + (k + 1) % seg) for k in range(seg)]
        vs, fs = by_surf.setdefault(surf, ([], []))
        off = sum(len(q) for q in vs)
        vs.append(V); fs.append(np.asarray(F) + off)
    for surf, (vs, fs) in by_surf.items():
        mesh_obj(f'JUNCTION_PADS_{surf}', np.vstack(vs), np.vstack(fs), pad_coll, mat=mats[surf])


def build_bridges(coll, mats, types):
    roads = {f['properties']['id']: f for f in load('data/roads/roads.geojson')['features']}
    for b in load('data/roads/bridges.geojson')['features']:
        pr = b['properties']
        if pr['kind'] not in ('bridge', 'viaduct', 'overpass'):
            continue
        r = roads.get(pr['road'])
        if r is None:
            continue
        rp = road_pts(r)
        c = np.asarray(b['geometry']['coordinates'], float)
        bx, by, _ = px2b(c[:, 0], c[:, 1], 0)
        d0 = np.hypot(rp[:, 0] - bx[0], rp[:, 1] - by[0]).argmin()
        d1 = np.hypot(rp[:, 0] - bx[-1], rp[:, 1] - by[-1]).argmin()
        i0, i1 = sorted((d0, d1))
        deck = rp[max(0, i0 - 1):i1 + 2]
        if len(deck) < 2:
            continue
        hw = r['properties']['width_m'] / 2 + 0.6
        steel = 'steel' in pr['structure'] or 'truss' in pr['structure']
        Vs, Fs = [], []
        def add(V, F):
            if V is None:
                return
            off = sum(len(v) for v in Vs); Vs.append(V); Fs.append(np.asarray(F) + off)
        # slab top + bottom + edges
        top, Ft = ribbon(deck, -hw, hw, LIFT - 0.02, ground=False)
        bot = top.copy(); bot[:, 2] -= 1.4
        n = len(top)
        V = np.vstack([top, bot])
        k = np.arange(n // 2 - 1) * 2
        F = list(Ft) + list(np.stack([k + n, k + 1 + n, k + 3 + n, k + 2 + n], 1)) \
            + list(np.stack([k, k + n, k + 2 + n, k + 2], 1)) + list(np.stack([k + 1, k + 3, k + 3 + n, k + 1 + n], 1))
        add(V, F)
        # parapets (0.9 m walls)
        for side in (-hw, hw - 0.3):
            a, Fa = ribbon(deck, side, side + 0.3, LIFT, ground=False)
            b2 = a.copy(); b2[:, 2] += 0.9
            nn = len(a); kk = np.arange(nn // 2 - 1) * 2
            add(np.vstack([a, b2]), list(np.stack([kk + nn, kk + 1 + nn, kk + 3 + nn, kk + 2 + nn], 1)) + list(np.stack([kk, kk + 2, kk + 2 + nn, kk + nn], 1)) + list(np.stack([kk + 1, kk + 1 + nn, kk + 3 + nn, kk + 3], 1)))
        # piers
        spacing = 1e9 if pr['kind'] == 'overpass' else (30 if steel else 24)
        acc = 0
        for q in range(1, len(deck) - 1):
            acc += math.hypot(deck[q, 0] - deck[q - 1, 0], deck[q, 1] - deck[q - 1, 1])
            if acc < spacing:
                continue
            acc = 0
            gx, gy = deck[q, 0] / MPP + OX, -deck[q, 1] / MPP + OY
            ground = float(T.at(gx, gy)) - 2
            top_z = deck[q, 2] - 1.3
            if top_z - ground < 1.5:
                continue
            t = deck[q + 1, :2] - deck[q - 1, :2]; t /= max(np.hypot(*t), 1e-9)
            nrm = np.array([t[1], -t[0]])
            corners = []
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corners.append(deck[q, :2] + t * 0.9 * sx + nrm * (hw - 1.2) * sy)
            V = np.array([[cx_, cy_, top_z] for cx_, cy_ in corners] + [[cx_, cy_, ground] for cx_, cy_ in corners])
            F = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
            add(V, F)
        ob = mesh_obj(f"BRIDGE {pr['id']}", np.vstack(Vs), np.vstack(Fs), coll, mat=mats['steel' if steel else 'concrete'])
        ob['kind'] = pr['kind']; ob['structure'] = pr['structure']; ob['world_id'] = pr['id']; ob['road'] = pr['road']


def build_rail(coll, mats):
    for f in load('data/railways/railways.geojson')['features']:
        p = f['properties']
        Pb = road_pts(f)
        hw = p['width_m'] / 2
        Vs, Fs, mi = [], [], []
        def add(V, F, m):
            if V is None:
                return
            off = sum(len(v) for v in Vs); Vs.append(V); Fs.append(np.asarray(F) + off); mi.extend([m] * len(F))
        add(*ribbon(Pb, -hw, hw, 0.15), 0)
        for t in range(p['tracks']):
            c = (t - (p['tracks'] - 1) / 2) * 4.0
            add(*ribbon(Pb, c - 1.3, c + 1.3, 0.22), 1)
            for s in (-0.72, 0.72):
                add(*ribbon(Pb, c + s - 0.05, c + s + 0.05, 0.36), 2)
        ob = mesh_obj(f"RAIL {p['id']}", np.vstack(Vs), np.vstack(Fs), coll)
        for m in (mats['ballast'], mats['tie'], mats['rail']):
            ob.data.materials.append(m)
        ob.data.polygons.foreach_set('material_index', np.asarray(mi, np.int32))
        ob['world_id'] = p['id']; ob['tracks'] = p['tracks']


def build_backdrop(coll, mat):
    meta = load('data/terrain/backdrop.json')
    u = np.fromfile(P('data/terrain/backdrop_u16.bin'), '<u2').reshape(meta['h'], meta['w'])
    h = meta['min_m'] + u.astype(np.float32) * (meta['max_m'] - meta['min_m']) / 65535
    step = 2
    hs = h[::step, ::step]
    ny, nx = hs.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    px = meta['origin_px'][0] + (ii * step + 0.5) * meta['cell_px']
    py = meta['origin_px'][1] + (jj * step + 0.5) * meta['cell_px']
    rx0, ry0, rx1, ry1 = meta['map_rect_cells']
    inside = (ii * step >= rx0 + 1) & (ii * step < rx1 - 1) & (jj * step >= ry0 + 1) & (jj * step < ry1 - 1)
    z = np.where(inside, hs - 30, hs)
    bx, by, bz = px2b(px, py, z)
    V = np.stack([bx, by, bz], -1).reshape(-1, 3)
    a = (jj[:-1, :-1] * nx + ii[:-1, :-1])
    keep = ~(inside[:-1, :-1] & inside[1:, 1:])
    a = a[keep]
    F = np.stack([a, a + nx, a + nx + 1, a + 1], 1)
    ob = mesh_obj('BACKDROP', V, F, coll, mat=mat, smooth=True)
    return ob


# ------------------------------------------------------------------ materials
def node_mat(name, build):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs[0], out.inputs['Surface'])
    build(nt, bsdf, out)
    return m


def inp(bsdf, *names):
    for n in names:
        if n in bsdf.inputs:
            return bsdf.inputs[n]
    raise KeyError(names)


def simple(name, rgb, rough=0.9, metal=0.0, bump=0.0, noise_scale=40.0, var=0.12):
    def b(nt, bsdf, out):
        inp(bsdf, 'Base Color').default_value = (*rgb, 1)
        inp(bsdf, 'Roughness').default_value = rough
        inp(bsdf, 'Metallic').default_value = metal
        if var > 0:
            tc = nt.nodes.new('ShaderNodeTexCoord')
            nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = noise_scale; nz.inputs['Detail'].default_value = 6
            nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
            mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
            mix.inputs['Factor'].default_value = var
            mix.inputs['A'].default_value = (*rgb, 1)
            nt.links.new(nz.outputs['Color'], mix.inputs['B'])
            nt.links.new(mix.outputs['Result'], inp(bsdf, 'Base Color'))
            if bump > 0:
                bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump
                nt.links.new(nz.outputs['Fac'], bp.inputs['Height'])
                nt.links.new(bp.outputs['Normal'], inp(bsdf, 'Normal'))
    return node_mat(name, b)


def terrain_material():
    img = bpy.data.images.load(P('data/terrain/albedo_2x.jpg'), check_existing=True)
    def b(nt, bsdf, out):
        tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img; tex.interpolation = 'Cubic'
        tc = nt.nodes.new('ShaderNodeTexCoord')
        # macro/micro variation (stylized realism: soft painterly breakup, not noise grain)
        n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = 0.02; n1.inputs['Detail'].default_value = 4
        n2 = nt.nodes.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = 0.35; n2.inputs['Detail'].default_value = 8
        nt.links.new(tc.outputs['Object'], n1.inputs['Vector']); nt.links.new(tc.outputs['Object'], n2.inputs['Vector'])
        m1 = nt.nodes.new('ShaderNodeMix'); m1.data_type = 'RGBA'; m1.blend_type = 'OVERLAY'; m1.inputs['Factor'].default_value = 0.18
        nt.links.new(tex.outputs['Color'], m1.inputs['A']); nt.links.new(n1.outputs['Color'], m1.inputs['B'])
        # steep faces -> rock
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Normal'], sep.inputs[0])
        ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.inputs['From Min'].default_value = 0.78; ramp.inputs['From Max'].default_value = 0.62
        nt.links.new(sep.outputs['Z'], ramp.inputs['Value'])
        vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 0.6
        nt.links.new(tc.outputs['Object'], vor.inputs['Vector'])
        rock = nt.nodes.new('ShaderNodeMix'); rock.data_type = 'RGBA'; rock.inputs['A'].default_value = (0.16, 0.15, 0.13, 1); rock.inputs['B'].default_value = (0.27, 0.25, 0.21, 1)
        nt.links.new(vor.outputs['Distance'], rock.inputs['Factor'])
        m2 = nt.nodes.new('ShaderNodeMix'); m2.data_type = 'RGBA'
        nt.links.new(ramp.outputs['Result'], m2.inputs['Factor']); nt.links.new(m1.outputs['Result'], m2.inputs['A']); nt.links.new(rock.outputs['Result'], m2.inputs['B'])
        nt.links.new(m2.outputs['Result'], inp(bsdf, 'Base Color'))
        inp(bsdf, 'Roughness').default_value = 0.95
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.35; bp.inputs['Distance'].default_value = 0.5
        mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'ADD'
        nt.links.new(n2.outputs['Fac'], mx.inputs[0]); nt.links.new(vor.outputs['Distance'], mx.inputs[1])
        nt.links.new(mx.outputs[0], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], inp(bsdf, 'Normal'))
    return node_mat('MAT_Terrain', b)


def water_material():
    def b(nt, bsdf, out):
        inp(bsdf, 'Base Color').default_value = (0.035, 0.085, 0.09, 1)
        inp(bsdf, 'Roughness').default_value = 0.06
        inp(bsdf, 'IOR').default_value = 1.333
        tc = nt.nodes.new('ShaderNodeTexCoord')
        wave = nt.nodes.new('ShaderNodeTexNoise'); wave.inputs['Scale'].default_value = 0.4; wave.inputs['Detail'].default_value = 6
        nt.links.new(tc.outputs['Object'], wave.inputs['Vector'])
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.12
        nt.links.new(wave.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], inp(bsdf, 'Normal'))
    return node_mat('MAT_Water', b)


def foliage_material(name, base):
    def b(nt, bsdf, out):
        at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_type = 'INSTANCER'; at.attribute_name = 'tint'
        inp(bsdf, 'Roughness').default_value = 0.8
        tc = nt.nodes.new('ShaderNodeTexCoord')
        # leaf-cluster breakup: darker hollows, lighter sun-facing clumps
        vz = nt.nodes.new('ShaderNodeTexVoronoi'); vz.inputs['Scale'].default_value = 1.6
        nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.9; nz.inputs['Detail'].default_value = 3
        nt.links.new(tc.outputs['Object'], vz.inputs['Vector']); nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
        mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['To Min'].default_value = 0.55; mr.inputs['To Max'].default_value = 1.15
        nt.links.new(vz.outputs['Distance'], mr.inputs['Value'])
        mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
        nt.links.new(at.outputs['Color'], mul.inputs['A'])
        cc = nt.nodes.new('ShaderNodeCombineColor'); nt.links.new(mr.outputs['Result'], cc.inputs[0]); nt.links.new(mr.outputs['Result'], cc.inputs[1]); nt.links.new(mr.outputs['Result'], cc.inputs[2])
        nt.links.new(cc.outputs[0], mul.inputs['B'])
        nt.links.new(mul.outputs['Result'], inp(bsdf, 'Base Color'))
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.6
        mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'ADD'
        nt.links.new(vz.outputs['Distance'], mx.inputs[0]); nt.links.new(nz.outputs['Fac'], mx.inputs[1])
        nt.links.new(mx.outputs[0], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], inp(bsdf, 'Normal'))
        sub = inp(bsdf, 'Subsurface Weight', 'Subsurface')
        sub.default_value = 0.08
    return node_mat(name, b)


def make_materials():
    return {
        'hwy': LM.pbr('MAT_Road_Asphalt_Hwy', 'asphalt_02', 5.0, (0.24, 0.24, 0.25, 1), 0.8),
        'city': LM.pbr('MAT_Road_Asphalt_City', 'asphalt_02', 5.0, (0.27, 0.27, 0.27, 1), 0.85),
        'local': LM.pbr('MAT_Road_Asphalt_Local', 'asphalt_02', 5.0, (0.3, 0.29, 0.28, 1), 0.9),
        'chip': LM.pbr('MAT_Road_Chipseal', 'asphalt_02', 4.0, (0.36, 0.33, 0.29, 1), 0.95),
        'gravel': LM.pbr('MAT_Road_Gravel', 'gravel_road', 4.0, (1, 0.95, 0.85, 1), 1.0),
        'dirt': simple('MAT_Road_RedClay', (0.33, 0.17, 0.09), 1.0, var=0.35, noise_scale=6, bump=0.2),
        'yellow': simple('MAT_Marking_Yellow', (0.75, 0.52, 0.08), 0.6, var=0.1),
        'white': simple('MAT_Marking_White', (0.78, 0.77, 0.72), 0.6, var=0.1),
        'concrete': simple('MAT_Concrete', (0.46, 0.44, 0.4), 0.85, var=0.2, noise_scale=3, bump=0.1),
        'steel': simple('MAT_Steel_Truss', (0.2, 0.26, 0.22), 0.5, metal=0.6, var=0.1),
        'ballast': simple('MAT_Rail_Ballast', (0.2, 0.18, 0.16), 1.0, var=0.4, noise_scale=20, bump=0.3),
        'tie': simple('MAT_Rail_Ties', (0.1, 0.07, 0.05), 0.95, var=0.2),
        'rail': simple('MAT_Rail_Steel', (0.35, 0.34, 0.33), 0.35, metal=0.9, var=0.05),
        'terrain': LM.terrain_material(P('data/terrain/albedo_2x.jpg')),
        'water': water_material(),
        'backdrop': simple('MAT_Backdrop', (0.025, 0.045, 0.022), 1.0, var=0.5, noise_scale=0.01, bump=0.4),
    }


# ------------------------------------------------------------------ vegetation (Geometry Nodes instancing)
def _blob(bm, center, radius, squash=1.0, jitter=0.28, subdiv=2, seed=0):
    import random, mathutils
    rnd = random.Random(seed)
    g = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=radius, matrix=mathutils.Matrix.Translation(center))
    ph = [rnd.uniform(0, 6.28) for _ in range(3)]
    for v in g['verts']:
        d = v.co - mathutils.Vector(center)
        n = d.normalized()
        # low-frequency lumps = clustered leaf masses (reads as foliage, not a ball)
        f = 1 + jitter * (math.sin(n.x * 4.1 + ph[0]) * math.sin(n.y * 3.7 + ph[1]) * math.sin(n.z * 4.9 + ph[2]))
        v.co = mathutils.Vector(center) + mathutils.Vector((d.x * f, d.y * f, d.z * f * squash))
    return g


def _trunk(bm, h, r0, r1, lean=(0, 0)):
    import mathutils
    bmesh.ops.create_cone(bm, cap_ends=False, segments=7, radius1=r0, radius2=r1, depth=h,
                          matrix=mathutils.Matrix.Translation((lean[0] * 0.5, lean[1] * 0.5, h / 2)))


def tree_prototypes(coll, mats):
    """5 prototypes, named so Collection Info (alphabetical) index == proto id:
    0 oak, 1 tulip poplar, 2 maple (hardwoods) | 3 white pine, 4 hemlock (conifers)."""
    import random
    specs = []
    # --- hardwoods: trunk + clustered canopy masses
    def hardwood(name, h_trunk, masses, seed):
        bm = bmesh.new()
        _trunk(bm, h_trunk, 0.42, 0.22)
        for k, (c, r, sq) in enumerate(masses):
            _blob(bm, c, r, sq, 0.3, 2, seed + k)
        return name, bm
    rnd = random.Random(11)
    oak = [((0, 0, 8.5), 4.4, 0.8)] + [((rnd.uniform(-2.6, 2.6), rnd.uniform(-2.6, 2.6), rnd.uniform(8, 11)), rnd.uniform(2.2, 3.0), 0.85) for _ in range(6)]
    pop = [((0, 0, 12.5), 3.4, 1.35)] + [((rnd.uniform(-1.5, 1.5), rnd.uniform(-1.5, 1.5), rnd.uniform(10, 16)), rnd.uniform(1.8, 2.4), 1.1) for _ in range(6)]
    mpl = [((0, 0, 9), 4.0, 0.95)] + [((rnd.uniform(-2.2, 2.2), rnd.uniform(-2.2, 2.2), rnd.uniform(7.5, 11.5)), rnd.uniform(2.0, 2.7), 0.9) for _ in range(7)]
    specs += [hardwood('P0_Oak', 6.0, oak, 1), hardwood('P1_TulipPoplar', 9.0, pop, 20), hardwood('P2_Maple', 6.5, mpl, 40)]
    # --- conifers: irregular whorls
    import mathutils
    def conifer(name, tiers, h, seed):
        r = random.Random(seed)
        bm = bmesh.new()
        _trunk(bm, h * 0.35, 0.35, 0.2)
        for k, (z, rad, dep) in enumerate(tiers):
            g = bmesh.ops.create_cone(bm, cap_ends=True, segments=9, radius1=rad, radius2=rad * 0.15, depth=dep,
                                      matrix=mathutils.Matrix.Translation((r.uniform(-0.2, 0.2), r.uniform(-0.2, 0.2), z)))
            for v in g['verts']:
                if abs(v.co.z - (z - dep / 2)) < 0.01:
                    ang = math.atan2(v.co.y, v.co.x)
                    f = 1 + 0.22 * math.sin(ang * 3 + k) * r.uniform(0.6, 1.2)
                    v.co.x *= f; v.co.y *= f; v.co.z -= r.uniform(0, 0.6)
        return name, bm
    pine = [(5.5 + i * 2.6, 3.4 - i * 0.45 + (0.4 if i % 2 else 0), 3.2) for i in range(6)]
    hem = [(3.5 + i * 2.1, 3.3 - i * 0.42, 3.4) for i in range(7)]
    specs += [conifer('P3_WhitePine', pine, 20, 5), conifer('P4_Hemlock', hem, 18, 6)]
    protos = []
    for name, bm in specs:
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        me.materials.append(mats['bark']); me.materials.append(mats['leaf_cf' if name[1] in '34' else 'leaf_hw'])
        for p in me.polygons:
            p.material_index = 0 if (abs(p.center.x) < 0.6 and abs(p.center.y) < 0.6 and p.center.z < (5.5 if name[1] in '012' else 3.0)) else 1
            p.use_smooth = True
        ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
        ob.hide_viewport = False
        protos.append(ob)
    return protos


HARDWOOD = [(0x5f, 0x7f, 0x34), (0x6b, 0x8a, 0x3a), (0x7a, 0x8a, 0x36), (0x56, 0x72, 0x2e), (0x8c, 0x8a, 0x34), (0xb0, 0x85, 0x2e), (0xc0, 0x78, 0x2a), (0x9a, 0x5a, 0x26)]
HARD_W = [0.22, 0.2, 0.14, 0.14, 0.1, 0.09, 0.06, 0.05]
CONIFER = [(0x2f, 0x4f, 0x2c), (0x36, 0x58, 0x2f), (0x2a, 0x44, 0x28), (0x3e, 0x5f, 0x34)]


def srgb2lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# early-autumn Blue Ridge palette (linear tints; foliage cards are neutral olive)
PAL = {  # graphics ref: deep olive greens, ~35% warm ochre/amber/rust accents in the hardwoods
    'oak':    [((0.07, 0.085, 0.035), 0.45), ((0.09, 0.09, 0.035), 0.2), ((0.19, 0.11, 0.03), 0.2), ((0.2, 0.07, 0.025), 0.15)],
    'maple':  [((0.075, 0.085, 0.035), 0.3), ((0.28, 0.1, 0.02), 0.3), ((0.27, 0.055, 0.02), 0.2), ((0.26, 0.15, 0.03), 0.2)],
    'poplar': [((0.085, 0.095, 0.035), 0.45), ((0.22, 0.15, 0.035), 0.4), ((0.14, 0.115, 0.035), 0.15)],
    'dogwood': [((0.08, 0.1, 0.04), 0.4), ((0.26, 0.05, 0.03), 0.6)],
    'pine':   [((0.05, 0.08, 0.045), 0.6), ((0.06, 0.09, 0.045), 0.4)],
    'hemlock': [((0.04, 0.068, 0.042), 0.7), ((0.05, 0.075, 0.045), 0.3)],
}
FAMILY = {'A': 'oak', 'B': 'maple', 'C': 'poplar', 'D': 'dogwood', 'E': 'pine', 'F': 'hemlock'}


def build_vegetation(coll, protos):
    tr = np.fromfile(P('public/world/trees.bin'), '<f4').reshape(-1, 5)
    rng = np.random.default_rng(5)
    bx, by, _ = px2b(tr[:, 0], tr[:, 1], 0)
    z = tr[:, 2] - 0.3
    kind = tr[:, 4].astype(np.int32)
    names = [p.name for p in protos]  # alphabetical == Collection Info child order
    hw = [i for i, n in enumerate(names) if n[0] in 'ABC']
    hw_w = np.array([{'A': 0.42, 'B': 0.33, 'C': 0.25}[names[i][0]] / sum(1 for j in hw if names[j][0] == names[i][0]) for i in hw])
    cf = [i for i, n in enumerate(names) if n[0] in 'EF']
    dg = [i for i, n in enumerate(names) if n[0] == 'D']
    proto = np.where(kind == 1, rng.choice(cf, len(tr)), rng.choice(hw, len(tr), p=hw_w / hw_w.sum()))
    under = (kind == 0) & (rng.random(len(tr)) < 0.06)
    proto[under] = rng.choice(dg, under.sum())
    col = np.zeros((len(tr), 4), np.float32); col[:, 3] = 1
    for i, n in enumerate(names):
        sel = proto == i
        if not sel.any():
            continue
        pal = PAL[FAMILY[n[0]]]
        cols = np.array([c for c, _ in pal]); w = np.array([w for _, w in pal])
        pick = rng.choice(len(pal), sel.sum(), p=w / w.sum())
        col[sel, :3] = cols[pick] * (0.82 + 0.36 * rng.random((sel.sum(), 1)))
    scale = tr[:, 3] * (0.85 + 0.3 * rng.random(len(tr)))
    me = bpy.data.meshes.new('VEG_points')
    me.vertices.add(len(tr))
    me.vertices.foreach_set('co', np.stack([bx, by, z], 1).astype(np.float32).ravel())
    for n, typ, data in (('kind', 'INT', proto), ('scale', 'FLOAT', scale), ('rot', 'FLOAT', rng.random(len(tr)) * 6.283)):
        a = me.attributes.new(n, typ, 'POINT'); a.data.foreach_set('value', np.asarray(data, np.int32 if typ == 'INT' else np.float32))
    a = me.attributes.new('tint', 'FLOAT_COLOR', 'POINT'); a.data.foreach_set('color', col.ravel())
    ob = bpy.data.objects.new('VEG_points', me); coll.objects.link(ob)
    pc = collection('VEG_prototypes', coll)
    for p in protos:
        for c in list(p.users_collection):
            c.objects.unlink(p)
        pc.objects.link(p)
        p.location = (0, 0, 0)
    ng = bpy.data.node_groups.new('GN_TreeScatter', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N = ng.nodes; L = ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    ci = N.new('GeometryNodeCollectionInfo'); ci.inputs['Collection'].default_value = pc; ci.inputs['Separate Children'].default_value = True; ci.inputs['Reset Children'].default_value = True
    iop = N.new('GeometryNodeInstanceOnPoints'); iop.inputs['Pick Instance'].default_value = True
    ak = N.new('GeometryNodeInputNamedAttribute'); ak.data_type = 'INT'; ak.inputs['Name'].default_value = 'kind'
    asc = N.new('GeometryNodeInputNamedAttribute'); asc.data_type = 'FLOAT'; asc.inputs['Name'].default_value = 'scale'
    ar = N.new('GeometryNodeInputNamedAttribute'); ar.data_type = 'FLOAT'; ar.inputs['Name'].default_value = 'rot'
    cmb = N.new('ShaderNodeCombineXYZ')
    L.new(ar.outputs['Attribute'], cmb.inputs['Z'])
    L.new(gi.outputs[0], iop.inputs['Points']); L.new(ci.outputs[0], iop.inputs['Instance'])
    L.new(ak.outputs['Attribute'], iop.inputs['Instance Index'])
    L.new(asc.outputs['Attribute'], iop.inputs['Scale']); L.new(cmb.outputs[0], iop.inputs['Rotation'])
    L.new(iop.outputs[0], go.inputs[0])
    mod = ob.modifiers.new('TreeScatter', 'NODES'); mod.node_group = ng
    return ob


# ------------------------------------------------------------------ lighting / world / cameras
def setup_world():
    sc = bpy.context.scene
    _setup_world_base(sc)
    LM.hdri_world(P('assets/external/polyhaven/kloppenheim_06_puresky/kloppenheim_06_puresky_4k.hdr'), float(ARG('--sky', 0.8)), float(ARG('--skyrot', 0)))


def _setup_world_base(sc):
    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    sky = nt.nodes.new('ShaderNodeTexSky'); sky.sky_type = 'NISHITA'
    sky.sun_elevation = math.radians(13); sky.sun_rotation = math.radians(250)  # late afternoon, sun in the WSW
    sky.altitude = 400; sky.air_density = 1.4; sky.dust_density = 2.2; sky.ozone_density = 1.0; sky.sun_intensity = 0.5
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = 0.22
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(sky.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs['Surface'])
    # atmospheric perspective via mist pass (compositor) - cheaper and more controllable than a world volume
    world.mist_settings.start = 1100; world.mist_settings.depth = 9000; world.mist_settings.falloff = 'LINEAR'
    sc.view_layers[0].use_pass_mist = True
    sc.view_layers[0].use_pass_z = True
    sc.use_nodes = True
    ct = sc.node_tree
    for n in list(ct.nodes):
        ct.nodes.remove(n)
    rl = ct.nodes.new('CompositorNodeRLayers'); comp = ct.nodes.new('CompositorNodeComposite')
    mix = ct.nodes.new('CompositorNodeMixRGB'); mix.blend_type = 'MIX'; mix.inputs[2].default_value = (0.5, 0.6, 0.78, 1)
    mul = ct.nodes.new('CompositorNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 0.8
    # mist only on geometry (the sky is at infinite depth and keeps its own colour)
    geo_mask = ct.nodes.new('CompositorNodeMath'); geo_mask.operation = 'LESS_THAN'; geo_mask.inputs[1].default_value = 30000
    ct.links.new(rl.outputs['Depth'], geo_mask.inputs[0])
    mm = ct.nodes.new('CompositorNodeMath'); mm.operation = 'MULTIPLY'
    ct.links.new(rl.outputs['Mist'], mul.inputs[0]); ct.links.new(mul.outputs[0], mm.inputs[0]); ct.links.new(geo_mask.outputs[0], mm.inputs[1])
    ct.links.new(mm.outputs[0], mix.inputs[0])
    ct.links.new(rl.outputs['Image'], mix.inputs[1])
    # cinematic grade (graphics ref: warm highlights, cool shadows, rich saturation)
    cb = ct.nodes.new('CompositorNodeColorBalance'); cb.correction_method = 'LIFT_GAMMA_GAIN'
    cb.lift = (0.97, 0.99, 1.04); cb.gamma = (1.02, 1.0, 0.97); cb.gain = (1.08, 1.02, 0.9)
    hs = ct.nodes.new('CompositorNodeHueSat'); hs.inputs['Saturation'].default_value = 1.18
    ct.links.new(mix.outputs[0], cb.inputs[1]); ct.links.new(cb.outputs[0], hs.inputs['Image'])
    ct.links.new(hs.outputs[0], comp.inputs[0])
    sun = bpy.data.lights.new('SUN', 'SUN'); sun.energy = 6.0; sun.angle = math.radians(0.8); sun.color = (1.0, 0.7, 0.44)
    so = bpy.data.objects.new('SUN', sun); sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(90 - 13), 0, math.radians(250 + 90))


def cam(name, loc_px, look_px, lens=35, coll=None):
    x, y, zoff = loc_px
    tx, ty, tzoff = look_px
    bx, by, _ = px2b(x, y, 0); tbx, tby, _ = px2b(tx, ty, 0)
    bz = float(T.at(min(max(x, 0), W - 1), min(max(y, 0), H - 1))) + zoff
    tbz = float(T.at(tx, ty)) + tzoff
    cd = bpy.data.cameras.new(name); cd.lens = lens; cd.clip_start = 1; cd.clip_end = 40000
    ob = bpy.data.objects.new(name, cd); (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location = (float(bx), float(by), bz)
    d = __import__('mathutils').Vector((float(tbx) - float(bx), float(tby) - float(by), tbz - bz))
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return ob


def setup_cameras():
    c = collection('CAMERAS')
    cams = {
        # hero: from the ridge south-west of Hollow Ridge looking up the valley (graphics ref composition)
        'CAM_LD_Field': cam('CAM_LD_Field', (812, 140, 2.2), (850, 110, -2), 30, c),
        'CAM_Ref_Match': cam('CAM_Ref_Match', (858, 286, 45), (1300, 380, 90), 27, c),
        'CAM_HollowRidge_Overlook': cam('CAM_HollowRidge_Overlook', (1190, 470, 120), (1060, 360, 10), 26, c),
        'CAM_HollowRidge_Valley': cam('CAM_HollowRidge_Valley', (930, 480, 140), (1060, 350, 0), 26, c),
        'CAM_Tannersville': cam('CAM_Tannersville', (1520, 330, 140), (1720, 250, 0), 28, c),
        'CAM_LaurelCity': cam('CAM_LaurelCity', (420, 470, 180), (250, 300, 0), 30, c),
        'CAM_LaurelRiver_Bridges': cam('CAM_LaurelRiver_Bridges', (640, 470, 130), (560, 360, 0), 24, c),
        'CAM_US19_LaurelGap': cam('CAM_US19_LaurelGap', (780, 300, 150), (860, 395, 0), 24, c),
        'CAM_Interchange_SR400': cam('CAM_Interchange_SR400', (360, 470, 110), (300, 400, 0), 30, c),
    }
    # pixel-exact validation camera: orthographic top-down covering the source image
    cd = bpy.data.cameras.new('CAM_Validation_Top'); cd.type = 'ORTHO'; cd.ortho_scale = W * MPP; cd.clip_end = 5000
    ob = bpy.data.objects.new('CAM_Validation_Top', cd); c.objects.link(ob)
    ob.location = (0, 0, 2000); ob.rotation_euler = (0, 0, 0)
    cams['CAM_Validation_Top'] = ob
    return cams


def render_settings(samples=48):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = 12
    sc.cycles.volume_bounces = 0
    sc.cycles.volume_step_rate = 8.0
    sc.render.resolution_x, sc.render.resolution_y = 1600, 900
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.exposure = -0.4
    try:
        sc.view_settings.look = 'AgX - Punchy'
    except TypeError:
        pass
    sc.render.image_settings.file_format = 'JPEG'
    sc.render.image_settings.quality = 92


# ------------------------------------------------------------------ main
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    types = load('data/roads/road_types.json')['types']
    mats = make_materials()
    mats['bark'] = simple('MAT_Bark', (0.06, 0.045, 0.03), 0.95, var=0.2)
    mats['leaf_hw'] = foliage_material('MAT_Foliage_Hardwood', None)
    mats['leaf_cf'] = foliage_material('MAT_Foliage_Conifer', None)
    root = collection('WORLD')
    print('terrain...'); terr = build_terrain(collection('TERRAIN', root), mats['terrain'])
    print('water...'); build_water(collection('WATER', root), mats['water'])
    print('roads...'); build_roads(collection('ROADS', root), mats, types)
    print('bridges...'); build_bridges(collection('BRIDGES', root), mats, types)
    print('rail...'); build_rail(collection('RAIL', root), mats)
    print('backdrop...'); build_backdrop(collection('BACKDROP', root), mats['backdrop'])
    if not OPT('--no-trees'):
        print('vegetation...')
        vc = collection('VEGETATION', root)
        build_vegetation(vc, LT.build_prototypes(vc))
    collection('BUILDINGS (deferred)', root)
    setup_world()
    cams = setup_cameras()
    if not OPT('--no-groundcover'):
        print('ground cover...')
        gcc = collection('GROUNDCOVER', root)
        protos = GC.load_protos(gcc)
        for c in gcc.children:
            c.hide_render = True; c.hide_viewport = True  # prototypes only
        hero = [(o.location.x, o.location.y) for n, o in cams.items() if n != 'CAM_Validation_Top']
        # grass only around the ground in front of each hero camera
        import mathutils
        pts = []
        for n, o in cams.items():
            if n == 'CAM_Validation_Top':
                continue
            f = o.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
            f.z = 0
            if f.length > 0:
                f.normalize()
            pts.append((o.location.x + f.x * 120, o.location.y + f.y * 120))
        GC.scatter_modifier(terr, protos, pts, float(ARG('--grass-radius', 170)))
    render_settings(int(ARG('--samples', 48)))
    bpy.context.scene.camera = cams['CAM_HollowRidge_Overlook']
    out = P(ARG('--out', 'exports/blender/world.blend'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print('saved', out)
    if OPT('--glb'):
        glb = out.replace('.blend', '.glb')
        # glb = engine interchange for terrain/roads/bridges/rail/water/cameras; vegetation ships as
        # instance data (public/world/trees.bin) since 245k GPU instances bloat glTF viewers
        bpy.ops.object.select_all(action='DESELECT')
        for o in bpy.data.objects:
            if not (o.name.startswith(('VEG_', 'BACKDROP')) or o.name[:2] in ('A_', 'B_', 'C_', 'D_', 'E_', 'F_')) and o.type in ('MESH', 'CAMERA'):
                o.select_set(True)
        bpy.ops.export_scene.gltf(filepath=glb, export_format='GLB', use_selection=True, export_apply=True,
                                  export_extras=True, export_cameras=True, export_draco_mesh_compression_enable=True)
        print('exported', glb)


if __name__ == '__main__':
    main()
