"""Export the engineered roads / junctions / bridges / rail / roadside infrastructure / water to the web viewer.

    blender -b --factory-startup --python tools/blender/export_web_infra.py -- [--stats] [--bbox x0,y0,x1,y1]

Builds the same geometry as build_world.py (lib_roads, lib_infrastructure, lib_water - the real libs,
not a re-implementation), then writes compact buffers for src/components/world3d/infra.ts:

    public/world/infra/infra.json   material table, prototypes, chunk + draw directory
    public/world/infra/infra_{roads,water,struct}.bin   vertex / index / instance buffers (4-byte aligned blocks)
    public/world/infra/*.jpg        <=1k textures (python3 tools/blender/export_web_infra.py: system python + PIL)

Web coordinates (three.js, Y-up):  (x, y, z) = (bx, bz, -by)  (Blender Z-up world metres).

Draw groups (one draw per group per 500 m chunk; materials of a group share one shader,
the material is a per-vertex id so a chunk stays one draw call):
    ground   pavement / shoulders / sidewalks / curbs / verges / ballast (asphalt markings from the
             lib_roads attributes: rl, rs, len-rs, hw, mk, np, age, surf, esh, lw, uin, sa, sb, xa, xb)
    struct   bridges, piers, abutments, girders, trusses, rails, tunnel portals, retaining walls (always drawn)
    detail   guardrail W-beam, culvert headwalls, crossing panels, boulders (distance culled)
    bed      riverbed + wet shore film (alpha edges)
    water    river / creek surfaces (depth, foam, wake, shore, flow)
    wires    power / telephone / fence wires (polylines -> screen-space ribbons on the web)
    inst     prototypes (poles, signs, posts, ties, gates, delineators) + per-instance transforms

Geometry is lightly decimated where it is exactly linear (rows of swept grids that linear
interpolation reproduces within 2-3 cm and attribute tolerance) - lossless in look, ~3x fewer
triangles on straight roads.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
TEXTURES = {  # web id -> (polyhaven asset, map kind, grayscale?, size)
    'asphalt_l': ('asphalt_02', 'diff', True, 1024), 'asphalt_h': ('asphalt_02', 'disp', True, 1024),
    'gravel_l': ('gravel_road', 'diff', True, 512), 'clay_l': ('red_dirt_mud_01', 'diff', True, 512),
    'bed_rocks': ('river_small_rocks', 'diff', False, 1024), 'bed_gravel': ('gravel_ground_01', 'diff', False, 512),
    'rock': ('rock_face', 'diff', False, 512),
}


# ------------------------------------------------------------------ textures (system python + PIL)
def textures():
    from PIL import Image, ImageOps
    OUT = os.path.join(ROOT, 'public/world/infra')
    os.makedirs(OUT, exist_ok=True)
    ph = os.path.join(ROOT, 'assets/external/polyhaven')
    for k, (aid, kind, gray, size) in TEXTURES.items():
        d = os.path.join(ph, aid, 'textures')
        cand = sorted(x for x in os.listdir(d) if f'_{kind}_' in x and x.lower().endswith(('.jpg', '.png')))
        if not cand:
            print('missing', aid, kind); continue
        im = Image.open(os.path.join(d, cand[0]))
        if im.mode in ('I;16', 'I;16B', 'I'):
            a = np.asarray(im, np.float32)
            im = Image.fromarray(np.clip(a / (256.0 if a.max() > 255 else 1.0), 0, 255).astype(np.uint8))
        im = im.convert('L' if (gray and kind == 'disp') else 'RGB')
        im = im.resize((size, size), Image.LANCZOS)
        if gray and kind == 'disp':
            im = ImageOps.autocontrast(im, cutoff=0.5)       # height (non-colour data)
        elif gray:
            # detail luminance: linear luminance normalised to mean 0.5, stored linear (the web shader
            # multiplies by 2 -> mean 1 and applies the Blender material's measured mean / curve)
            a = np.asarray(im, np.float32) / 255
            a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
            L = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            print('  ', k, 'linear mean luminance', round(float(L.mean()), 4))
            im = Image.fromarray(np.clip(L / L.mean() * 0.5 * 255, 0, 255).astype(np.uint8))
        im.save(os.path.join(OUT, f'{k}.jpg'), quality=82, optimize=True)
        print('texture', k, os.path.getsize(os.path.join(OUT, f'{k}.jpg')) // 1024, 'kB')


try:
    import bpy  # noqa: F401
except ImportError:  # system python (PIL): only the web textures
    textures()
    sys.exit(0)
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
sys.argv = [sys.argv[0]]  # build_world parses its own ARGS at import; keep it clean
import build_world as BW  # noqa: E402  (terrain sampler T, landuse weights, collections)
import lib_roads as LR  # noqa: E402
import lib_infrastructure as LI  # noqa: E402
import lib_water as LW  # noqa: E402
import lib_materials as LM  # noqa: E402

OUT = os.path.join(ROOT, 'public/world/infra')
OPT = lambda k: k in ARGS
def ARG(k, d=None):
    return ARGS[ARGS.index(k) + 1] if k in ARGS else d

CHUNK_M = 500.0          # web chunk size (m) for the pavement group
CHUNK_BIG = 1000.0       # all other groups (fewer draw calls; they are cheap or distance culled)
GROUP_CHUNK = {'ground': CHUNK_M}
X0, Z0 = -2500.0, -833.75  # world min corner (px 0,0)


# ------------------------------------------------------------------ lossless row decimation of swept grids
_grid_orig = LR.Mesh.grid
DECIMATE = [not OPT('--no-decimate')]
STATS = {'rows_in': 0, 'rows_out': 0}


def _keep_rows(G, attrs2d, tol_col, atol=0.015, max_span=80.0):
    """Greedy: keep a row only if dropping it would move any vertex more than tol_col (per column) or
    change a 2D attribute by more than atol, relative to linear interpolation between kept rows."""
    r = G.shape[0]
    if r <= 2:
        return np.arange(r)
    mid = G[:, G.shape[1] // 2]
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(mid, axis=0), axis=1))]
    keep = [0]
    i = 0
    while i < r - 1:
        j = i + 1
        while j + 1 < r and t[j + 1] - t[i] <= max_span:
            c = j + 1
            den = max(t[c] - t[i], 1e-9)
            f = ((t[i + 1:c] - t[i]) / den)[:, None]
            P = G[i][None] * (1 - f[:, :, None]) + G[c][None] * f[:, :, None]
            err = np.linalg.norm(P - G[i + 1:c], axis=2)
            if (err > tol_col[None]).any():
                break
            ok = True
            for A in attrs2d:
                Ai = A[i][None] * (1 - f) + A[c][None] * f
                if (np.abs(Ai - A[i + 1:c]) > atol).any():
                    ok = False
                    break
            if not ok:
                break
            j = c
        keep.append(j)
        i = j
    return np.asarray(keep)


def _grid_decimated(self, G, band_mats, **attrs):
    r, c, _ = G.shape
    if not DECIMATE[0] or r < 4:
        return _grid_orig(self, G, band_mats, **attrs)
    A2 = {k: np.asarray(v) for k, v in attrs.items() if np.ndim(v) == 2 and np.shape(v)[0] == r}
    tol = np.full(c, 0.03)
    if 'rl' in A2 and 'hw' in attrs:
        hw = float(np.max(attrs['hw'])) if np.ndim(attrs['hw']) else float(attrs['hw'])
        u = np.abs(A2['rl']).max(0)
        tol = np.where(u > hw + 0.6, 0.3, 0.03)   # verge / ditch / slope columns follow the terrain loosely
        tol[[0, -1]] = 0.4  # skirt columns (under the terrain)
    keep = _keep_rows(np.asarray(G, float), list(A2.values()), tol)
    STATS['rows_in'] += r; STATS['rows_out'] += len(keep)
    at = {}
    for k, v in attrs.items():
        at[k] = np.asarray(v)[keep] if k in A2 else v
    return _grid_orig(self, np.asarray(G)[keep], band_mats, **at)


LR.Mesh.grid = _grid_decimated


# junction pads: lib_roads drapes them with <= 3 m triangles (x64 per ear-clip triangle); the web terrain
# is sampled at 2.5 m and the web shader biases pavement depth, so one 9 m level keeps the drape (1.9 M -> ~0.15 M tris)
_sd = LR._subdivide
LR._subdivide = lambda V, F, max_len=3.0, levels=3: _sd(V, F, 9.0, 1)

# boulders: 80-triangle rocks (web budget) instead of 320
_bp = LW._boulder_proto
LW._boulder_proto = lambda rng, subdiv=2: _bp(rng, 1)


# ------------------------------------------------------------------ build
def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    t0 = time.time()
    BW.LUW = BW.landuse_weights()
    LM.add_terrain_attributes = lambda *a, **k: None   # terrain-material ecology attrs: not needed for export
    root = BW.collection('WORLD')
    print('water...', flush=True)
    LW.build_water(BW.collection('WATER', root), None)
    rctx = {'T': BW.T, 'collection': BW.collection, 'mesh_obj': BW.mesh_obj, 'luw': None}
    bb = ARG('--bbox')
    if bb:
        rctx['bbox'] = tuple(float(v) for v in bb.split(','))
    print('roads...', flush=True)
    LR.build(root, rctx)
    print('infrastructure...', flush=True)
    LI.build(root, rctx)
    print(f'built in {time.time() - t0:.0f}s; decimation rows {STATS["rows_in"]} -> {STATS["rows_out"]}', flush=True)


def inventory():
    tot = 0
    by = {}
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            me = ob.data
            me.calc_loop_triangles()
            n = len(me.loop_triangles)
            key = ob.name.split(' ')[0] if ' ' in ob.name else ob.name.split('_C')[0]
            if ob.modifiers:
                key = 'INST ' + ob.name
                n = len(me.vertices)
            by.setdefault(key, [0, 0, set()])
            by[key][0] += 1; by[key][1] += n
            by[key][2].update(m.name for m in me.materials if m)
            if not ob.modifiers:
                tot += n
        elif ob.type == 'CURVE':
            by.setdefault('CURVE ' + ob.name, [1, sum(len(s.points) for s in ob.data.splines), set()])
    for k, (c, n, ms) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f'  {k:50s} objs {c:5d}  tris/insts {n:9d}  {sorted(ms)}')
    print('total static tris', tot)


# ------------------------------------------------------------------ material table
# ground kinds (one shader, per-vertex kind)
GROUND = {'MAT_Road_Asphalt': 0, 'MAT_Road_Gravel': 1, 'MAT_Road_RedClay': 2, 'MAT_Road_GravelShoulder': 3,
          'MAT_Road_Concrete': 4, 'MAT_Road_Sidewalk': 5, 'MAT_Road_Verge': 6, 'MAT_Terrain_PBR': 6, 'MAT_Terrain': 6,
          'MAT_Rail_Ballast': 7}
# structure palette (linear albedo as in lib_roads / lib_infrastructure / lib_water), shader modes:
# 0 flat (fine noise), 1 cast concrete, 2 painted / weathered steel (rust), 3 wood, 4 rubble masonry,
# 5 riprap, 6 river boulder (moss, wet), 7 galvanized
MODES = {'flat': 0, 'concrete': 1, 'steel': 2, 'wood': 3, 'stone': 4, 'riprap': 5, 'rock': 6, 'galv': 7}
PALETTE = [
    # name, colour (linear), roughness, metalness, mode, rust
    ('MAT_Bridge_Concrete', (0.44, 0.43, 0.40), 0.88, 0.0, 'concrete', 0.0),
    ('MAT_Bridge_Steel_Truss', (0.2, 0.225, 0.215), 0.55, 0.55, 'steel', 0.35),
    ('MAT_Bridge_Steel_Girder', (0.22, 0.24, 0.23), 0.6, 0.55, 'steel', 0.4),
    ('MAT_Bridge_Railing', (0.5, 0.52, 0.52), 0.4, 0.55, 'steel', 0.15),
    ('MAT_Bridge_Riprap', (0.24, 0.23, 0.21), 0.9, 0.0, 'riprap', 0.0),
    ('MAT_Rail_Steel', (0.32, 0.29, 0.26), 0.45, 0.55, 'steel', 0.6),
    ('MAT_Rail_Tie', (0.075, 0.055, 0.04), 0.95, 0.0, 'wood', 0.0),
    ('MAT_Infra_PoleWood', (0.11, 0.085, 0.065), 0.93, 0.0, 'wood', 0.0),
    ('MAT_Infra_Timber', (0.14, 0.11, 0.08), 0.93, 0.0, 'wood', 0.0),
    ('MAT_Infra_InsulatorGlass', (0.35, 0.55, 0.45), 0.15, 0.0, 'flat', 0.0),
    ('MAT_Infra_Galvanized', (0.42, 0.43, 0.43), 0.45, 0.8, 'galv', 0.0),
    ('MAT_Infra_SteelDark', (0.12, 0.12, 0.12), 0.5, 0.6, 'flat', 0.0),
    ('MAT_Infra_Wire', (0.05, 0.05, 0.05), 0.4, 0.7, 'flat', 0.0),
    ('MAT_Sign_White', (0.72, 0.72, 0.7), 0.5, 0.0, 'flat', 0.0),
    ('MAT_Sign_Black', (0.012, 0.012, 0.012), 0.6, 0.0, 'flat', 0.0),
    ('MAT_Sign_Red', (0.45, 0.018, 0.015), 0.5, 0.0, 'flat', 0.0),
    ('MAT_Sign_Yellow', (0.78, 0.55, 0.03), 0.5, 0.0, 'flat', 0.0),
    ('MAT_Sign_Back', (0.3, 0.3, 0.3), 0.5, 0.7, 'galv', 0.0),
    ('MAT_Infra_StoneWall', (0.3, 0.28, 0.24), 0.92, 0.0, 'stone', 0.0),
    ('MAT_Infra_Void', (0.005, 0.005, 0.005), 1.0, 0.0, 'flat', 0.0),
    ('MAT_Infra_Reflector', (0.8, 0.45, 0.05), 0.2, 0.0, 'flat', 0.0),
    ('MAT_Infra_SignalLens', (0.32, 0.012, 0.008), 0.15, 0.0, 'flat', 0.0),
    ('MAT_Infra_Refractor', (0.62, 0.6, 0.52), 0.2, 0.0, 'flat', 0.0),
    ('MAT_Signal_Housing', (0.6, 0.42, 0.03), 0.55, 0.0, 'flat', 0.0),
    ('MAT_Signal_Lens_Off', (0.05, 0.04, 0.03), 0.15, 0.0, 'flat', 0.0),
    ('MAT_Signal_Red_Lit', (0.9, 0.06, 0.02), 0.3, 0.0, 'flat', 0.0, 2.5),
    ('MAT_Signal_Green_Lit', (0.05, 0.75, 0.45), 0.3, 0.0, 'flat', 0.0, 2.0),
    ('MAT_Signal_Walk_Lit', (0.95, 0.4, 0.05), 0.3, 0.0, 'flat', 0.0, 1.6),
    ('MAT_RiverRock', (0.22, 0.21, 0.19), 0.8, 0.0, 'rock', 0.0),
    ('MAT_Road_Concrete', (0.3, 0.295, 0.275), 0.88, 0.0, 'concrete', 0.0),
    ('MAT_Unknown', (0.3, 0.3, 0.3), 0.8, 0.0, 'flat', 0.0),
]
PAL_ID = {p[0]: i for i, p in enumerate(PALETTE)}
WATER_MATS = {'MAT_Water'}
BED_MATS = {'MAT_Riverbed': 0, 'MAT_WetShore': 1}
def group_of(obname, matname):
    """Draw group for a triangle of object obname with material matname."""
    if matname in WATER_MATS:
        return 'water'
    if matname in BED_MATS:
        return 'bed'
    if matname in GROUND and not obname.startswith(('BRIDGE', 'INFRA')):
        return 'verge' if GROUND[matname] == 6 else 'ground'
    if obname.startswith(('BRIDGE', 'INFRA Tunnel', 'INFRA Retaining')):
        return 'struct'
    if matname == 'MAT_Rail_Steel':
        return 'struct'   # rails stay drawn to the horizon (the ballast is ground): no line ending at 750 m
    return 'detail'


def web(v):
    """Blender Z-up (n,3) -> web Y-up."""
    v = np.asarray(v, np.float64).reshape(-1, 3)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


# ------------------------------------------------------------------ mesh extraction
def mesh_arrays(ob):
    """Triangulated, vertex-split (by corner normal + material) arrays of a mesh object (web coords)."""
    me = ob.data
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    if nt == 0:
        return None
    tl = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get('loops', tl)
    tm = np.empty(nt, np.int32); me.loop_triangles.foreach_get('material_index', tm)
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype(np.float64)
    cn = np.empty(len(me.loops) * 3, np.float32); me.corner_normals.foreach_get('vector', cn)
    cn = cn.reshape(-1, 3)
    mw = np.array(ob.matrix_world)
    if not np.allclose(mw, np.eye(4)):
        co = co @ mw[:3, :3].T + mw[:3, 3]
        cn = cn @ np.linalg.inv(mw[:3, :3])
        cn /= np.maximum(np.linalg.norm(cn, axis=1), 1e-9)[:, None]
    vi = lv[tl]
    nq = np.clip(np.round(cn[tl] * 127), -127, 127).astype(np.int32)
    mc = np.repeat(tm, 3)
    key = np.stack([vi, nq[:, 0], nq[:, 1], nq[:, 2], mc], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    attrs = {}
    for a in me.attributes:
        if a.domain != 'POINT' or a.name.startswith('.') or a.name in ('position',):
            continue
        if a.data_type == 'FLOAT':
            d = np.empty(len(me.vertices), np.float32); a.data.foreach_get('value', d)
        elif a.data_type == 'FLOAT_VECTOR':
            d = np.empty(len(me.vertices) * 3, np.float32); a.data.foreach_get('vector', d); d = d.reshape(-1, 3)
        else:
            continue
        attrs[a.name] = d[uk[:, 0]]
    # open-boundary vertices (edges used by one face): alpha-edged films fade to 0 there
    le = np.empty(len(me.loops), np.int32); me.loops.foreach_get('edge_index', le)
    ecount = np.bincount(le, minlength=len(me.edges))
    ev = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get('vertices', ev)
    bnd = np.zeros(len(me.vertices), np.float32)
    bnd[ev.reshape(-1, 2)[ecount == 1].ravel()] = 1.0
    attrs['_bnd'] = bnd[uk[:, 0]]
    mats = [m.name if m else 'MAT_Unknown' for m in me.materials] or ['MAT_Unknown']
    n = uk[:, 1:4].astype(np.float32) / 127.0
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-6)[:, None]
    return {'P': web(co[uk[:, 0]]), 'N': web(n), 'mat': uk[:, 4], 'I': inv.reshape(-1, 3), 'tmat': tm,
            'mats': mats, 'attrs': attrs}


def g_attrs(group, A, sub):
    """Per-vertex attribute blocks of a draw group for vertices sub of mesh arrays A."""
    at = A['attrs']
    n = len(sub)
    get = lambda k, d=0.0: at[k][sub] if k in at else np.full(n, d, np.float32)
    mname = np.asarray(A['mats'])[np.minimum(A['mat'][sub], len(A['mats']) - 1)]
    u8 = lambda x: np.clip(np.round(x), 0, 255).astype(np.uint8)
    if group == 'ground':
        L = get('len')
        rs = get('rs')
        kind = np.array([GROUND.get(m, 6) for m in mname], np.float32)
        return {
            # stations along the edge (m) as u16 in 1/32 m (<= 2048 m edges; dash / stop-bar precision 3 cm)
            'gS': np.clip(np.round(np.stack([rs, np.where(L > 0, L - rs, 0)], 1) * 32), 0, 65535).astype(np.uint16),
            'gR': np.stack([np.clip(np.round(get('rl') * 100), -32767, 32767), np.clip(np.round(get('hw') * 100), 0, 32767)], 1).astype(np.int16),
            'gB': np.stack([u8(get('mk')), u8(get('np') * 255), u8(get('age') * 255), u8(get('surf') * 255)], 1),
            'gC': np.stack([u8(get('esh') * 20), u8(get('lw') * 20), u8(get('uin') * 20), u8(kind)], 1),
            'gD': np.stack([u8(get('sa') * 4), u8(get('sb') * 4), u8(get('xa') * 4), u8(get('xb') * 4)], 1),
        }
    if group == 'verge':
        return {}
    if group == 'water':
        fl = at['flow'][sub] if 'flow' in at else np.zeros((n, 3), np.float32)
        return {
            'wA': np.stack([u8(get('depth') / 6 * 255), u8(get('foam') * 255), u8(get('wake') * 255), u8(get('shore') / 6 * 255)], 1),
            'wF': np.stack([np.clip(np.round(fl[:, 0] * 127), -127, 127), np.clip(np.round(-fl[:, 1] * 127), -127, 127),
                            np.round(get('_bnd') * 127), np.zeros(n)], 1).astype(np.int8),
        }
    if group == 'bed':
        kind = np.array([BED_MATS.get(m, 0) for m in mname], np.float32)
        e = np.where(kind > 0, get('wet'), get('edge')) * (1.0 - get('_bnd'))
        return {'bA': np.stack([u8(kind), u8((get('depth') + 2) / 8 * 255), u8((get('bank') + 1) / 2 * 255), u8(e * 255)], 1)}
    pid = np.array([PAL_ID.get(m, PAL_ID['MAT_Unknown']) for m in mname], np.float32)
    return {'sA': np.stack([u8(pid), u8(get('wet') * 255), np.zeros(n, np.uint8), np.zeros(n, np.uint8)], 1)}


# ------------------------------------------------------------------ binary writer
class Bin:
    def __init__(self, name):
        self.name = name
        self.b = bytearray()

    def put(self, arr):
        arr = np.ascontiguousarray(arr)
        while len(self.b) % 4:
            self.b.append(0)
        off = len(self.b)
        self.b += arr.tobytes()
        return {'f': self.name, 'o': off, 'n': int(arr.size), 't': arr.dtype.str.lstrip('<|=')}


def quant_pos(P):
    lo = P.min(0); hi = P.max(0)
    ext = np.maximum(hi - lo, 1e-3)
    q = np.round((P - lo) / ext * 65535).astype(np.uint16)
    return q, lo, ext


def chunk_of(xz, size=CHUNK_M):
    cx = np.clip(((xz[:, 0] - X0) // size).astype(int), 0, int(5000 // size) - 1 + (5000 % size > 0))
    cz = np.clip(((xz[:, 1] - Z0) // size).astype(int), 0, int(1667.5 // size))
    return cx, cz


def export():
    os.makedirs(OUT, exist_ok=True)
    acc = {}  # (cx, cz, group) -> dict of lists
    protos = {}
    inst = []
    wires = {}
    proto_names = set()
    for c in bpy.data.collections:
        if c.name == 'A2_PROTOTYPES':
            proto_names |= {o.name for o in c.all_objects}
    t0 = time.time()
    for ob in bpy.data.objects:
        if ob.type == 'CURVE':
            cu = ob.data
            kind = 'power' if 'Power' in ob.name else 'tele' if 'Telephone' in ob.name else 'fence'
            for sp in cu.splines:
                co = np.empty(len(sp.points) * 4, np.float32); sp.points.foreach_get('co', co)
                P = web(co.reshape(-1, 4)[:, :3])
                cx, cz = chunk_of(P[:1, [0, 2]], CHUNK_BIG)
                wires.setdefault((int(cx[0]), int(cz[0])), []).append((P, cu.bevel_depth, kind))
            continue
        if ob.type != 'MESH':
            continue
        if ob.modifiers and ob.modifiers[0].type == 'NODES':
            ng = ob.modifiers[0].node_group
            pname = ng.name.replace('GN_A2_Instancer_', '') if ng else None
            if not pname:
                continue
            me = ob.data
            n = len(me.vertices)
            co = np.empty(n * 3, np.float32); me.vertices.foreach_get('co', co)
            g = lambda k: (lambda d: (me.attributes[k].data.foreach_get('value', d), d)[1])(np.empty(n, np.float32)) if k in me.attributes else np.zeros(n, np.float32)
            rx, ry, rz, sc = g('rotx'), g('roty'), g('rotz'), g('scl')
            sc = np.where(sc == 0, 1, sc)
            P = web(co)
            # Blender euler XYZ (R = Rz Ry Rx) -> web: RotY(rz) RotZ(-ry) RotX(rx)  == three Euler(rx, rz, -ry, 'YZX')
            rows = np.stack([P[:, 0], P[:, 1], P[:, 2], rx, rz, -ry, sc], 1).astype(np.float32)
            inst.append({'name': ob.name, 'proto': pname, 'rows': rows})
            continue
        if ob.name in proto_names:
            A = mesh_arrays(ob)
            if A:
                protos[ob.name] = A
            continue
        A = mesh_arrays(ob)
        if A is None:
            continue
        tmn = np.asarray(A['mats'])[np.minimum(A['tmat'], len(A['mats']) - 1)]
        grp = np.array([group_of(ob.name, m) for m in A['mats']] + ['detail'])[np.minimum(A['tmat'], len(A['mats']))]
        cen = A['P'][A['I']].mean(1)
        for gname in np.unique(grp):
            cx, cz = chunk_of(cen[:, [0, 2]], GROUP_CHUNK.get(str(gname), CHUNK_BIG))
            key = cx * 100 + cz * 10
            for kk in np.unique(key[grp == gname]):
                sel = (grp == gname) & (key == kk)
                I = A['I'][sel]
                sub, inv = np.unique(I.ravel(), return_inverse=True)
                a = acc.setdefault((int(kk // 100), int((kk % 100) // 10), str(gname)), {'P': [], 'N': [], 'I': [], 'A': [], 'n': 0, 't': 0, 'pick': []})
                a['P'].append(A['P'][sub]); a['N'].append(A['N'][sub])
                a['I'].append(inv.reshape(-1, 3) + a['n'])
                a['A'].append(g_attrs(str(gname), A, sub))
                a['pick'].append([a['t'], ob.name])
                a['n'] += len(sub); a['t'] += int(sel.sum())
    print(f'extracted in {time.time() - t0:.0f}s', flush=True)
    BINS = {k: Bin(f'infra_{k}.bin') for k in ('roads', 'water', 'struct')}
    BIN_OF = {'ground': 'roads', 'verge': 'roads', 'water': 'water', 'bed': 'water', 'struct': 'struct', 'detail': 'struct'}
    meta = {'version': 2, 'chunk_m': {'ground': CHUNK_M, 'other': CHUNK_BIG}, 'origin': [X0, Z0], 'palette': [], 'ground_kinds': GROUND,
            'textures': {k: f'{k}.jpg' for k in TEXTURES}, 'chunks': [], 'protos': {}, 'instances': [], 'wires': []}
    for pe in PALETTE:
        name, col, rough, metal, mode, rust = pe[:6]
        meta['palette'].append({'name': name, 'color': col, 'rough': rough, 'metal': metal, 'mode': MODES[mode], 'rust': rust,
                                'emit': pe[6] if len(pe) > 6 else 0.0})
    tri = {}
    for (cx, cz, gname), a in sorted(acc.items()):
        P = np.vstack(a['P']); N = np.vstack(a['N']); I = np.vstack(a['I'])
        q, lo, ext = quant_pos(P)
        B = BINS[BIN_OF[gname]]
        nq = np.clip(np.round(N * 127), -127, 127).astype(np.int8)
        nq = np.c_[nq, np.zeros(len(nq), np.int8)]
        idx = I.astype(np.uint16 if len(P) < 65536 else np.uint32).ravel()
        d = {'cx': cx, 'cz': cz, 'group': gname, 'lo': lo.tolist(), 'ext': ext.tolist(), 'nv': len(P), 'nt': len(I),
             'pos': B.put(q), 'idx': B.put(idx), 'attrs': {}, 'pick': a['pick']}
        if gname != 'water':
            d['nrm'] = B.put(nq)
        for k in a['A'][0]:
            d['attrs'][k] = B.put(np.vstack([x[k] for x in a['A']]))
        meta['chunks'].append(d)
        tri[gname] = tri.get(gname, 0) + len(I)
    B = BINS['struct']
    for name, A in protos.items():
        allsub = np.arange(len(A['P']))
        at = g_attrs('struct', A, allsub)
        nq = np.c_[np.clip(np.round(A['N'] * 127), -127, 127).astype(np.int8), np.zeros(len(A['N']), np.int8)]
        meta['protos'][name] = {'nv': len(A['P']), 'nt': len(A['I']), 'pos': B.put(A['P'].astype(np.float32)), 'nrm': B.put(nq),
                                'idx': B.put(A['I'].astype(np.uint16).ravel()), 'sA': B.put(at['sA'])}
    for it in inst:
        meta['instances'].append({'name': it['name'], 'proto': it['proto'], 'count': len(it['rows']), 'rows': B.put(it['rows'])})
    for (cx, cz), lst in sorted(wires.items()):
        pts = np.vstack([w[0] for w in lst]).astype(np.float32)
        cnt = np.array([len(w[0]) for w in lst], np.uint16)
        rad = np.array([w[1] * 1000 for w in lst], np.uint8)  # radius (mm)
        meta['wires'].append({'cx': cx, 'cz': cz, 'n': len(lst), 'pts': B.put(pts), 'cnt': B.put(cnt), 'rad': B.put(rad)})
    import gzip
    for B in BINS.values():
        with open(os.path.join(OUT, B.name), 'wb') as f:
            f.write(B.b)
        print(f'{B.name} {len(B.b) / 1e6:.1f} MB (gzip {len(gzip.compress(bytes(B.b), 6)) / 1e6:.1f} MB)')
    if os.path.exists(os.path.join(OUT, 'infra.bin')):
        os.remove(os.path.join(OUT, 'infra.bin'))
    with open(os.path.join(OUT, 'infra.json'), 'w') as f:
        json.dump(meta, f, separators=(',', ':'))
    ninst = {it['proto']: len(it['rows']) for it in inst}
    print('triangles per group:', tri, 'total', sum(tri.values()))
    print('draws (chunk x group):', len(meta['chunks']), ' protos:', len(protos), ' instances:', ninst)
    print(f'infra.json {os.path.getsize(os.path.join(OUT, "infra.json")) / 1e3:.0f} kB')


if __name__ == '__main__':
    build()
    if OPT('--stats'):
        inventory()
    export()
