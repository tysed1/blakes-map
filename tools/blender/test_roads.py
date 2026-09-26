"""Road-level test renders for roads & infrastructure (Agent 2). Not part of the world build.

    blender -b --python tools/blender/test_roads.py -- --cams US19_Gap_Driver,HR_Main_Ped \
        [--res 640x360] [--samples 12] [--bbox x0,y0,x1,y1] [--trees] [--no-infra] [--outdir exports/a2/renders] \
        [--save exports/a2/roadtest.blend]

Builds terrain + water + roads/bridges/rail (+ infrastructure) through build_world's own
functions, then temporary cameras placed ON the road data: driver eye (1.2 m above the
right-hand lane), pedestrian (1.7 m on the sidewalk / shoulder), low cinematic (5 m) and
medium aerial (100 m). Camera specs: (anchor px, look-at px, height mode, height, lens,
lateral offset m). Heights are relative to the nearest road surface (or terrain).
"""
import bpy, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_world as BW
import lib_roads as LR

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OPT = lambda k: k in ARGS


def ARG(k, d=None):
    return ARGS[ARGS.index(k) + 1] if k in ARGS else d


# name: (x, y) anchor px, (x, y) look-at px, mode ('road' | 'terrain'), height m, lens, lateral m (+ right of view dir)
CAMS = {
    # US 19 through the Laurel Gap (2-lane highway, the ref's "winding highway")
    'US19_Gap_Driver': ((676, 337), (726, 355), 'road', 1.2, 30, 1.8),
    'US19_Gap_Low': ((745, 352), (800, 373), 'road', 5.0, 28, 12.0),
    'US19_Gap_Aerial': ((700, 285), (790, 372), 'terrain', 100.0, 30, 0.0),
    'US19_Bridge_Low': ((538, 300), (545, 317), 'terrain', 6.0, 28, 0.0),
    'US19_Bridge_Driver': ((512, 318), (560, 315), 'road', 1.2, 30, 1.8),
    # Hollow Ridge main street (pedestrian on the sidewalk) + aerial
    'HR_Main_Ped': ((1020, 352), (1080, 341), 'road', 1.7, 28, 6.5),
    'HR_Main_Driver': ((1000, 356), (1060, 344), 'road', 1.2, 30, 1.8),
    'HR_Aerial': ((1000, 420), (1060, 345), 'terrain', 100.0, 30, 0.0),
    # SR 400 / US 19 diamond
    'IC_SR400_Aerial': ((360, 470), (310, 410), 'terrain', 110.0, 30, 0.0),
    'IC_SR400_Low': ((332, 425), (305, 408), 'terrain', 5.0, 26, 0.0),
    'IC_SR400_Driver': ((270, 443), (300, 420), 'road', 1.2, 30, 1.8),
    # SR 60 / US 76 compressed diamond
    'IC_SR60_Aerial': ((1930, 230), (1875, 150), 'terrain', 110.0, 30, 0.0),
    # rural road + steel truss (Hollow Creek Rd)
    'Rural_Truss_Driver': ((884, 442), (913, 444), 'road', 1.2, 30, 1.5),
    'Rural_Truss_Low': ((905, 462), (913, 444), 'terrain', 5.0, 28, 0.0),
    'Rural_Driver': ((801, 451), (826, 442), 'road', 1.2, 30, 1.5),
    'Gravel_Low': ((694, 306), (720, 318), 'road', 3.0, 28, 0.0),
    'Jct_Mill_Low': ((780, 375), (763, 360), 'terrain', 8.0, 26, 0.0),
}


class RoadZ:
    """Nearest road-centreline elevation + direction (from the graded data)."""

    def __init__(self):
        pts, zs = [], []
        for f in LR.load('data/roads/roads.geojson')['features']:
            if f['properties'].get('virtual'):
                continue
            c = np.asarray(f['geometry']['coordinates'], float)
            if c.shape[1] < 3:
                continue
            for a, b in zip(c[:-1], c[1:]):
                n = max(1, int(np.hypot(*(b[:2] - a[:2])) * 2))
                t = np.linspace(0, 1, n, endpoint=False)[:, None]
                q = a + (b - a) * t
                pts.append(q[:, :2]); zs.append(q[:, 2])
        self.P = np.vstack(pts); self.Z = np.concatenate(zs)

    def at(self, x, y):
        d = np.hypot(self.P[:, 0] - x, self.P[:, 1] - y)
        i = int(np.argmin(d))
        return float(self.Z[i]), self.P[i]


def make_cam(name, spec, rz, coll):
    (x, y), (tx, ty), mode, h, lens, lat = spec
    if mode == 'road':
        z0, q = rz.at(x, y)
        x, y = q
        tz, tq = rz.at(tx, ty)
        tx, ty = tq
    else:
        z0 = float(BW.T.at(x, y)); tz = float(BW.T.at(tx, ty))
    bx, by = LR.px2w(x, y); tbx, tby = LR.px2w(tx, ty)
    d = np.array([tbx - bx, tby - by]); d /= max(np.hypot(*d), 1e-9)
    right = np.array([d[1], -d[0]])
    bx, by = bx + right[0] * lat, by + right[1] * lat
    cd = bpy.data.cameras.new('TCAM_' + name); cd.lens = lens; cd.clip_start = 0.3; cd.clip_end = 30000
    ob = bpy.data.objects.new('TCAM_' + name, cd); coll.objects.link(ob)
    ob.location = (float(bx), float(by), z0 + h)
    import mathutils
    look_z = tz + (1.0 if mode == 'road' else 0.0) if h < 20 else tz
    v = mathutils.Vector((float(tbx) - float(bx), float(tby) - float(by), look_z - (z0 + h)))
    ob.rotation_euler = v.to_track_quat('-Z', 'Y').to_euler()
    return ob


def main():
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    types = LR.load('data/roads/road_types.json')['types']
    mats = BW.make_materials()
    root = BW.collection('WORLD')
    print('terrain...'); BW.build_terrain(BW.collection('TERRAIN', root), mats['terrain'])
    print('water...'); BW.build_water(BW.collection('WATER', root), mats['water'])
    bbox = tuple(map(float, ARG('--bbox').split(','))) if ARG('--bbox') else None
    ctx = {'T': BW.T, 'collection': BW.collection, 'mesh_obj': BW.mesh_obj, 'luw': BW.LUW, 'bbox': bbox, 'types': types}
    print('roads...'); LR.build(root, ctx)
    if not OPT('--no-infra'):
        try:
            import lib_infrastructure as LI
            print('infrastructure...'); LI.build(root, ctx)
        except ImportError:
            pass
    if OPT('--trees'):
        vc = BW.collection('VEGETATION', root)
        BW.LT.build_vegetation(vc)
    print('backdrop...'); BW.build_backdrop(BW.collection('BACKDROP', root), mats['backdrop'])
    BW.setup_world()
    BW.render_settings(int(ARG('--samples', 12)))
    sc = bpy.context.scene
    w, h = map(int, ARG('--res', '640x360').split('x'))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.cycles.use_adaptive_sampling = True
    # sun as in the coordinator's renders (late afternoon)
    import importlib.util
    rz = RoadZ()
    cams = ARG('--cams', 'US19_Gap_Driver').split(',')
    cc = BW.collection('A2_TEST_CAMERAS')
    outdir = BW.P(ARG('--outdir', 'exports/a2/renders'))
    os.makedirs(outdir, exist_ok=True)
    print(f'build {time.time() - t0:.0f}s')
    if ARG('--save'):
        bpy.ops.wm.save_as_mainfile(filepath=BW.P(ARG('--save')), compress=True)
    sun = bpy.data.objects.get('SUN')
    if sun is not None:
        az, el = map(float, ARG('--sun', '255,18').split(','))
        th = math.radians(90 - az)
        import mathutils
        dv = mathutils.Vector((math.cos(th) * math.cos(math.radians(el)), math.sin(th) * math.cos(math.radians(el)), math.sin(math.radians(el))))
        sun.rotation_euler = (-dv).to_track_quat('-Z', 'Y').to_euler()
        sun.data.energy = float(ARG('--sun-energy', 10))
    sc.view_settings.exposure = float(ARG('--exposure', 0.7))
    for n in sc.world.node_tree.nodes:
        if n.type == 'BACKGROUND':
            n.inputs['Strength'].default_value = float(ARG('--sky', 0.6))
    if ARG('--hide'):
        for pre in ARG('--hide').split(','):
            for o in bpy.data.objects:
                if o.name.startswith(pre):
                    o.hide_render = True
    for name in cams:
        if name.startswith('TOP'):
            # TOP:x0,y0,x1,y1 orthographic plan view of a px box (geometry check)
            x0, y0, x1, y1 = map(float, name.split(':')[1].split('_'))
            cd = bpy.data.cameras.new('TCAM_TOP'); cd.type = 'ORTHO'; cd.ortho_scale = max(x1 - x0, (y1 - y0) * w / h) * LR.MPP
            cd.clip_end = 5000
            ob = bpy.data.objects.new('TCAM_TOP', cd); cc.objects.link(ob)
            bx, by = LR.px2w((x0 + x1) / 2, (y0 + y1) / 2)
            ob.location = (float(bx), float(by), 1500.0)
            sc.camera = ob
            sc.render.filepath = os.path.join(outdir, 'TOP_%d_%d.jpg' % (x0, y0))
            bpy.ops.render.render(write_still=True)
            continue
        if name not in CAMS:
            print('unknown cam', name); continue
        ob = make_cam(name, CAMS[name], rz, cc)
        print('cam', name, tuple(round(v, 1) for v in ob.location))
        sc.camera = ob
        sc.render.filepath = os.path.join(outdir, name + '.jpg')
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print('rendered', sc.render.filepath, f'{time.time() - t1:.0f}s')


if __name__ == '__main__':
    main()
