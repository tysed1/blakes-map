import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';
import { World, xyz } from '../../core/data';
import { pxToWorld, worldToPx, IMG_W, IMG_H } from '../../core/coords';
import { bus, Selection } from '../../core/bus';
import { buildTerrain, buildWater, CHUNK } from './terrain';
import { buildRoads, buildBridges, buildRail, highlightMesh, roadWorldPoints, setGround } from './roads';
import { buildTrees } from './vegetation';
import { assetUrl } from '../../core/data';
import { buildBackdrop, buildSky } from './backdrop';

export type CamMode = 'orbit' | 'top' | 'free';

export class World3D {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  private orbit: OrbitControls;
  private mapc: MapControls;
  mode: CamMode = 'orbit';
  private groups: Record<string, THREE.Object3D> = {};
  private pickMaps: Map<THREE.Mesh, string[]>[] = [];
  private sun: THREE.DirectionalLight;
  private hl: THREE.Object3D | null = null;
  private raycaster = new THREE.Raycaster();
  private running = false;
  private keys = new Set<string>();
  private yaw = 0; private pitch = -0.25;
  private terrainMat!: THREE.MeshStandardMaterial;
  private albedo!: THREE.Texture; private baseTex!: THREE.Texture;
  treeDistance = 5200;
  private clock = new THREE.Clock();
  private sky!: THREE.Mesh;

  constructor(private el: HTMLElement, private w: World) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    el.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(50, 1, 1, 30000);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true; this.orbit.dampingFactor = 0.08;
    this.orbit.maxPolarAngle = Math.PI * 0.495; this.orbit.minDistance = 15; this.orbit.maxDistance = 9000;
    this.orbit.screenSpacePanning = false;
    this.mapc = new MapControls(this.camera, this.renderer.domElement);
    this.mapc.enabled = false; this.mapc.enableRotate = false; this.mapc.minDistance = 30; this.mapc.maxDistance = 9000;
    this.mapc.screenSpacePanning = true;

    // atmosphere (graphics ref: warm late-afternoon light, blue haze in the distance)
    this.scene.background = new THREE.Color(0xb9c9d8);
    this.scene.fog = new THREE.FogExp2(0xb4c2cc, 0.00016);
    this.sky = buildSky(new THREE.Vector3(-0.8, 0.28, -0.3));
    this.scene.add(this.sky);
    const hemi = new THREE.HemisphereLight(0xcfe0ff, 0x4d4a33, 0.9);
    this.scene.add(hemi);
    this.sun = new THREE.DirectionalLight(0xffdcb0, 2.6);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    const sc = this.sun.shadow.camera as THREE.OrthographicCamera;
    sc.left = -700; sc.right = 700; sc.top = 700; sc.bottom = -700; sc.near = 10; sc.far = 4000;
    this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 0.6;
    this.scene.add(this.sun, this.sun.target);
    this.build();
    this.setView(1000, 330, 1400, 0.9, 0.6);
    this.bindInput();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    bus.select.on((s) => this.showHighlight(s));
  }

  private build() {
    const loader = new THREE.TextureLoader();
    this.albedo = loader.load(assetUrl('albedo.jpg'));
    this.albedo.colorSpace = THREE.SRGBColorSpace; this.albedo.anisotropy = 8;
    this.baseTex = loader.load(assetUrl('base_map.png'));
    this.baseTex.colorSpace = THREE.SRGBColorSpace; this.baseTex.anisotropy = 8;
    this.terrainMat = new THREE.MeshStandardMaterial({ map: this.albedo, roughness: 0.96, metalness: 0 });
    const terrain = buildTerrain(this.w.terrain, this.terrainMat);
    const water = buildWater(this.w.waterLevel, this.w.landuseRaster, new THREE.MeshStandardMaterial({
      color: 0x355f6e, roughness: 0.35, metalness: 0.0, transparent: true, opacity: 0.9,
    }));
    const tex = (hex: number, rough = 0.9) => new THREE.MeshStandardMaterial({ color: hex, roughness: rough, metalness: 0, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
    const roadMats: Record<string, THREE.Material> = {
      hwy: tex(0x4a4b4e, 0.85), city: tex(0x55565a, 0.88), local: tex(0x5e5d5a, 0.9), chip: tex(0x6d665c, 0.95),
      gravel: tex(0x9c8f76, 1), dirt: tex(0x9a6a47, 1), yellow: tex(0xe0b83c, 0.7), white: tex(0xe8e6de, 0.7), median: tex(0x5b6e3a, 1),
    };
    setGround(this.w.terrain);
    const roads = buildRoads(this.w, roadMats);
    const bridges = buildBridges(this.w, this.w.terrain, { concrete: new THREE.MeshStandardMaterial({ color: 0xb8b2a6, roughness: 0.9, side: THREE.DoubleSide }), steel: new THREE.MeshStandardMaterial({ color: 0x6f7b73, roughness: 0.6, metalness: 0.4, side: THREE.DoubleSide }) });
    const rail = buildRail(this.w, { ballast: tex(0x6b625a, 1), tie: tex(0x4a3b2e, 1), rail: new THREE.MeshStandardMaterial({ color: 0x9aa0a6, metalness: 0.7, roughness: 0.35 }) });
    const trees = buildTrees(this.w.trees, this.w.manifest.trees.stride);
    Object.assign(this.groups, { terrain, water, roads: roads.group, bridges: bridges.group, rail: rail.group, trees });
    this.pickMaps.push(roads.pick, bridges.pick, rail.pick);
    for (const g of Object.values(this.groups)) this.scene.add(g);
    Promise.all([fetch(assetUrl('backdrop.json')).then((r) => r.json()), fetch(assetUrl('backdrop_u16.bin')).then((r) => r.arrayBuffer()), fetch(assetUrl('backdrop_water_u8.bin')).then((r) => r.arrayBuffer())])
      .then(([meta, hb, wb]) => { const b = buildBackdrop(meta, hb, wb); this.groups.backdrop = b; this.scene.add(b); this.backdropReady = true; });
  }
  backdropReady = false;

  setLayer(k: string, on: boolean) {
    const g = this.groups[k];
    if (g) g.visible = on;
  }

  /** Drape the source map onto the terrain instead of the stylized albedo (alignment check). */
  setDrape(on: boolean) {
    this.terrainMat.map = on ? this.baseTex : this.albedo;
    this.terrainMat.needsUpdate = true;
  }

  setMode(m: CamMode) {
    const target = this.orbit.target.clone();
    this.mode = m;
    this.orbit.enabled = m === 'orbit';
    this.mapc.enabled = m === 'top';
    if (m === 'top') {
      const h = Math.max(300, this.camera.position.distanceTo(target));
      this.mapc.target.copy(target);
      this.camera.position.set(target.x, target.y + h, target.z + 0.01);
      this.camera.up.set(0, 1, 0);
      this.mapc.update();
    } else if (m === 'orbit') {
      this.orbit.target.copy(this.mode === 'orbit' ? target : this.mapc.target);
      this.orbit.update();
    } else {
      const d = new THREE.Vector3(); this.camera.getWorldDirection(d);
      this.yaw = Math.atan2(-d.x, -d.z); this.pitch = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1));
    }
  }

  /** Position camera to look at source px (x,y) from distance dist. */
  setView(x: number, y: number, dist = 600, pitch = 0.85, yaw = 0.4) {
    const [X, , Z] = pxToWorld(x, y);
    const Y = this.w.terrain.at(x, y);
    const t = new THREE.Vector3(X, Y, Z);
    const off = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)).multiplyScalar(dist);
    this.camera.position.copy(t).add(off);
    this.orbit.target.copy(t); this.mapc.target.copy(t);
    if (this.mode === 'top') this.camera.position.set(t.x, t.y + dist, t.z + 0.01);
    this.camera.lookAt(t);
    this.orbit.update();
  }

  focus(x: number, y: number, r = 60) { this.setView(x, y, Math.max(120, r * 6), this.mode === 'top' ? Math.PI / 2 : 0.7, 0.5); }

  targetPx() { const t = this.mode === 'top' ? this.mapc.target : this.orbit.target; return worldToPx(t.x, t.z); }

  start() { if (this.running) return; this.running = true; this.clock.getDelta(); this.loop(); }
  stop() { this.running = false; }

  private loop = () => {
    if (!this.running) return;
    requestAnimationFrame(this.loop);
    const dt = Math.min(0.05, this.clock.getDelta());
    if (this.mode === 'orbit') this.orbit.update();
    else if (this.mode === 'top') this.mapc.update();
    else this.fly(dt);
    this.updateCulling();
    this.renderer.render(this.scene, this.camera);
  };

  private updateCulling() {
    const cam = this.camera.position;
    // trees: distance culling per chunk mesh
    const td2 = this.treeDistance * this.treeDistance;
    for (const m of this.groups.trees.children as THREE.InstancedMesh[]) {
      const s = m.boundingSphere;
      if (!s) continue;
      m.visible = s.center.distanceToSquared(cam) < td2;
    }
    // sun + shadow frustum follow the view target (late-afternoon sun from the WSW)
    const t = this.mode === 'top' ? this.mapc.target : this.mode === 'orbit' ? this.orbit.target : cam;
    this.sun.target.position.copy(t);
    this.sun.position.set(t.x - 1400, t.y + 700, t.z - 500);
    this.sky.position.copy(cam);
    // fog density eases with altitude so overview shots stay readable
    const alt = cam.y - t.y;
    (this.scene.fog as THREE.FogExp2).density = THREE.MathUtils.clamp(0.00019 - alt * 0.00000003, 0.00009, 0.00019);
  }

  private fly(dt: number) {
    const speed = (this.keys.has('shift') ? 260 : 70) * dt;
    const f = new THREE.Vector3(-Math.sin(this.yaw) * Math.cos(this.pitch), Math.sin(this.pitch), -Math.cos(this.yaw) * Math.cos(this.pitch));
    const r = new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    const p = this.camera.position;
    if (this.keys.has('w')) p.addScaledVector(f, speed);
    if (this.keys.has('s')) p.addScaledVector(f, -speed);
    if (this.keys.has('d')) p.addScaledVector(r, speed);
    if (this.keys.has('a')) p.addScaledVector(r, -speed);
    if (this.keys.has('e')) p.y += speed;
    if (this.keys.has('q')) p.y -= speed;
    const px = worldToPx(p.x, p.z);
    const ground = this.w.terrain.at(Math.min(Math.max(px.x, 0), IMG_W), Math.min(Math.max(px.y, 0), IMG_H)) + 1.8;
    if (p.y < ground) p.y = ground;
    this.camera.lookAt(p.clone().add(f));
  }

  private bindInput() {
    const dom = this.renderer.domElement;
    dom.tabIndex = 0;
    window.addEventListener('keydown', (e) => { if (this.running && !(e.target instanceof HTMLInputElement)) this.keys.add(e.key.toLowerCase()); if (e.shiftKey) this.keys.add('shift'); });
    window.addEventListener('keyup', (e) => { this.keys.delete(e.key.toLowerCase()); if (!e.shiftKey) this.keys.delete('shift'); });
    let drag = false, down: [number, number] | null = null, last: [number, number] = [0, 0];
    dom.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; last = down; drag = false; });
    dom.addEventListener('pointermove', (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) drag = true;
      if (this.mode === 'free' && down && e.buttons) {
        this.yaw -= (e.clientX - last[0]) * 0.004; this.pitch = THREE.MathUtils.clamp(this.pitch - (e.clientY - last[1]) * 0.004, -1.5, 1.5);
      }
      last = [e.clientX, e.clientY];
      this.hover(e);
    });
    dom.addEventListener('pointerup', (e) => { if (!drag) this.click(e); down = null; });
    dom.addEventListener('pointerleave', () => bus.cursor.emit(null));
  }

  private ndc(e: PointerEvent) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  private hoverT = 0;
  private hover(e: PointerEvent) {
    const now = performance.now();
    if (now - this.hoverT < 60) return;
    this.hoverT = now;
    const hit = this.cast(e, [this.groups.terrain]);
    if (hit) { const p = worldToPx(hit.point.x, hit.point.z); bus.cursor.emit({ ...p, z: hit.point.y }); }
  }

  private cast(e: PointerEvent, objs: THREE.Object3D[]) {
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const hits = this.raycaster.intersectObjects(objs, true).filter((h) => h.object.visible);
    return hits[0] ?? null;
  }

  private click(e: PointerEvent) {
    const pickables = [this.groups.roads, this.groups.bridges, this.groups.rail, this.groups.terrain, this.groups.water].filter((g) => g.visible);
    const hit = this.cast(e, pickables);
    if (!hit) return;
    const p = worldToPx(hit.point.x, hit.point.z);
    for (const pm of this.pickMaps) {
      const owners = pm.get(hit.object as THREE.Mesh);
      if (owners && hit.faceIndex != null) {
        const id = owners[hit.faceIndex];
        const kind = hit.object.userData.kind as string;
        const data = kind === 'rail' ? this.w.rail.find((r) => r.id === id) : kind === 'bridge' ? this.w.bridges.find((b) => b.id === id) : this.w.roads.find((r) => r.id === id);
        if (data) { bus.select.emit({ kind: kind as any, id, data, at: p }); return; }
      }
    }
    const i = Math.floor(p.y) * IMG_W + Math.floor(p.x);
    const cls = this.w.manifest.landuse[this.w.landuseRaster[i]];
    bus.select.emit({ kind: 'landuse', id: cls?.name ?? 'none', at: p, data: { class: cls?.name, elevation_m: +hit.point.y.toFixed(1) } });
  }

  private showHighlight(s: Selection) {
    if (this.hl) { this.scene.remove(this.hl); this.hl = null; }
    if (!s) return;
    const d = s.data;
    if (s.kind === 'road' && d?.c) {
      this.hl = highlightMesh(roadWorldPoints(d), d.width_m);
    } else if (s.kind === 'rail' && d?.c) {
      this.hl = highlightMesh(xyz(d.c).map(([x, y, z]) => pxToWorld(x, y, z)), d.width_m);
    } else if (s.at) {
      const [X, , Z] = pxToWorld(s.at.x, s.at.y);
      const Y = this.w.terrain.at(s.at.x, s.at.y);
      const ring = new THREE.Mesh(new THREE.RingGeometry(6, 9, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x00e5ff, depthTest: false, transparent: true }));
      ring.position.set(X, Y + 1, Z);
      this.hl = ring;
    }
    if (this.hl) { this.hl.renderOrder = 10; this.scene.add(this.hl); }
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Render a still for QA / screenshots. */
  updateCullingPublic() { this.orbit.update(); this.updateCulling(); }
  snapshot(): string { this.updateCulling(); this.renderer.render(this.scene, this.camera); return this.renderer.domElement.toDataURL('image/png'); }
}

function skyTexture() {
  const c = document.createElement('canvas');
  c.width = 2; c.height = 256;
  const g = c.getContext('2d')!;
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, '#7fa6d6');
  grad.addColorStop(0.45, '#b9cde0');
  grad.addColorStop(0.62, '#f2d7b4');
  grad.addColorStop(1, '#c9b79a');
  g.fillStyle = grad; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export { CHUNK };
