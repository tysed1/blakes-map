import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';
import { World, xyz } from '../../core/data';
import { pxToWorld, worldToPx, IMG_W, IMG_H } from '../../core/coords';
import { bus, Selection } from '../../core/bus';
import { buildTerrain, buildWater, CHUNK } from './terrain';
import { buildRoads, buildBridges, buildRail, highlightMesh, roadWorldPoints, setGround } from './roads';
import { buildVegetation, Vegetation } from './vegetation';
import { assetUrl, bin } from '../../core/data';
import { buildBackdrop, buildSky } from './backdrop';

export type CamMode = 'orbit' | 'top' | 'free';

export class World3D {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  private orbit: OrbitControls;
  private mapc: MapControls;
  mode: CamMode = 'free';
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
  private veg: Vegetation | null = null;
  private time = { value: 0 };
  flySpeed = 1;
  onSpeed?: (v: number) => void;
  readonly sunDir = new THREE.Vector3(-1400, 520, -500).normalize();
  readonly haze = new THREE.Color().setRGB(0.6, 0.58, 0.55);

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

    // atmosphere matched to the Blender renders (render.py --sun 255,18 --sky 0.6 --exposure 0.7,
    // kloppenheim_06_puresky HDRI, blue-grey aerial haze, warm low sun from the WSW)
    this.renderer.toneMappingExposure = 1.0;
    this.scene.fog = new THREE.FogExp2(this.haze, 0.00026);
    this.scene.background = this.haze;
    this.sky = buildSky(this.sunDir, null, this.haze);
    this.scene.add(this.sky);
    new THREE.TextureLoader().load(assetUrl('sky.jpg'), (t) => {
      t.colorSpace = THREE.SRGBColorSpace; t.mapping = THREE.EquirectangularReflectionMapping;
      t.generateMipmaps = false; t.minFilter = THREE.LinearFilter; // no mip seam at the equirect wrap
      const skyMat = this.sky.material as THREE.ShaderMaterial;
      skyMat.uniforms.sky.value = t; skyMat.uniforms.hasSky.value = 1;
      const pm = new THREE.PMREMGenerator(this.renderer);
      this.scene.environment = pm.fromEquirectangular(t).texture;
      this.scene.environmentIntensity = 0.55;
      (this.scene as any).environmentRotation = new THREE.Euler(0, Math.atan2(this.sunDir.z, this.sunDir.x) + (0.612 - 0.5) * 2 * Math.PI, 0);
      pm.dispose();
    });
    const hemi = new THREE.HemisphereLight(0xbcd2ee, 0x5a5236, 0.55);
    this.scene.add(hemi);
    this.sun = new THREE.DirectionalLight(0xffd6a6, 3.4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    const sc = this.sun.shadow.camera as THREE.OrthographicCamera;
    sc.left = -600; sc.right = 600; sc.top = 600; sc.bottom = -600; sc.near = 10; sc.far = 5000;
    this.sun.shadow.bias = -0.0004; this.sun.shadow.normalBias = 0.8;
    this.scene.add(this.sun, this.sun.target);
    this.build();
    this.orbit.enabled = false;
    this.heroView();
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
      color: 0x23434a, roughness: 0.08, metalness: 0.0, transparent: true, opacity: 0.88, envMapIntensity: 1.4,
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
    const trees = new THREE.Group(); trees.name = 'trees';
    Object.assign(this.groups, { terrain, water, roads: roads.group, bridges: bridges.group, rail: rail.group, trees });
    bin('vegetation_f32.bin').then((b) => {
      this.veg = buildVegetation(new Float32Array(b), this.time);
      trees.add(this.veg.group);
      this.vegReady = true;
    });
    this.pickMaps.push(roads.pick, bridges.pick, rail.pick);
    for (const g of Object.values(this.groups)) this.scene.add(g);
    Promise.all([fetch(assetUrl('backdrop.json')).then((r) => r.json()), bin('backdrop_u16.bin'), bin('backdrop_water_u8.bin')])
      .then(([meta, hb, wb]) => { const b = buildBackdrop(meta, hb, wb); this.groups.backdrop = b; this.scene.add(b); this.backdropReady = true; });
  }
  backdropReady = false;
  vegReady = false;

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
    if (this.mode === 'free' && m !== 'free') {
      // orbit / top pivot on the ground point the fly camera is looking at
      const d = new THREE.Vector3(); this.camera.getWorldDirection(d);
      const hit = new THREE.Raycaster(this.camera.position, d, 1, 6000).intersectObject(this.groups.terrain, true)[0];
      this.orbit.target.copy(hit ? hit.point : this.camera.position.clone().addScaledVector(d, 600));
    }
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
    } else this.syncFly();
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
    if (this.mode === 'free') this.syncFly();
  }

  private syncFly() {
    const d = new THREE.Vector3(); this.camera.getWorldDirection(d);
    this.yaw = Math.atan2(-d.x, -d.z); this.pitch = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1));
  }

  /** Opening shot matching the Blender CAM_Ref_Match framing (px 858,286 +45 m, looking east). */
  heroView() {
    const [X, , Z] = pxToWorld(858, 286);
    const [TX, , TZ] = pxToWorld(1300, 380);
    const y0 = this.w.terrain.at(858, 286) + 45, ty = this.w.terrain.at(1300, 380) + 90;
    this.camera.position.set(X, y0 + 60, Z);
    const d = new THREE.Vector3(TX - X, ty - y0 - 60, TZ - Z).normalize();
    this.yaw = Math.atan2(-d.x, -d.z); this.pitch = Math.asin(d.y);
    this.camera.lookAt(this.camera.position.clone().add(d));
    this.orbit.target.set(TX, ty, TZ); this.mapc.target.set(TX, ty, TZ);
  }

  focus(x: number, y: number, r = 60) { this.setView(x, y, Math.max(120, r * 6), this.mode === 'top' ? Math.PI / 2 : 0.7, 0.5); }

  targetPx() { const t = this.mode === 'top' ? this.mapc.target : this.orbit.target; return worldToPx(t.x, t.z); }

  start() { if (this.running) return; this.running = true; this.clock.getDelta(); this.loop(); }
  stop() { this.running = false; }

  private loop = () => {
    if (!this.running) return;
    requestAnimationFrame(this.loop);
    const dt = Math.min(0.05, this.clock.getDelta());
    this.time.value += dt;
    if (this.mode === 'orbit') this.orbit.update();
    else if (this.mode === 'top') this.mapc.update();
    else this.fly(dt);
    this.updateCulling();
    this.renderer.render(this.scene, this.camera);
  };

  private updateCulling() {
    const cam = this.camera.position;
    // trees: distance culling per chunk mesh
    this.veg?.update(cam, this.treeDistance);
    // sun + shadow frustum follow the view target (late-afternoon sun from the WSW)
    let t = this.mode === 'top' ? this.mapc.target : this.mode === 'orbit' ? this.orbit.target : cam;
    if (this.mode === 'free') { const d = new THREE.Vector3(); this.camera.getWorldDirection(d); d.y = 0; t = cam.clone().addScaledVector(d.normalize(), 350); t.y = cam.y - 60; }
    this.sun.target.position.copy(t);
    this.sun.position.copy(t).addScaledVector(this.sunDir, 2000);
    this.sky.position.copy(cam);
    // fog density eases with altitude so overview shots stay readable
    const alt = cam.y - t.y;
    (this.scene.fog as THREE.FogExp2).density = THREE.MathUtils.clamp(0.00026 - alt * 0.00000004, 0.00012, 0.00026);
  }

  private fly(dt: number) {
    const speed = (this.keys.has('shift') ? 320 : 80) * this.flySpeed * dt;
    const f = new THREE.Vector3(-Math.sin(this.yaw) * Math.cos(this.pitch), Math.sin(this.pitch), -Math.cos(this.yaw) * Math.cos(this.pitch));
    const r = new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    const p = this.camera.position;
    const k = (a: string, b: string) => this.keys.has(a) || this.keys.has(b);
    if (k('w', 'arrowup')) p.addScaledVector(f, speed);
    if (k('s', 'arrowdown')) p.addScaledVector(f, -speed);
    if (k('d', 'arrowright')) p.addScaledVector(r, speed);
    if (k('a', 'arrowleft')) p.addScaledVector(r, -speed);
    if (k('e', ' ')) p.y += speed;
    if (k('q', 'c')) p.y -= speed;
    p.y = Math.min(p.y, 4000);
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
    dom.addEventListener('wheel', (e) => {
      if (this.mode !== 'free') return;
      e.preventDefault();
      this.flySpeed = THREE.MathUtils.clamp(this.flySpeed * (e.deltaY > 0 ? 0.85 : 1.18), 0.1, 12);
      this.onSpeed?.(this.flySpeed);
    }, { passive: false });
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
