import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';
import { World, xyz } from '../../core/data';
import { pxToWorld, worldToPx, IMG_W, IMG_H } from '../../core/coords';
import { bus, Selection } from '../../core/bus';
import { buildTerrain, CHUNK } from './terrain';
import { highlightMesh, roadWorldPoints, setGround } from './roads';
import { buildInfra, Infra } from './infra'; // infra (agent)
import { buildTrees, Trees } from './trees';
import { assetUrl, bin } from '../../core/data';
import { buildBackdrop, buildSky } from './backdrop';
import { installAtmosphere, createCinematic, Cinematic } from './cinematic';
import { terrainMaterial } from './terrainMaterial';
import { buildGroundcover, Groundcover } from './groundcover'; // groundcover (agent)

export type CamMode = 'orbit' | 'top' | 'free';
export type Quality = 'low' | 'medium' | 'high' | 'ultra';
export const QUALITY: Record<Quality, { maxDpr: number; bloom: boolean }> = {
  low: { maxDpr: 1, bloom: false }, medium: { maxDpr: 1.25, bloom: true }, high: { maxDpr: 1.5, bloom: true }, ultra: { maxDpr: 2, bloom: true },
};

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
  private veg: Trees | null = null;
  private skyOff?: number;
  private time = { value: 0 };
  flySpeed = 1;
  onSpeed?: (v: number) => void;
  readonly sunDir = new THREE.Vector3(-1400, 360, -560).normalize();   // ~13.5 deg above the WSW horizon
  readonly sunColor = new THREE.Color().setRGB(1.0, 0.72, 0.46);
  readonly sunIntensity = 8;
  readonly haze = new THREE.Color().setRGB(0.19, 0.26, 0.38);          // blue ridge haze
  readonly sunHaze = new THREE.Color().setRGB(0.95, 0.66, 0.4);         // golden forward scatter
  cine!: Cinematic;

  constructor(private el: HTMLElement, private w: World) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: location.hostname === "localhost", powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.info.autoReset = false;
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

    // golden-hour look (graphics ref.png): low warm sun from the WSW, blue aerial perspective with
    // warm forward scatter toward the sun, HDRI sky + image-based ambient, cascaded shadows, filmic post
    installAtmosphere({ sunDir: this.sunDir, haze: this.haze, sunHaze: this.sunHaze, baseHeight: 330, falloff: 0.0028 });
    this.renderer.toneMappingExposure = 1.1;
    this.scene.fog = new THREE.FogExp2(this.haze, 0.00016);
    this.scene.background = this.haze;
    this.sky = buildSky(this.sunDir, null, this.haze, this.sunHaze);
    this.scene.add(this.sky);
    new THREE.TextureLoader().load(assetUrl('sky.jpg'), (t) => {
      t.colorSpace = THREE.SRGBColorSpace; t.mapping = THREE.EquirectangularReflectionMapping;
      t.generateMipmaps = false; t.minFilter = THREE.LinearFilter; // no mip seam at the equirect wrap
      const skyMat = this.sky.material as THREE.ShaderMaterial;
      skyMat.uniforms.sky.value = t; skyMat.uniforms.hasSky.value = 1;
      const pm = new THREE.PMREMGenerator(this.renderer);
      this.scene.environment = pm.fromEquirectangular(t).texture;
      this.scene.environmentIntensity = 0.5;
      (this.scene as any).environmentRotation = new THREE.Euler(0, Math.atan2(this.sunDir.z, this.sunDir.x) + (0.612 - 0.5) * 2 * Math.PI, 0);
      pm.dispose();
    });
    const hemi = new THREE.HemisphereLight(0xa9c4ea, 0x4a4630, 0.45);
    this.scene.add(hemi);
    this.cine = createCinematic(this.renderer, this.scene, this.camera, this.sunDir, this.sunColor, this.sunIntensity);
    this.sun = this.cine.csm!.lights[0];
    this.build();
    this.orbit.enabled = false;
    this.heroView();
    fetch(assetUrl('cams.json')).then((r) => r.json()).then((c) => { this.cams = c; this.camsReady = true; }).catch(() => (this.camsReady = true));
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
    this.terrainMat = terrainMaterial(this.albedo, 8);
    const terrain = buildTerrain(this.w.terrain, this.terrainMat);
    // --- infra (agent) ---
    // engineered roads / junctions / bridges / rail / roadside infrastructure / rivers, exported from the
    // Blender build (tools/blender/export_web_infra.py -> public/world/infra/); layer groups stay
    // this.groups.roads / bridges / rail / water, filled when the export has loaded
    setGround(this.w.terrain); // selection highlight drape (roads.ts)
    const infraPick = new Map<THREE.Mesh, string[]>();
    const infraLayer = (name: string) => { const g = new THREE.Group(); g.name = name; return { group: g, pick: infraPick }; };
    const roads = infraLayer('roads'), bridges = infraLayer('bridges'), rail = infraLayer('rail'), water = infraLayer('water').group;
    buildInfra({ albedo: this.albedo, time: this.time }).then((inf) => {
      this.infra = inf;
      roads.group.add(inf.roads); bridges.group.add(inf.bridges); rail.group.add(inf.rail); water.add(inf.water);
      inf.pick.forEach((v, k) => infraPick.set(k, v));
      this.camera.updateMatrixWorld(); inf.update(this.camera);
      this.infraReady = true;
    }).catch((e) => { console.error('infra', e); this.infraReady = true; });
    // --- /infra ---
    const trees = new THREE.Group(); trees.name = 'trees';
    Object.assign(this.groups, { terrain, water, roads: roads.group, bridges: bridges.group, rail: rail.group, trees });
    buildTrees(this.w.terrain, this.sunDir).then((t) => {
      this.veg = t;
      t.uniforms.uTime = this.time;
      t.setQuality(this.quality);
      trees.add(t.group);
      t.update(this.camera, true);
      this.vegReady = true;
    }).catch((e) => { console.error('trees', e); this.vegReady = true; });
    // --- groundcover (agent) ---
    const groundcover = new THREE.Group(); groundcover.name = 'groundcover';
    this.groups.groundcover = groundcover;
    const veg = () => this.veg; // grass sways with the trees: same uTime, live view of the trees' uWind
    buildGroundcover(this.w, { albedo: this.albedo, sunDir: this.sunDir, quality: this.quality, uniforms: { uTime: this.time, uWind: { get value() { return veg()?.uniforms.uWind.value ?? 1; } } } })
      .then((gc) => {
        this.gc = gc; groundcover.add(gc.group); this.groundcoverQuality = (q) => gc.setQuality(q);
        this.camera.updateMatrixWorld(); gc.update(this.camera); this.gcReady = true;
      })
      .catch((e) => { console.error('groundcover', e); this.gcReady = true; });
    // --- /groundcover ---
    this.pickMaps.push(roads.pick, bridges.pick, rail.pick);
    for (const g of Object.values(this.groups)) this.scene.add(g);
    Promise.all([fetch(assetUrl('backdrop.json')).then((r) => r.json()), bin('backdrop_u16.bin'), bin('backdrop_water_u8.bin')])
      .then(([meta, hb, wb]) => { const b = buildBackdrop(meta, hb, wb); this.groups.backdrop = b; this.scene.add(b); this.backdropReady = true; });
  }
  backdropReady = false;
  vegReady = false;
  camsReady = false;
  gc: Groundcover | null = null; gcReady = false; // groundcover (agent)
  infra: Infra | null = null; infraReady = false; // infra (agent)

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
    this.syaw = this.yaw; this.spitch = this.pitch; this.vel.set(0, 0, 0);
  }

  cams: Record<string, { pos: number[]; dir: number[]; fov: number }> = {};
  /** Jump to a Blender scene camera (public/world/cams.json, tools/blender/dump_cams.py). */
  setCam(name: string) {
    const c = this.cams[name];
    if (!c) return;
    this.setMode('free');
    this.camera.position.fromArray(c.pos);
    const d = new THREE.Vector3().fromArray(c.dir);
    this.camera.fov = c.fov; this.camera.updateProjectionMatrix();
    this.camera.lookAt(this.camera.position.clone().add(d));
    this.syncFly();
  }

  /** Opening shot. */
  heroView() {
    // high oblique over Hollow Ridge looking south down the valley, low sun raking in from the west
    this.setView(1050, 330, 650, 0.3, Math.PI);
    this.syncFly();
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
    this.renderFrame();
  };

  private prepT = 0;
  /** Composer frame; new materials (late-loading layers) get hooked into the cascaded shadows first. */
  private renderFrame() {
    if (this.prepT-- <= 0) { this.cine.prepare(this.scene); this.prepT = 30; }
    this.renderer.info.reset();
    const t0 = performance.now();
    this.cine.render();
    this.adapt(performance.now() - t0);
  }

  // ---------------------------------------------------------------- quality + adaptive resolution
  quality: Quality = 'high';
  renderScale = 1;
  autoScale = true;
  onStats?: (s: { fps: number; ms: number; tris: number; calls: number; scale: number; trees: number }) => void;
  private frames: number[] = [];
  private lastStat = 0;
  setQuality(q: Quality) {
    this.quality = q;
    const Q = QUALITY[q];
    this.veg?.setQuality(q);
    this.cine.setShadowQuality(q);
    this.cine.bloom.enabled = Q.bloom;
    this.renderScale = 1;
    this.applyScale();
    this.groundcoverQuality?.(q);
  }
  groundcoverQuality?: (q: Quality) => void;
  private applyScale() {
    const Q = QUALITY[this.quality];
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, Q.maxDpr) * this.renderScale);
    this.resize();
  }
  /** Frame-time governor: drop render resolution before dropping frames, recover when there's headroom. */
  private adapt(cpuMs: number) {
    const now = performance.now();
    this.frames.push(now);
    while (this.frames.length && now - this.frames[0] > 1000) this.frames.shift();
    if (now - this.lastStat < 500) return;
    this.lastStat = now;
    const fps = this.frames.length;
    const info = this.renderer.info.render;
    this.onStats?.({ fps, ms: cpuMs, tris: info.triangles, calls: info.calls, scale: this.renderScale, trees: this.veg?.stats().instances ?? 0 });
    if (!this.autoScale || this.frames.length < 5) return;
    if (fps < 42 && this.renderScale > 0.55) { this.renderScale = Math.max(0.55, this.renderScale - 0.1); this.applyScale(); }
    else if (fps > 58 && this.renderScale < 1) { this.renderScale = Math.min(1, this.renderScale + 0.05); this.applyScale(); }
  }

  private updateCulling() {
    const cam = this.camera.position;
    // trees: distance culling per chunk mesh
    this.camera.updateMatrixWorld(); this.veg?.update(this.camera);
    this.gc?.update(this.camera); // groundcover (agent)
    this.infra?.update(this.camera); // infra (agent)
    this.sky.position.copy(cam);
    // clouds drift slowly across the HDRI sky
    const su = (this.sky.material as THREE.ShaderMaterial).uniforms;
    this.skyOff ??= su.uOff.value as number;
    su.uOff.value = this.skyOff + this.time.value * 0.00035;
  }

  private vel = new THREE.Vector3();
  private flight: { p0: THREE.Vector3; p1: THREE.Vector3; q0: THREE.Quaternion; q1: THREE.Quaternion; f0: number; f1: number; t: number; dur: number } | null = null;
  private tourList: string[] = [];
  private tourI = 0;
  private tourHold = 0;

  /** Eased flight to a Blender shot (position ease-in-out, rotation slerp, arcs up over the terrain). */
  flyTo(name: string, dur = 6) {
    const c = this.cams[name];
    if (!c) return;
    if (this.mode !== 'free') this.setMode('free');
    const p1 = new THREE.Vector3().fromArray(c.pos);
    const look = new THREE.Object3D(); look.position.copy(p1); look.lookAt(p1.clone().add(new THREE.Vector3().fromArray(c.dir)));
    this.flight = { p0: this.camera.position.clone(), p1, q0: this.camera.quaternion.clone(), q1: look.quaternion.clone(), f0: this.camera.fov, f1: c.fov, t: 0, dur };
  }

  /** Cinematic tour through the Blender camera set (T toggles). */
  tour(on = !this.tourList.length) {
    this.tourList = on ? Object.keys(this.cams).filter((k) => k !== 'TC_high_1000') : [];
    this.tourI = 0; this.tourHold = 0;
    if (on && this.tourList.length) this.flyTo(this.tourList[0], 5);
  }

  private fly(dt: number) {
    if (this.flight) {
      const F = this.flight;
      F.t = Math.min(1, F.t + dt / F.dur);
      const e = F.t * F.t * (3 - 2 * F.t);
      const p = F.p0.clone().lerp(F.p1, e);
      p.y += Math.sin(Math.PI * e) * Math.min(250, F.p0.distanceTo(F.p1) * 0.12);
      this.camera.position.copy(p);
      this.camera.quaternion.copy(F.q0).slerp(F.q1, e);
      this.camera.fov = F.f0 + (F.f1 - F.f0) * e; this.camera.updateProjectionMatrix();
      if (F.t >= 1) { this.flight = null; this.syncFly(); }
      return;
    }
    if (this.tourList.length) {
      // hold each shot with a slow push-in, then move on
      this.tourHold += dt;
      const d = new THREE.Vector3(); this.camera.getWorldDirection(d);
      this.camera.position.addScaledVector(d, dt * 4);
      if (this.tourHold > 7) { this.tourHold = 0; this.tourI = (this.tourI + 1) % this.tourList.length; this.flyTo(this.tourList[this.tourI], 7); }
      return;
    }
    const f = new THREE.Vector3(-Math.sin(this.yaw) * Math.cos(this.pitch), Math.sin(this.pitch), -Math.cos(this.yaw) * Math.cos(this.pitch));
    const r = new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    const p = this.camera.position;
    const k = (a: string, b: string) => this.keys.has(a) || this.keys.has(b);
    // speed scales with height above ground: precise near the grass, fast over the ridges
    const px0 = worldToPx(p.x, p.z);
    const agl = Math.max(2, p.y - this.w.terrain.at(Math.min(Math.max(px0.x, 0), IMG_W), Math.min(Math.max(px0.y, 0), IMG_H)));
    const speed = (this.keys.has('shift') ? 4 : 1) * this.flySpeed * THREE.MathUtils.clamp(agl * 0.9, 12, 260);
    const want = new THREE.Vector3();
    if (k('w', 'arrowup')) want.add(f);
    if (k('s', 'arrowdown')) want.sub(f);
    if (k('d', 'arrowright')) want.add(r);
    if (k('a', 'arrowleft')) want.sub(r);
    if (k('e', ' ')) want.y += 1;
    if (k('q', 'c')) want.y -= 1;
    if (want.lengthSq() > 0) want.normalize().multiplyScalar(speed);
    this.vel.lerp(want, 1 - Math.exp(-dt * 5));   // inertia: glide in and out of moves
    p.addScaledVector(this.vel, dt);
    p.y = Math.min(p.y, 4000);
    const px = worldToPx(p.x, p.z);
    const ground = this.w.terrain.at(Math.min(Math.max(px.x, 0), IMG_W), Math.min(Math.max(px.y, 0), IMG_H)) + 1.8;
    if (p.y < ground) p.y = ground;
    // look smoothing
    this.syaw += (this.yaw - this.syaw) * (1 - Math.exp(-dt * 14));
    this.spitch += (this.pitch - this.spitch) * (1 - Math.exp(-dt * 14));
    const fs = new THREE.Vector3(-Math.sin(this.syaw) * Math.cos(this.spitch), Math.sin(this.spitch), -Math.cos(this.syaw) * Math.cos(this.spitch));
    this.camera.lookAt(p.clone().add(fs));
  }
  private syaw = 0; private spitch = -0.25;

  private bindInput() {
    const dom = this.renderer.domElement;
    dom.tabIndex = 0;
    window.addEventListener('keydown', (e) => {
      if (!this.running || e.target instanceof HTMLInputElement) return;
      const key = e.key.toLowerCase();
      if (key === 't') { this.tour(); return; }
      if (this.tourList.length || this.flight) { this.tourList = []; if (this.flight) { this.flight = null; this.syncFly(); } }
      this.keys.add(key); if (e.shiftKey) this.keys.add('shift');
    });
    window.addEventListener('keyup', (e) => { this.keys.delete(e.key.toLowerCase()); if (!e.shiftKey) this.keys.delete('shift'); });
    let drag = false, down: [number, number] | null = null, last: [number, number] = [0, 0];
    dom.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; last = down; drag = false; });
    dom.addEventListener('pointermove', (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) drag = true;
      if (this.mode === 'free' && down && e.buttons) {
        if (this.tourList.length || this.flight) { this.tourList = []; this.flight = null; this.syncFly(); }
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
    this.cine?.setSize(w, h, this.renderer.getPixelRatio());
  }

  /** Render a still for QA / screenshots. */
  updateCullingPublic() { this.orbit.update(); this.updateCulling(); }
  snapshot(): string { this.updateCulling(); this.cine.prepare(this.scene); this.cine.render(); return this.renderer.domElement.toDataURL('image/png'); }
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
