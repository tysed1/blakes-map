import { IMG_W, IMG_H } from './coords';

export interface RoadType { label: string; width_m: number; lanes: number; divided: boolean; shoulder_m: number; surface: string; markings: string; speed_mph: number; material: string; color: string; z: number }
export interface Road {
  id: string; type: string; name: string | null; route: string | null; from: string; to: string;
  width_m: number; lanes: number; surface: string; material: string; oneway: boolean; grade_separated: boolean;
  settlement: string | null; source: string; length_m: number; bridge_spans: [number, number][];
  max_grade_pct?: number; virtual?: boolean; interchange?: string | null; c: number[]; [k: string]: unknown;
}
export interface Manifest {
  name: string; image: { w: number; h: number };
  terrain: { w: number; h: number; min_m: number; max_m: number; file: string; water: string };
  landuse: { id: number; name: string; color: string }[];
  road_types: Record<string, RoadType>; junction_kinds: Record<string, string>;
  trees: { file: string; count: number; stride: number };
}
export interface World {
  manifest: Manifest; roads: Road[]; nodes: any[]; bridges: any[]; rail: any[];
  water: { lines: any[]; bodies: any[] }; landuse: any[]; regions: any[]; settlements: any[]; landmarks: any[]; qa: any;
  terrain: Heightfield; waterLevel: Heightfield; landuseRaster: Uint8Array; trees: Float32Array;
}

/** Heightfield on the source-pixel grid. Sample (i,j) sits at px (i+0.5, j+0.5). */
export class Heightfield {
  constructor(public w: number, public h: number, public data: Float32Array) {}
  static fromU16(buf: ArrayBuffer, w: number, h: number, lo: number, hi: number): Heightfield {
    const u = new Uint16Array(buf);
    const f = new Float32Array(u.length);
    const k = (hi - lo) / 65535;
    for (let i = 0; i < u.length; i++) f[i] = lo + u[i] * k;
    return new Heightfield(w, h, f);
  }
  /** Bilinear sample at continuous source px. */
  at(px: number, py: number): number {
    const x = Math.min(Math.max(px - 0.5, 0), this.w - 1.001);
    const y = Math.min(Math.max(py - 0.5, 0), this.h - 1.001);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const d = this.data, w = this.w, i = y0 * w + x0;
    return d[i] * (1 - fx) * (1 - fy) + d[i + 1] * fx * (1 - fy) + d[i + w] * (1 - fx) * fy + d[i + w + 1] * fx * fy;
  }
}

const BASE = `${import.meta.env.BASE_URL}world/`;

/** World JSON; in the packed deploy the large ones ship gzipped in the same .wasm containers as bin(). */
async function json<T>(f: string): Promise<T> {
  if (PACKED) {
    packIndex ??= fetch(BASE + 'pack.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
    if ((await packIndex)?.[f]) return JSON.parse(new TextDecoder().decode(await bin(f)));
  }
  const r = await fetch(BASE + f);
  if (!r.ok) throw new Error(`failed to load ${f}`);
  return r.json();
}
/**
 * Binary data. Static hosts that only serve web media types (the Artifact deploy, see
 * tools/deploy/prepare_artifact.py) get every *.bin packed, built with VITE_PACKED_BIN=1:
 * world/pack.json maps each file to `<name>.wasm`, a valid WebAssembly module whose single custom
 * section carries the gzip of the file after a lossless reversible filter (byte-plane shuffle,
 * optionally with delta coding) that roughly halves the gzip size of heightfields and vertex buffers.
 * Hosts without a pack.json fall back to the legacy `<name>.gz.b64.txt`.
 */
const PACKED = import.meta.env.VITE_PACKED_BIN === '1';
type Filter = 'raw' | 'sh2' | 'sh4' | 'd16sh2' | 'd32sh4';
/** filter: a Filter, optionally prefixed 'c<N>:' = N 32-bit columns stored column-major */
interface PackEntry { file: string; filter: Filter | string; size: number }
let packIndex: Promise<Record<string, PackEntry> | null> | null = null;

function unshuffle(src: Uint8Array, k: number): Uint8Array {
  const n = src.length / k, out = new Uint8Array(src.length);
  for (let j = 0; j < k; j++) { const o = j * n; for (let i = 0; i < n; i++) out[i * k + j] = src[o + i]; }
  return out;
}
export function unfilter(b: Uint8Array, spec: string): ArrayBuffer {
  const m = /^c(\d+):(.*)$/.exec(spec);
  const filter = (m ? m[2] : spec) as Filter;
  const out = unfilterBytes(b, filter);
  if (!m) return out;
  // column-major -> row-major (32-bit columns)
  const n = +m[1], src = new Uint32Array(out), rows = src.length / n, dst = new Uint32Array(src.length);
  for (let c = 0; c < n; c++) { const o = c * rows; for (let r = 0; r < rows; r++) dst[r * n + c] = src[o + r]; }
  return dst.buffer;
}
function unfilterBytes(b: Uint8Array, filter: Filter): ArrayBuffer {
  if (filter === 'raw') return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  const k = filter.endsWith('sh4') ? 4 : 2;
  const u = unshuffle(b, k);
  if (filter === 'd16sh2') { const a = new Uint16Array(u.buffer); for (let i = 1; i < a.length; i++) a[i] = (a[i] + a[i - 1]) & 0xffff; }
  else if (filter === 'd32sh4') { const a = new Uint32Array(u.buffer); for (let i = 1; i < a.length; i++) a[i] = (a[i] + a[i - 1]) >>> 0; }
  return u.buffer as ArrayBuffer;
}
/** Payload of a single-custom-section wasm container (magic, version, id 0, LEB size, LEB name len, name). */
function wasmPayload(buf: ArrayBuffer): Uint8Array {
  const b = new Uint8Array(buf);
  let p = 9;
  const leb = () => { let v = 0, sh = 0, c; do { c = b[p++]; v |= (c & 0x7f) << sh; sh += 7; } while (c & 0x80); return v; };
  const size = leb(), start = p, nameLen = leb();
  p += nameLen;
  return b.subarray(p, start + size);
}
async function gunzip(data: Uint8Array): Promise<Uint8Array> {
  const ds = new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(ds).arrayBuffer());
}
export async function bin(f: string): Promise<ArrayBuffer> {
  if (!PACKED) {
    const r = await fetch(BASE + f);
    if (!r.ok) throw new Error(`failed to load ${f}`);
    return r.arrayBuffer();
  }
  packIndex ??= fetch(BASE + 'pack.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const e = (await packIndex)?.[f];
  if (e) {
    const r = await fetch(BASE + e.file);
    if (!r.ok) throw new Error(`failed to load ${e.file}`);
    return unfilter(await gunzip(wasmPayload(await r.arrayBuffer())), e.filter);
  }
  const r = await fetch(BASE + f + '.gz.b64.txt');
  if (!r.ok) throw new Error(`failed to load ${f}`);
  const b64 = (await r.text()).trim();
  return (await gunzip(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)))).buffer as ArrayBuffer;
}

export const assetUrl = (f: string) => BASE + f;

export async function loadWorld(progress: (msg: string) => void): Promise<World> {
  progress('manifest');
  const manifest = await json<Manifest>('manifest.json');
  progress('vector data');
  const [roads, nodes, bridges, rail, water, landuse, regions, settlements, landmarks, qa] = await Promise.all([
    json<Road[]>('roads.json'), json<any[]>('nodes.json'), json<any[]>('bridges.json'), json<any[]>('rail.json'),
    json<any>('water.json'), json<any[]>('landuse.json'), json<any[]>('regions.json'), json<any[]>('settlements.json'),
    json<any[]>('landmarks.json'), json<any>('qa.json'),
  ]);
  progress('terrain');
  const t = manifest.terrain;
  // (legacy trees.bin is no longer loaded: the 3D view streams the species scatter itself)
  const [tb, wb, lb] = await Promise.all([bin(t.file), bin(t.water), bin('landuse_u8.bin')]);
  const trb = new ArrayBuffer(0);
  return {
    manifest, roads, nodes, bridges, rail, water, landuse, regions, settlements, landmarks, qa,
    terrain: Heightfield.fromU16(tb, IMG_W, IMG_H, t.min_m, t.max_m),
    waterLevel: Heightfield.fromU16(wb, IMG_W, IMG_H, t.min_m, t.max_m),
    landuseRaster: new Uint8Array(lb), trees: new Float32Array(trb),
  };
}

/** Flat [x,y,z,x,y,z..] -> points */
export function xyz(c: number[]): [number, number, number][] {
  const o: [number, number, number][] = [];
  for (let i = 0; i < c.length; i += 3) o.push([c[i], c[i + 1], c[i + 2]]);
  return o;
}
export function xy(c: number[]): [number, number][] {
  const o: [number, number][] = [];
  for (let i = 0; i < c.length; i += 2) o.push([c[i], c[i + 1]]);
  return o;
}
