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

async function json<T>(f: string): Promise<T> {
  const r = await fetch(BASE + f);
  if (!r.ok) throw new Error(`failed to load ${f}`);
  return r.json();
}
/**
 * Binary rasters. Static hosts that only serve web media types (the Artifact deploy, see
 * tools/deploy/prepare_artifact.py) get them as gzip + base64 text: built with VITE_PACKED_BIN=1.
 */
const PACKED = import.meta.env.VITE_PACKED_BIN === '1';
export async function bin(f: string): Promise<ArrayBuffer> {
  const r = await fetch(BASE + f + (PACKED ? '.gz.b64.txt' : ''));
  if (!r.ok) throw new Error(`failed to load ${f}`);
  if (!PACKED) return r.arrayBuffer();
  const b64 = (await r.text()).trim();
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const ds = new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(ds).arrayBuffer();
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
  const [tb, wb, lb, trb] = await Promise.all([bin(t.file), bin(t.water), bin('landuse_u8.bin'), bin(manifest.trees.file)]);
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
