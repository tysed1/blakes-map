/**
 * Canonical coordinate system (mirror of data/world/world.json + tools/lib/coords.py).
 *
 * All world data is stored in SOURCE-IMAGE PIXELS of assets/maps/source/base_map.webp
 * (2000 x 667, continuous, x right, y down). Every view converts through here only.
 *
 *   world (metres, Y-up, three.js / glTF):  X = (px - 1000) * 2.5   Z = (py - 333.5) * 2.5   Y = elevation
 *   Leaflet CRS.Simple:                      lat = -py, lng = px
 *   normalized:                              u = px / 2000, v = py / 667
 *   Blender (Z-up):                          bx = X, by = -Z, bz = Y
 */
export const IMG_W = 2000;
export const IMG_H = 667;
export const MPP = 2.5; // metres per source pixel
export const ORIGIN_PX: [number, number] = [1000, 333.5];

export interface Px { x: number; y: number }

export function pxToWorld(px: number, py: number, elev = 0): [number, number, number] {
  return [(px - ORIGIN_PX[0]) * MPP, elev, (py - ORIGIN_PX[1]) * MPP];
}

export function worldToPx(X: number, Z: number): Px {
  return { x: X / MPP + ORIGIN_PX[0], y: Z / MPP + ORIGIN_PX[1] };
}

export function pxToLatLng(px: number, py: number): [number, number] {
  return [-py, px];
}

export function latLngToPx(lat: number, lng: number): Px {
  return { x: lng, y: -lat };
}

export function pxToNorm(px: number, py: number): [number, number] {
  return [px / IMG_W, py / IMG_H];
}

export function formatPx(p: Px): string {
  const [X, , Z] = pxToWorld(p.x, p.y);
  return `px ${p.x.toFixed(1)}, ${p.y.toFixed(1)}  ·  world ${X.toFixed(0)} m E, ${(-Z).toFixed(0)} m N`;
}
