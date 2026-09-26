/** Tiny event bus shared by the 2D and 3D views (selection + focus sync). */
export type Selection = { kind: 'road' | 'node' | 'bridge' | 'rail' | 'water' | 'landuse' | 'region' | 'settlement' | 'landmark' | 'qa'; id: string; data: any; at?: { x: number; y: number } } | null;

type Handler<T> = (v: T) => void;
class Topic<T> {
  private hs: Handler<T>[] = [];
  on(h: Handler<T>) { this.hs.push(h); }
  emit(v: T) { for (const h of this.hs) h(v); }
}
export const bus = {
  select: new Topic<Selection>(),
  /** focus point in source px (+ optional zoom radius in px) */
  focus: new Topic<{ x: number; y: number; r?: number }>(),
  cursor: new Topic<{ x: number; y: number; z?: number } | null>(),
};
