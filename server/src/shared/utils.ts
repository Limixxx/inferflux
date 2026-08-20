import { SimParams, ModelPreset } from "./types";

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : (v > hi ? hi : v);
}

export function fmtTokens(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(v % 1e6 === 0 ? 0 : 2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(v % 1e3 === 0 ? 0 : 1) + "k";
  return v.toFixed(0);
}

export function fmtMs(v: number): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  if (v >= 10000) return (v / 1000).toFixed(1) + " s";
  if (v >= 100) return v.toFixed(0) + " ms";
  return v.toFixed(1) + " ms";
}

export function fmtPct(v: number): string {
  return Number.isNaN(v) ? "—" : (v * 100).toFixed(0) + "%";
}

export function fmtNum(v: number): string {
  return Number.isNaN(v) ? "—" : (v >= 100 ? v.toFixed(0) : v.toFixed(1));
}

/** KV cell size in bytes/token — mirrors pool_configurator._compute_cell_size */
export function cellSizeOf(P: SimParams): number {
  if (P.mla) return (P.kvLoraRank + P.qkRope) * P.layers * P.dtypeBytes;
  const growLayers = P.hybrid ? P.fullLayers : P.layers;
  return growLayers * P.kvHeads * P.headDim * P.dtypeBytes * 2;
}

/** gpuKvBudget: ~70% of HBM for KV */
export function gpuKvBudget(hbm: number): number {
  return Math.round(hbm * 0.7);
}

/** Token count in a given prefill chunk */
export function chunkTokens(P: SimParams, uncachedLen: number, idx: number): number {
  return Math.min(P.chunkSize, uncachedLen - idx * P.chunkSize);
}

/** Minimal request-like shape consumed by chunkPrefillMs. */
interface ChunkPrefillRequest {
  uncachedLen: number;
  cachedLen: number;
}

/** Wall-clock ms for one prefill chunk */
export function chunkPrefillMs(P: SimParams, r: ChunkPrefillRequest, idx: number): number {
  const q = Math.min(P.chunkSize, r.uncachedLen - idx * P.chunkSize);
  const gemmMs = q / P.prefillTokPerSec * 1000;
  const effFlops = P.prefillTokPerSec * 2 * (P.activeB * 1e9);
  const fullLayers = P.hybrid ? P.fullLayers : P.layers;
  const swaLayers  = P.hybrid ? (P.layers - P.fullLayers) : 0;
  const flopsPerLayer = 4 * P.qHeads * P.headDim;
  const coefFull = flopsPerLayer * fullLayers / effFlops * 1000;
  const coefSwa  = flopsPerLayer * swaLayers  / effFlops * 1000;
  const ctx = r.cachedLen + idx * P.chunkSize + q / 2;
  const W = P.swaWindow || 0;
  const swaCtx = W > 0 ? Math.min(W, ctx) : ctx;
  return gemmMs + coefFull * q * ctx + coefSwa * q * swaCtx;
}

export const LOG_STEPS = 1000;
export const SNAP_GRID = [1, 2, 3, 5, 7, 10];

export function logSliderToVal(def: { min: number; max: number }, pos: number): number {
  if (pos <= 0) return def.min;
  if (pos >= LOG_STEPS) return def.max;
  const lo = Math.log(def.min), hi = Math.log(def.max);
  const raw = Math.exp(lo + (hi - lo) * (pos / LOG_STEPS));
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const base = raw / mag;
  let snap = SNAP_GRID[0];
  for (const g of SNAP_GRID)
    if (Math.abs(Math.log(base / g)) < Math.abs(Math.log(base / snap))) snap = g;
  return Math.max(def.min, Math.min(def.max, Math.round(snap * mag)));
}

export function nearestStepIdx(def: { steps: number[] }, val: number): number {
  let idx = 0, best = Infinity;
  for (let i = 0; i < def.steps.length; i++){
    const d = Math.abs(def.steps[i] - val);
    if (d < best){ best = d; idx = i; }
  }
  return idx;
}

export function logValToSlider(def: { min: number; max: number }, val: number): number {
  const lo = Math.log(def.min), hi = Math.log(def.max);
  return Math.max(0, Math.min(LOG_STEPS,
    Math.round((Math.log(val) - lo) / (hi - lo) * LOG_STEPS)));
}
