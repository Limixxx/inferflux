/* =====================================================================
 *  PD-Disaggregation Simulator — Shared Type Definitions
 *  All interfaces and type aliases used across the simulation engine,
 *  HTTP service, and API layer.
 * ===================================================================== */

/* ---- Enumerations ---- */
export type LBPolicy = "least" | "round_robin" | "power_of_two" | "random";
export type DistType = "fixed" | "uniform" | "lognormal";
export type ArrivalDist = "poisson" | "uniform";

/** KVPoll states — mirrors sglang disaggregation/base/conn.py KVPoll */
export const KVPOLL = {
  Bootstrapping: "Bootstrapping",
  WaitingForInput: "WaitingForInput",
  Transferring: "Transferring",
  Success: "Success",
} as const;
export type KVPollState = (typeof KVPOLL)[keyof typeof KVPOLL];

/** Request lifecycle stages */
export type ReqStage =
  | "tokenize"
  | "p_bootstrap"
  | "p_waiting"
  | "p_prefill"
  | "p_transfer"
  | "d_waiting"
  | "d_running"
  | "response"
  | "done";

/* ---- Simulation Parameters (the P object) ---- */
export interface SimParams {
  qps: number;
  arrivalDist: ArrivalDist;
  inputLenMean: number;
  inputDist: DistType;
  outputLenMean: number;
  outputDist: DistType;
  cacheHitRate: number;
  numP: number;
  numD: number;
  lbPolicyP: LBPolicy;
  lbPolicyD: LBPolicy;
  tokenizeUsPerTok: number;
  transferOverheadMs: number;
  detokenizeMs: number;
  activeB: number;
  prefillTokPerSec: number;
  chunkSize: number;
  decodeMsBase: number;
  decodeMsPerReq: number;
  maxRunning: number;
  newTokenRatio: number;
  modelPreset: string;
  gpu: string;
  layers: number;
  fullLayers: number;
  kvHeads: number;
  headDim: number;
  dtypeBytes: number;
  hybrid: boolean;
  mla: boolean;
  kvLoraRank: number;
  qkRope: number;
  qHeads: number;
  swaWindow: number;
  kvGbP: number;
  kvGbD: number;
  bandwidthGBs: number;
}

/* ---- Model & GPU Presets ---- */
export interface ModelPreset {
  layers: number;
  fullLayers?: number;
  kvHeads: number;
  headDim: number;
  dtypeBytes: number;
  hybrid?: boolean;
  mla?: boolean;
  activeB: number;
  qHeads?: number;
  swaWindow?: number;
  kvLoraRank?: number;
  qkRope?: number;
}

export interface GpuPreset {
  hbm: number;
  tflops: number;
  bwTBs: number;
}

/* ---- Request Entity ---- */
export interface RequestStamps {
  recv: number;
  tokenized: number;
  bootstrapDone: number;
  pQueueExit: number;
  prefillDone: number;
  transferDone: number;
  preallocDone: number;
  dQueueExit: number;
  firstToken: number;
  lastToken: number;
  detokDone: number;
}

/** A single in-flight or completed simulation request. */
export interface SimRequest {
  id: number;
  room: number;
  inputLen: number;
  outputLen: number;
  cachedLen: number;
  uncachedLen: number;
  stage: ReqStage;
  kvPoll: KVPollState | null;
  p: any;    // PrefillInstance (forward ref, avoids circular import)
  d: any;    // DecodeInstance (forward ref, avoids circular import)
  readyAt: number;
  dReadyAt: number;
  dPrealloc: boolean;
  chunksTotal: number;
  chunksComputed: number;
  chunksQueued: number;
  chunksTransferred: number;
  bytesTotal: number;
  bytesDone: number;
  tokensOut: number;
  retracted: boolean;
  stamps: RequestStamps;
  lastTokenT: number;
}

/** Item on a TransferLink queue. */
export interface TransferItem {
  req: SimRequest;
  bytes: number;
  tokens: number;
  doneAtWas?: number;
}

/* ---- Metrics ---- */
export interface LatencyStats {
  avg: number;
  p50: number;
  p99: number;
}

export interface MetricSnapshot {
  n: number;
  ttft: LatencyStats;
  tpot: LatencyStats;
  e2e: LatencyStats;
  dHandshake: { avg: number; p99: number };
  bd: number[];
}

export interface Gauges {
  pQueue: number;
  dQueue: number;
  running: number;
  kvP: number;
  kvD: number;
  kvDpre: number;
  link: number;
  inflight: number;
}

/** Minimal engine interface for entity classes (avoids circular imports). */
export interface ISimEngine {
  P: SimParams;
  rng: () => number;
  now: number;
  retractTotal: number;
  allActive: Set<SimRequest>;
  responding: SimRequest[];
  sampleGauges(): Gauges;
}

/* ---- API Layer ---- */
export interface SimStateResponse {
  now: number;
  paused: boolean;
  params: SimParams;
  gauges: Gauges;
  snapshot: MetricSnapshot | null;
  breakdown: number[];
  series: Record<string, number[]>;
  retractTotal: number;
}

export interface SimCommandRequest {
  action: "start" | "pause" | "step" | "reset" | "speed";
  dt?: number;
}

export interface SimParamsRequest {
  params: Partial<SimParams>;
}

export interface SimPresetRequest {
  preset: string;
}
