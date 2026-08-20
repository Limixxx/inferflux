import http from "http";
import { SimParams, SimStateResponse, SimCommandRequest, SimParamsRequest, SimPresetRequest, SimRequest } from "../shared/types";
import { DEFAULTS, PRESETS, MODEL_PRESETS, GPU_PRESETS } from "../shared/presets";
import { gpuKvBudget } from "../shared/utils";
import { SimEngine } from "./SimEngine";

/** Serialize a request for the /render endpoint. */
function serializeReq(r: SimRequest, pIdx: number, dIdx: number): any {
  return {
    id: r.id, stage: r.stage, kvPoll: r.kvPoll,
    readyAt: r.readyAt, dReadyAt: r.dReadyAt, dPrealloc: r.dPrealloc,
    inputLen: r.inputLen, outputLen: r.outputLen, cachedLen: r.cachedLen,
    uncachedLen: r.uncachedLen, tokensOut: r.tokensOut, retracted: r.retracted,
    chunksTotal: r.chunksTotal, chunksComputed: r.chunksComputed,
    chunksQueued: r.chunksQueued, chunksTransferred: r.chunksTransferred,
    bytesTotal: r.bytesTotal, bytesDone: r.bytesDone,
    stamps: { ...r.stamps },
    pIdx, dIdx,
  };
}

const METRIC_WINDOW_MS = 10000;
const SERIES_KEYS = ["ttft","tpot","e2e","rps","tps","pQueue","dQueue","running","kvP","kvD","kvDpre","dHandshake","link","inflight"];

/**
 * SimService — standalone HTTP server exposing the simulation engine API.
 *
 * Endpoints:
 *   GET  /state        → SimStateResponse (current gauges, metrics, series)
 *   POST /command       → { action: "start"|"pause"|"step"|"reset", dt? }
 *   POST /params        → { params: Partial<SimParams> }
 *   POST /preset        → { preset: string }
 *   GET  /health        → { ok: true }
 */
export class SimService {
  private engine: SimEngine;
  private server: http.Server;
  private paused = false;
  private timeScale = 1;
  private loopTimer: ReturnType<typeof setInterval> | null = null;
  private readonly port: number;

  constructor(params: SimParams = { ...DEFAULTS }, port = 3001) {
    this.engine = new SimEngine(params);
    this.port = port;
    this.server = http.createServer((req, res) => this.handleRequest(req, res));
  }

  /** Start the simulation HTTP server and the sim loop. */
  start(): void {
    this.server.listen(this.port, () => {
      console.log(`[SimService] listening on http://localhost:${this.port}`);
    });
    this.startLoop();
  }

  /** Stop the server and sim loop. */
  stop(): void {
    if (this.loopTimer) { clearInterval(this.loopTimer); this.loopTimer = null; }
    this.server.close();
    console.log("[SimService] stopped");
  }

  private startLoop(): void {
    const FRAME_MS = 16;
    this.loopTimer = setInterval(() => {
      if (!this.paused) {
        this.engine.advance(FRAME_MS * this.timeScale);
      }
    }, FRAME_MS);
  }

  private handleRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

    const url = new URL(req.url || "/", `http://localhost:${this.port}`);
    const path = url.pathname;

    if (req.method === "GET" && path === "/health") {
      this.sendJson(res, { ok: true });
      return;
    }

    if (req.method === "GET" && path === "/state") {
      this.sendJson(res, this.getState());
      return;
    }

    if (req.method === "GET" && path === "/render") {
      this.sendJson(res, this.getRenderState());
      return;
    }

    if (req.method === "POST" && path === "/command") {
      this.readBody(req, (body: SimCommandRequest) => {
        this.handleCommand(body);
        this.sendJson(res, { ok: true, ...this.getState() });
      });
      return;
    }

    if (req.method === "POST" && path === "/params") {
      this.readBody(req, (body: SimParamsRequest) => {
        this.handleParams(body);
        this.sendJson(res, { ok: true, ...this.getState() });
      });
      return;
    }

    if (req.method === "POST" && path === "/preset") {
      this.readBody(req, (body: SimPresetRequest) => {
        this.handlePreset(body);
        this.sendJson(res, { ok: true, ...this.getState() });
      });
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not Found" }));
  }

  /** Build the full state response for API consumers. */
  getState(): SimStateResponse {
    const snap = this.engine.metrics.snapshot(this.engine.now, METRIC_WINDOW_MS);
    const breakdown = this.engine.metrics.recentBreakdown(10);
    const series: Record<string, number[]> = {};
    for (const k of SERIES_KEYS) {
      series[k] = this.engine.metrics.latestSeries(k);
    }
    return {
      now: this.engine.now,
      paused: this.paused,
      params: this.engine.P,
      gauges: this.engine.sampleGauges(),
      snapshot: snap,
      breakdown,
      series,
      retractTotal: this.engine.retractTotal,
    };
  }

  /** Build the full render state for the frontend Renderer (entity-level). */
  getRenderState(): any {
    const eng = this.engine;
    const pIdx = (p: any) => eng.pList.indexOf(p);
    const dIdx = (d: any) => eng.dList.indexOf(d);
    const sreq = (r: SimRequest) => serializeReq(r, r.p ? pIdx(r.p) : -1, r.d ? dIdx(r.d) : -1);

    return {
      now: eng.now,
      P: eng.P,
      retractTotal: eng.retractTotal,
      pList: eng.pList.map((p: any) => ({
        id: p.id, kvUsed: p.kvUsed, draining: p.draining,
        maxTokens: p.maxTokens(eng.P),
        bootstrapQ: p.bootstrapQ.map(sreq),
        waitingQ: p.waitingQ.map(sreq),
        slots: p.slots.map((s: any) => ({ req: sreq(s.req), busyUntil: s.busyUntil })),
        inflight: p.inflight.map(sreq),
        link: {
          queue: p.link.queue.map((it: any) => ({ bytes: it.bytes, tokens: it.tokens, doneAtWas: it.doneAtWas })),
          current: p.link.current ? { bytes: p.link.current.bytes, tokens: p.link.current.tokens, doneAtWas: p.link.current.doneAtWas, req: sreq(p.link.current.req) } : null,
          startAt: p.link.startAt, doneAt: p.link.doneAt, depth: p.link.depth,
        },
      })),
      dList: eng.dList.map((d: any) => ({
        id: d.id, kvUsed: d.kvUsed, draining: d.draining,
        maxTokens: d.maxTokens(eng.P), ntr: d.ntr, retractGlow: d.retractGlow,
        preallocQ: d.preallocQ.map(sreq),
        transferQ: d.transferQ.map(sreq),
        waitingQ: d.waitingQ.map(sreq),
        running: d.running.map(sreq),
      })),
      allActive: Array.from(eng.allActive).map(sreq),
      responding: eng.responding.map(sreq),
    };
  }

  private handleCommand(cmd: SimCommandRequest): void {
    switch (cmd.action) {
      case "start":
        this.paused = false;
        break;
      case "pause":
        this.paused = true;
        break;
      case "step":
        this.engine.advance(cmd.dt ?? 100);
        break;
      case "reset":
        this.engine.reset();
        break;
      case "speed":
        this.timeScale = cmd.dt ?? 1;
        break;
    }
  }

  private handleParams(body: SimParamsRequest): void {
    Object.assign(this.engine.P, body.params);
    // Don't reset — let syncTopology() handle topology on next step (matches original)
  }

  private handlePreset(body: SimPresetRequest): void {
    const preset = PRESETS[body.preset];
    if (preset) {
      Object.assign(this.engine.P, preset);
      this.applyModelPreset(this.engine.P.modelPreset);
      this.engine.reset();
    }
  }

  /** Sync model-derived fields (activeB, qHeads, swaWindow, …) to the selected preset. */
  private applyModelPreset(name: string): void {
    const P = this.engine.P;
    if (name !== "custom" && !MODEL_PRESETS[name]) name = DEFAULTS.modelPreset;
    P.modelPreset = name;
    if (name !== "custom") {
      const m = MODEL_PRESETS[name];
      P.layers = m.layers; P.kvHeads = m.kvHeads; P.headDim = m.headDim;
      P.dtypeBytes = m.dtypeBytes; P.mla = !!m.mla;
      P.hybrid = !!m.hybrid; P.fullLayers = m.fullLayers || m.layers;
      P.activeB = m.activeB; P.qHeads = m.qHeads || 32;
      P.swaWindow = m.hybrid ? (m.swaWindow || 0) : 0;
      if (m.mla) { P.kvLoraRank = m.kvLoraRank!; P.qkRope = m.qkRope!; }
    } else {
      P.mla = false; P.hybrid = false; P.swaWindow = 0;
    }
  }

  private sendJson(res: http.ServerResponse, data: unknown): void {
    const json = JSON.stringify(data, (_k, v) =>
      typeof v === "number" && Number.isNaN(v) ? null : v);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(json);
  }

  private readBody<T>(req: http.IncomingMessage, cb: (body: T) => void): void {
    let raw = "";
    req.on("data", (chunk) => { raw += chunk; });
    req.on("end", () => {
      try { cb(JSON.parse(raw) as T); }
      catch { cb({} as T); }
    });
  }
}
