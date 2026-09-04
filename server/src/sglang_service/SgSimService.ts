import http from "http";
import {
  createSimulator,
  DEFAULT_SIMULATOR_CONFIG,
  type SimulatorConfig,
  type SgSimInstance,
  type WorkloadConfig,
} from "../sglang";
import { validateParallelConfig } from "../sglang/parallel";

// ===== Preset configs for sglang demo =====
const SGLANG_PRESETS: Record<string, Partial<SimulatorConfig>> = {
  single: {
    tpSize: 1, dpSize: 1, epSize: 1, ppSize: 1, cpSize: 1,
  },
  tpOnly: {
    tpSize: 4, dpSize: 1, epSize: 1, ppSize: 1, cpSize: 1,
  },
  moeEP: {
    tpSize: 1, dpSize: 1, epSize: 4, ppSize: 1, cpSize: 1,
    modelConfig: { ...DEFAULT_SIMULATOR_CONFIG.modelConfig, isMoe: true, numExperts: 8, moeTopK: 2, moeIntermediateSize: 1408 },
    enableEplb: true,
    moeRoutingMode: "hash",
  },
  pp1f1b: {
    tpSize: 1, dpSize: 1, epSize: 1, ppSize: 4, cpSize: 1,
    ppPipelineSchedule: "1f1b",
    ppNumMicroBatches: 8,
  },
  cpLongSeq: {
    tpSize: 1, dpSize: 1, epSize: 1, ppSize: 1, cpSize: 2,
    maxSeqLen: 32768,
  },
  fullCombo: {
    tpSize: 4, dpSize: 2, epSize: 2, ppSize: 2, cpSize: 2,
    modelConfig: { ...DEFAULT_SIMULATOR_CONFIG.modelConfig, isMoe: true, numExperts: 8, moeTopK: 2, moeIntermediateSize: 1408 },
    enableEplb: true,
    moeRoutingMode: "hash",
    ppPipelineSchedule: "1f1b",
    ppNumMicroBatches: 8,
  },
};

/** Build a default workload config for the sglang demo. */
function defaultWorkloadConfig(overrides?: Partial<WorkloadConfig>): WorkloadConfig {
  return {
    numRequests: 200,
    arrivalDistribution: "poisson",
    arrivalRate: 10.0,
    inputLenDistribution: "uniform",
    inputLenMin: 128,
    inputLenMax: 1024,
    outputLenDistribution: "uniform",
    outputLenMin: 100,
    outputLenMax: 1024,
    sharedPrefixRatio: 0.3,
    sharedPrefixLen: 100,
    ...overrides,
  };
}

/**
 * SgSimService — HTTP server exposing the SGLang simulator control API.
 *
 * Endpoints:
 *   GET  /state        → scheduler snapshot + metrics + parallel summary
 *   POST /command       → { action: "start"|"pause"|"step"|"reset", dt? }
 *   POST /params        → { params: Partial<SimulatorConfig> } (rebuilds instance)
 *   POST /preset        → { preset: string }
 *   GET  /health        → { ok: true }
 */
export class SgSimService {
  private instance: SgSimInstance;
  private config: SimulatorConfig;
  private server: http.Server;
  private paused = false;
  private loopTimer: ReturnType<typeof setInterval> | null = null;
  private readonly port: number;
  private _tickCounter = 0;

  constructor(config: SimulatorConfig = { ...DEFAULT_SIMULATOR_CONFIG, cacheType: "naive" }, port = 3002) {
    this.config = { ...config, cacheType: config.cacheType ?? "naive" };
    this.instance = createSimulator(this.config);
    this.port = port;
    this.server = http.createServer((req, res) => this.handleRequest(req, res));
  }

  /** Start the SgSimService HTTP server. */
  start(): void {
    this.server.listen(this.port, () => {
      console.log(`[SgSimService] listening on http://localhost:${this.port}`);
    });
  }

  /** Stop the server and sim loop. */
  stop(): void {
    if (this.loopTimer) { clearInterval(this.loopTimer); this.loopTimer = null; }
    this.instance.shutdown();
    this.server.close();
    console.log("[SgSimService] stopped");
  }

  /** Get the current SimulatorConfig. */
  getConfig(): SimulatorConfig {
    return this.config;
  }

  /** Get the current SgSimInstance. */
  getInstance(): SgSimInstance {
    return this.instance;
  }

  private rebuildInstance(): void {
    this.instance.shutdown();
    if (this.loopTimer) { clearInterval(this.loopTimer); this.loopTimer = null; }
    this.instance = createSimulator(this.config);
    this._tickCounter = 0;
    this.paused = false;
    // Re-apply workload
    this.instance.loadWorkload(defaultWorkloadConfig());
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

    if (req.method === "POST" && path === "/command") {
      this.readBody(req, (body: { action: string; dt?: number }) => {
        this.handleCommand(body);
        this.sendJson(res, { ok: true, ...this.getState() });
      });
      return;
    }

    if (req.method === "POST" && path === "/params") {
      this.readBody(req, (body: { params: Partial<SimulatorConfig> }) => {
        try {
          this.handleParams(body);
          this.sendJson(res, { ok: true, ...this.getState() });
        } catch (e: any) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: e.message || String(e) }));
        }
      });
      return;
    }

    if (req.method === "POST" && path === "/preset") {
      this.readBody(req, (body: { preset: string }) => {
        this.handlePreset(body);
        this.sendJson(res, { ok: true, ...this.getState() });
      });
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not Found" }));
  }

  /** Build the state response. */
  getState(): Record<string, unknown> {
    const scheduler = this.instance.scheduler;
    return {
      tickCounter: this._tickCounter,
      paused: this.paused,
      config: this.config,
      scheduler: {
        pendingReqs: scheduler.prefillManager.pendingList.length,
        runningReqs: scheduler.decodeManager.runningReqs.size,
        availableTableIndices: scheduler.tableManager.availableSize,
        tickCounter: scheduler.tickCounter,
        globalStep: scheduler.globalStep,
        cacheSizeInfo: scheduler.cacheManager.availableSize,
      },
      metrics: this.instance.getMetrics(),
      parallel: this.instance.metrics.parallel.summary(),
    };
  }

  private handleCommand(cmd: { action: string; dt?: number }): void {
    switch (cmd.action) {
      case "start": {
        this.paused = false;
        if (!this.loopTimer) {
          const intervalMs = this.config.tickIntervalMs || 10;
          this.loopTimer = setInterval(() => {
            if (!this.paused) {
              this.instance.scheduler.runTick([]);
              this._tickCounter++;
            }
          }, intervalMs);
        }
        break;
      }
      case "pause":
        this.paused = true;
        break;
      case "step": {
        const steps = cmd.dt ?? 1;
        for (let i = 0; i < steps; i++) {
          this.instance.scheduler.runTick([]);
          this._tickCounter++;
        }
        break;
      }
      case "reset":
        this.rebuildInstance();
        break;
    }
  }

  private handleParams(body: { params: Partial<SimulatorConfig> }): void {
    const newConfig = { ...this.config, ...body.params };
    // Merge modelConfig if provided
    if (body.params.modelConfig) {
      newConfig.modelConfig = { ...this.config.modelConfig, ...body.params.modelConfig };
    }
    // Validate parallel config before applying
    const validation = validateParallelConfig(newConfig, newConfig.modelConfig);
    if (!validation.ok) {
      throw new Error(`Invalid parallel config: ${validation.errors.join("; ")}`);
    }
    this.config = newConfig;
    this.rebuildInstance();
  }

  private handlePreset(body: { preset: string }): void {
    const preset = SGLANG_PRESETS[body.preset];
    if (preset) {
      const newConfig = { ...DEFAULT_SIMULATOR_CONFIG, ...preset };
      if (preset.modelConfig) {
        newConfig.modelConfig = { ...DEFAULT_SIMULATOR_CONFIG.modelConfig, ...preset.modelConfig };
      }
      this.config = newConfig;
      this.rebuildInstance();
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
