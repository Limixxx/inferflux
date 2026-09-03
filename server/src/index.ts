/* =====================================================================
 *  Inferflux Simulator — Main Entry Point
 *
 *  Starts services based on --mode flag:
 *    pd     : PD-Disaggregation simulator (default)
 *    sglang : SGLang parallel simulator
 *    both   : Both simulators running simultaneously
 *
 *  Usage:
 *    node dist/index.js [--http-port=8888] [--sim-port=3001] [--sglang-port=3002] [--mode=pd|sglang|both]
 *    ts-node src/index.ts [--http-port=8888] [--sim-port=3001] [--sglang-port=3002] [--mode=pd|sglang|both]
 * ===================================================================== */

import { SimService } from "./sim/SimService";
import { SgSimService } from "./sglang_service/SgSimService";
import { HttpService } from "./http/HttpService";
import { DEFAULTS } from "./shared/presets";
import { DEFAULT_SIMULATOR_CONFIG } from "./sglang";

type RunMode = "pd" | "sglang" | "both";

function parsePort(arg: string | undefined, fallback: number): number {
  if (!arg) return fallback;
  const n = parseInt(arg, 10);
  return Number.isNaN(n) ? fallback : n;
}

function main(): void {
  const args = process.argv.slice(2);
  let httpPort = 8888;
  let simPort = 3001;
  let sglangPort = 3002;
  let mode: RunMode = "pd";

  for (const a of args) {
    if (a.startsWith("--http-port=")) httpPort = parsePort(a.split("=")[1], httpPort);
    else if (a.startsWith("--sim-port=")) simPort = parsePort(a.split("=")[1], simPort);
    else if (a.startsWith("--sglang-port=")) sglangPort = parsePort(a.split("=")[1], sglangPort);
    else if (a.startsWith("--mode=")) {
      const m = a.split("=")[1];
      if (m === "pd" || m === "sglang" || m === "both") mode = m;
    }
    else if (a === "--help" || a === "-h") {
      console.log(`
Inferflux Simulator — Server

Usage:
  node dist/index.js [options]

Options:
  --http-port=<port>      HTTP static server port  (default 8888)
  --sim-port=<port>       PD-Disagg Sim API port   (default 3001)
  --sglang-port=<port>    SGLang Sim API port      (default 3002)
  --mode=<pd|sglang|both> Simulator mode            (default pd)
  --help, -h              Show this help

Modes:
  pd      PD-Disaggregation simulator (original)
  sglang  SGLang parallel simulator
  both    Both simulators running simultaneously
`);
      process.exit(0);
    }
  }

  const shutdownFns: Array<() => void> = [];

  // --- PD-Disagg mode ---
  if (mode === "pd" || mode === "both") {
    const sim = new SimService({ ...DEFAULTS }, simPort);
    sim.start();
    const http_ = new HttpService(httpPort, undefined, simPort);
    http_.start();
    shutdownFns.push(() => { sim.stop(); http_.stop(); });

    console.log(`[PD-Disagg] Sim API → http://localhost:${simPort}`);
    console.log(`[PD-Disagg] Frontend → http://localhost:${httpPort}`);
  }

  // --- SGLang mode ---
  if (mode === "sglang" || mode === "both") {
    const sgSim = new SgSimService({ ...DEFAULT_SIMULATOR_CONFIG, cacheType: "naive" }, sglangPort);
    sgSim.start();

    // For sglang mode, reuse httpPort; for both mode, use httpPort + 1
    const sgHttpPort = mode === "both" ? httpPort + 1 : httpPort;
    const sgHttp = new HttpService(sgHttpPort, undefined, sglangPort, "/sglang.html");
    sgHttp.setSimulationMetrics(sgSim.getInstance().metrics);
    sgHttp.setSGHttpApi(sgSim.getInstance().httpApi);
    sgHttp.start();

    shutdownFns.push(() => { sgSim.stop(); sgHttp.stop(); });

    console.log(`[SGLang]    Sim API → http://localhost:${sglangPort}`);
    console.log(`[SGLang]    Frontend → http://localhost:${sgHttpPort}`);
  }

  // Graceful shutdown
  const shutdown = () => {
    console.log("\n[Server] shutting down…");
    for (const fn of shutdownFns) fn();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  console.log(`
╔══════════════════════════════════════════════════════════════╗
║  Inferflux Simulator — Running (${mode} mode)                       ║
║                                                                ║
║  Press Ctrl+C to stop.                                         ║
╚══════════════════════════════════════════════════════════════╝
`);
}

main();
