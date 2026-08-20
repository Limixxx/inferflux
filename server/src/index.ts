/* =====================================================================
 *  PD-Disaggregation Simulator — Main Entry Point
 *
 *  Starts two independent services:
 *    1. SimService  — simulation engine HTTP API (default :3001)
 *    2. HttpService — static file server + API proxy (default :8888)
 *
 *  Usage:
 *    node dist/index.js [--http-port=8888] [--sim-port=3001]
 *    ts-node src/index.ts [--http-port=8888] [--sim-port=3001]
 * ===================================================================== */

import { SimService } from "./sim/SimService";
import { HttpService } from "./http/HttpService";
import { DEFAULTS } from "./shared/presets";

function parsePort(arg: string | undefined, fallback: number): number {
  if (!arg) return fallback;
  const n = parseInt(arg, 10);
  return Number.isNaN(n) ? fallback : n;
}

function main(): void {
  const args = process.argv.slice(2);
  let httpPort = 8888;
  let simPort = 3001;

  for (const a of args) {
    if (a.startsWith("--http-port=")) httpPort = parsePort(a.split("=")[1], httpPort);
    else if (a.startsWith("--sim-port=")) simPort = parsePort(a.split("=")[1], simPort);
    else if (a === "--help" || a === "-h") {
      console.log(`
PD-Disaggregation Simulator — Server

Usage:
  node dist/index.js [options]

Options:
  --http-port=<port>   HTTP static server port  (default 8888)
  --sim-port=<port>    Simulation API port      (default 3001)
  --help, -h           Show this help

Services:
  HttpService  → http://localhost:${httpPort}  (serves frontend HTML)
  SimService   → http://localhost:${simPort}   (simulation engine API)

API Endpoints (via HttpService /api/* proxy or directly on SimService):
  GET  /state           Current simulation state (gauges, metrics, series)
  POST /command         { action: "start"|"pause"|"step"|"reset", dt? }
  POST /params          { params: Partial<SimParams> }
  POST /preset          { preset: string }
  GET  /health          Health check
`);
      process.exit(0);
    }
  }

  // 1. Start the Simulation Service (engine + REST API)
  const sim = new SimService({ ...DEFAULTS }, simPort);
  sim.start();

  // 2. Start the HTTP Service (static files + API proxy)
  const http_ = new HttpService(httpPort, undefined, simPort);
  http_.start();

  // Graceful shutdown
  const shutdown = () => {
    console.log("\n[Server] shutting down…");
    sim.stop();
    http_.stop();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  console.log(`
╔══════════════════════════════════════════════════════════════╗
║  PD-Disaggregation Simulator — Server Running                 ║
║                                                                ║
║  Frontend:  http://localhost:${httpPort}                          ║
║  Sim API:   http://localhost:${simPort}                          ║
║                                                                ║
║  Press Ctrl+C to stop.                                         ║
╚══════════════════════════════════════════════════════════════╝
`);
}

main();
