import http from "http";
import fs from "fs";
import path from "path";
import type { SimulationMetrics } from "../sglang/metrics";
import type { SGHttpApi } from "../sglang/api";

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

/**
 * HttpService — standalone HTTP server for serving static frontend files
 * and proxying /api/* requests to the SimService.
 *
 * Routes:
 *   GET  /                → pd-disagg.html (app entry)
 *   GET  /<file>          → static file from server/public/
 *   GET  /api/<path>      → proxied to SimService (localhost:simPort)
 */
export class HttpService {
  private server: http.Server;
  private readonly rootDir: string;
  private readonly port: number;
  private readonly simPort: number;
  private _simulationMetrics: SimulationMetrics | null = null;
  private _sgHttpApi: SGHttpApi | null = null;

  private readonly defaultHtml: string;

  constructor(port = 8888, rootDir?: string, simPort = 3001, defaultHtml = "/pd-disagg.html") {
    this.port = port;
    // Default root is server/public/ (2 levels up from dist/http/ or src/http/, then "public")
    this.rootDir = rootDir || path.resolve(__dirname, "..", "..", "public");
    this.simPort = simPort;
    this.defaultHtml = defaultHtml;
    this.server = http.createServer((req, res) => this.handleRequest(req, res));
  }

  /** 注入 SimulationMetrics 实例，用于 /internal/metrics 端点 */
  setSimulationMetrics(metrics: SimulationMetrics): void {
    this._simulationMetrics = metrics;
  }

  /** 注入 SGHttpApi 实例，用于 /v1/* 端点 */
  setSGHttpApi(api: SGHttpApi): void {
    this._sgHttpApi = api;
  }

  /** Start the HTTP server. */
  start(): void {
    this.server.listen(this.port, () => {
      console.log(`[HttpService] serving ${this.rootDir} on http://localhost:${this.port}`);
      console.log(`[HttpService] API proxy → http://localhost:${this.simPort}`);
    });
  }

  /** Stop the server. */
  stop(): void {
    this.server.close();
    console.log("[HttpService] stopped");
  }

  private handleRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
    const url = new URL(req.url || "/", `http://localhost:${this.port}`);
    let urlPath = decodeURIComponent(url.pathname);

    // S6: /v1/* 路由
    if (urlPath.startsWith("/v1/")) {
      // POST /v1/chat/completions — 代理到 SimService
      if (req.method === "POST" && urlPath === "/v1/chat/completions") {
        this.proxyToSim("/v1/chat/completions", req, res);
        return;
      }

      // GET /v1/internal/metrics — 直接读取 metrics
      if (req.method === "GET" && urlPath === "/v1/internal/metrics") {
        if (this._sgHttpApi) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(this._sgHttpApi.handleInternalMetrics()));
        } else {
          res.writeHead(503, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "SGHttpApi not available" }));
        }
        return;
      }

      // GET /v1/internal/state — 代理到 SimService
      if (req.method === "GET" && urlPath === "/v1/internal/state") {
        this.proxyToSim("/v1/internal/state", req, res);
        return;
      }
    }

    // Proxy /api/* to the SimService
    if (urlPath.startsWith("/api/")) {
      // P6: /api/internal/metrics 端点 — 直接返回 SimulationMetrics
      if (urlPath === "/api/internal/metrics") {
        if (this._simulationMetrics) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(this._simulationMetrics.toJSON()));
        } else {
          res.writeHead(503, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "SimulationMetrics not available" }));
        }
        return;
      }
      this.proxyToSim(urlPath.replace("/api", ""), req, res);
      return;
    }

    // Normalize: strip trailing slash, default to the app entry
    if (urlPath === "/" || urlPath === "") urlPath = this.defaultHtml;

    const filePath = path.join(this.rootDir, urlPath);

    // Prevent directory traversal
    if (!filePath.startsWith(this.rootDir)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }

    fs.stat(filePath, (err, stats) => {
      if (err || !stats.isFile()) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("404 Not Found");
        return;
      }
      const ext = path.extname(filePath).toLowerCase();
      const mime = MIME_TYPES[ext] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": mime });
      fs.createReadStream(filePath).pipe(res);
    });
  }

  /** Proxy a request to the SimService HTTP server. */
  private proxyToSim(
    simPath: string,
    req: http.IncomingMessage,
    res: http.ServerResponse,
  ): void {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const body = Buffer.concat(chunks);
      const proxyReq = http.request(
        {
          hostname: "localhost",
          port: this.simPort,
          path: simPath,
          method: req.method || "GET",
          headers: {
            "Content-Type": req.headers["content-type"] || "application/json",
            "Content-Length": body.length,
          },
        },
        (proxyRes) => {
          res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
          proxyRes.pipe(res);
        },
      );
      proxyReq.on("error", () => {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "SimService unavailable" }));
      });
      if (body.length) proxyReq.write(body);
      proxyReq.end();
    });
  }
}
