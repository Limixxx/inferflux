import http from "http";
import fs from "fs";
import path from "path";

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

  constructor(port = 8888, rootDir?: string, simPort = 3001) {
    this.port = port;
    // Default root is server/public/ (2 levels up from dist/http/ or src/http/, then "public")
    this.rootDir = rootDir || path.resolve(__dirname, "..", "..", "public");
    this.simPort = simPort;
    this.server = http.createServer((req, res) => this.handleRequest(req, res));
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

    // Proxy /api/* to the SimService
    if (urlPath.startsWith("/api/")) {
      this.proxyToSim(urlPath.replace("/api", ""), req, res);
      return;
    }

    // Normalize: strip trailing slash, default to the app entry (pd-disagg.html)
    if (urlPath === "/" || urlPath === "") urlPath = "/pd-disagg.html";

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
