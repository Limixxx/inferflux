// sglang-service.test.ts — Issue #7 验收测试：SgSimService 服务装配 + HTTP 控制端点

import http from "http";
import { SgSimService } from "../sglang_service/SgSimService";
import { DEFAULT_SIMULATOR_CONFIG } from "../sglang";

const TEST_PORT = 3099;

function httpGet(port: number, path: string): Promise<{ status: number; body: any }> {
  return new Promise((resolve) => {
    http.get(`http://localhost:${port}${path}`, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => {
        let body: any;
        try { body = JSON.parse(data); } catch { body = data; }
        resolve({ status: res.statusCode || 0, body });
      });
    }).on("error", (e) => resolve({ status: 0, body: e.message }));
  });
}

function httpPost(port: number, path: string, payload: any): Promise<{ status: number; body: any }> {
  return new Promise((resolve) => {
    const data = JSON.stringify(payload);
    const req = http.request(
      { hostname: "localhost", port, path, method: "POST", headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) } },
      (res) => {
        let body = "";
        res.on("data", (chunk) => { body += chunk; });
        res.on("end", () => {
          let parsed: any;
          try { parsed = JSON.parse(body); } catch { parsed = body; }
          resolve({ status: res.statusCode || 0, body: parsed });
        });
      },
    );
    req.on("error", (e) => resolve({ status: 0, body: e.message }));
    req.write(data);
    req.end();
  });
}

let service: SgSimService;
let passes = 0;
let failures = 0;
const results: Array<{ id: string; desc: string; pass: boolean; detail?: string }> = [];

function assert(condition: boolean, id: string, desc: string, detail?: string): void {
  if (condition) {
    passes++;
    results.push({ id, desc, pass: true });
  } else {
    failures++;
    results.push({ id, desc, pass: false, detail });
  }
}

async function runTests(): Promise<void> {
  // --- T-D1: Health endpoint ---
  {
    const r = await httpGet(TEST_PORT, "/health");
    assert(r.status === 200, "T-D1", "GET /health returns 200", `status=${r.status}`);
    assert(r.body && r.body.ok === true, "T-D1b", "GET /health body.ok === true", `body=${JSON.stringify(r.body)}`);
  }

  // --- T-D2: State endpoint ---
  {
    const r = await httpGet(TEST_PORT, "/state");
    assert(r.status === 200, "T-D2", "GET /state returns 200", `status=${r.status}`);
    assert(r.body && r.body.scheduler !== undefined, "T-D2a", "state has scheduler field", `keys=${r.body ? Object.keys(r.body).join(",") : "none"}`);
    assert(r.body && r.body.parallel !== undefined, "T-D2b", "state has parallel field", `parallel=${JSON.stringify(r.body?.parallel)}`);
    assert(r.body && r.body.metrics !== undefined, "T-D2c", "state has metrics field");
  }

  // --- T-D3: Step control ---
  {
    const before = await httpGet(TEST_PORT, "/state");
    const tickBefore = before.body?.tickCounter ?? 0;
    const r = await httpPost(TEST_PORT, "/command", { action: "step", dt: 3 });
    assert(r.status === 200, "T-D3", "POST /command step returns 200", `status=${r.status}`);
    const after = await httpGet(TEST_PORT, "/state");
    const tickAfter = after.body?.tickCounter ?? 0;
    assert(tickAfter >= tickBefore + 3, "T-D3a", "step advances tickCounter by dt", `before=${tickBefore} after=${tickAfter}`);
  }

  // --- T-D3b: Pause control ---
  {
    await httpPost(TEST_PORT, "/command", { action: "start" });
    await httpPost(TEST_PORT, "/command", { action: "pause" });
    const r1 = await httpGet(TEST_PORT, "/state");
    assert(r1.body && r1.body.paused === true, "T-D3b", "pause sets paused=true", `paused=${r1.body?.paused}`);
    // Reset
    await httpPost(TEST_PORT, "/command", { action: "reset" });
  }

  // --- T-D3c: Reset control ---
  {
    await httpPost(TEST_PORT, "/command", { action: "step", dt: 5 });
    const before = await httpGet(TEST_PORT, "/state");
    await httpPost(TEST_PORT, "/command", { action: "reset" });
    const after = await httpGet(TEST_PORT, "/state");
    assert(after.body && (after.body.tickCounter === 0), "T-D3c", "reset sets tickCounter to 0", `tick=${after.body?.tickCounter}`);
  }

  // --- T-D4: Params update ---
  {
    const r = await httpPost(TEST_PORT, "/params", { params: { tpSize: 4 } });
    assert(r.status === 200, "T-D4", "POST /params tpSize=4 returns 200", `status=${r.status} body=${JSON.stringify(r.body)?.slice(0,200)}`);
    const s = await httpGet(TEST_PORT, "/state");
    assert(s.body?.config?.tpSize === 4, "T-D4a", "after /params tpSize, config.tpSize === 4", `tpSize=${s.body?.config?.tpSize}`);
    // Reset back
    await httpPost(TEST_PORT, "/params", { params: { tpSize: 1 } });
  }

  // --- T-D5: Preset ---
  {
    const r = await httpPost(TEST_PORT, "/preset", { preset: "fullCombo" });
    assert(r.status === 200, "T-D5", "POST /preset fullCombo returns 200", `status=${r.status}`);
    const s = await httpGet(TEST_PORT, "/state");
    const c = s.body?.config;
    assert(c && c.tpSize === 4 && c.dpSize === 2 && c.epSize === 2 && c.ppSize === 2 && c.cpSize === 2,
      "T-D5a", "fullCombo sets (4,2,2,2,2)", `tp=${c?.tpSize} dp=${c?.dpSize} ep=${c?.epSize} pp=${c?.ppSize} cp=${c?.cpSize}`);
    // Reset
    await httpPost(TEST_PORT, "/preset", { preset: "single" });
  }

  // --- T-D5b: Invalid parallel config returns 400 ---
  {
    const r = await httpPost(TEST_PORT, "/params", { params: { epSize: 2 } });
    assert(r.status === 400, "T-D5b", "epSize=2 with isMoe=false returns 400", `status=${r.status}`);
  }

  // --- Boundary: single preset ---
  {
    await httpPost(TEST_PORT, "/preset", { preset: "single" });
    const s = await httpGet(TEST_PORT, "/state");
    const p = s.body?.parallel;
    assert(p && p.tpSize === 1 && p.dpSize === 1 && p.epSize === 1 && p.ppSize === 1 && p.cpSize === 1,
      "B-single", "single preset: all parallel dims = 1", `p=${JSON.stringify(p)}`);
  }

  // --- Boundary: step without start ---
  {
    await httpPost(TEST_PORT, "/command", { action: "reset" });
    const r = await httpPost(TEST_PORT, "/command", { action: "step", dt: 1 });
    assert(r.status === 200, "B-step-no-start", "step works without prior start", `status=${r.status}`);
    const s = await httpGet(TEST_PORT, "/state");
    assert((s.body?.tickCounter ?? 0) >= 1, "B-step-no-start-a", "tickCounter incremented", `tick=${s.body?.tickCounter}`);
  }
}

async function main(): Promise<void> {
  console.log("=== SgSimService 验收测试 ===\n");

  service = new SgSimService({ ...DEFAULT_SIMULATOR_CONFIG, cacheType: "naive" }, TEST_PORT);
  service.start();

  // Wait for server to be ready
  await new Promise((resolve) => setTimeout(resolve, 500));

  try {
    await runTests();
  } finally {
    service.stop();
  }

  console.log("\n--- 测试结果 ---");
  for (const r of results) {
    const icon = r.pass ? "✅" : "❌";
    const extra = r.detail ? ` (${r.detail})` : "";
    console.log(`${icon} ${r.id}: ${r.desc}${extra}`);
  }
  console.log(`\n合计: ${passes} 通过 / ${failures} 失败`);

  if (failures > 0) {
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
