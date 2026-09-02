// Service verification test — /api/internal/metrics endpoint
import http from "http";
import assert from "assert";
import { HttpService } from "../http/HttpService";
import { MockEngine } from "../sglang/engine";
import { DEFAULT_SIMULATOR_CONFIG } from "../sglang/types";

const PORT = 19876;

function fetchJSON(port: number, path: string): Promise<{ status: number; body: any }> {
  return new Promise((resolve, reject) => {
    http.get(`http://localhost:${port}${path}`, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode ?? 0, body: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode ?? 0, body: data });
        }
      });
    }).on("error", reject);
  });
}

async function main() {
  // 创建带并行配置的引擎
  const config = { ...DEFAULT_SIMULATOR_CONFIG, tpSize: 2, ppSize: 2, epSize: 1, cpSize: 1 };
  const engine = new MockEngine(config);
  engine.forwardBatchSeqLen(128);

  // 启动 HttpService 并注入 metrics
  const svc = new HttpService(PORT, undefined, 3001);
  svc.setSimulationMetrics(engine.metrics);
  svc.start();

  // 等待服务启动
  await new Promise((r) => setTimeout(r, 500));

  let pass = true;
  let fails: string[] = [];

  try {
    // Test 1: 未注入 metrics 时返回 503
    const svc2 = new HttpService(PORT + 1, undefined, 3001);
    svc2.start();
    await new Promise((r) => setTimeout(r, 200));
    const r0 = await fetchJSON(PORT + 1, "/api/internal/metrics");
    if (r0.status === 503) {
      console.log("✓ T1: No metrics → 503");
    } else {
      console.log(`✗ T1: Expected 503, got ${r0.status}`);
      fails.push("T1"); pass = false;
    }
    svc2.stop();

    // Test 2: 注入 metrics 后返回 200 + parallel 数据
    const r1 = await fetchJSON(PORT, "/api/internal/metrics");
    if (r1.status === 200) {
      console.log("✓ T2: Status 200");
    } else {
      console.log(`✗ T2: Expected 200, got ${r1.status}`);
      fails.push("T2"); pass = false;
    }

    // Test 3: parallel 对象存在且有并行指标
    const p = r1.body.parallel;
    const checks: [string, string, number][] = [
      ["tpCommTicks", "number", 0],
      ["ppBubbleTicks", "number", 0],
      ["ppSendRecvTicks", "number", 0],
      ["tpSize", "number", 2],
      ["ppSize", "number", 2],
      ["worldSize", "number", 4],
    ];
    for (const [key, type, minOrVal] of checks) {
      const val = p?.[key];
      if (typeof val !== type) {
        console.log(`✗ T3: parallel.${key} type=${typeof val}, expected ${type}`);
        fails.push(`T3-${key}`); pass = false;
      } else if (type === "number" && minOrVal > 0 && val !== minOrVal) {
        console.log(`✗ T3: parallel.${key}=${val}, expected ${minOrVal}`);
        fails.push(`T3-${key}`); pass = false;
      } else {
        console.log(`✓ T3: parallel.${key}=${val}`);
      }
    }

    // Test 4: parallel 对象包含全部并行维度指标字段
    const requiredKeys = [
      "tpCommTicks", "dpAttnCommTicks", "epCommTicks",
      "ppBubbleTicks", "cpCommTicks",
      "worldSize", "tpSize", "dpSize", "epSize", "ppSize", "cpSize",
    ];
    let t4ok = true;
    for (const key of requiredKeys) {
      if (!(key in p)) {
        console.log(`✗ T4: parallel.${key} missing`);
        t4ok = false;
      }
    }
    if (t4ok) {
      console.log("✓ T4: all parallel metric fields present");
    } else {
      fails.push("T4"); pass = false;
    }
  } catch (e) {
    console.log("✗ Error:", (e as Error).message);
    pass = false;
  }

  svc.stop();

  console.log(`\n=== Service Verification Result ===`);
  console.log(pass ? "ALL PASS ✓" : `FAIL: ${fails.join(", ")}`);

  process.exit(pass ? 0 : 1);
}

main();
