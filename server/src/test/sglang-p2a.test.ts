import assert from "assert";
import {
  DPRankState,
  DataParallelController,
} from "../sglang";

/**
 * Issue #24 验收测试 — P2a: DataParallelController — 标准 DP
 *
 * Run with:  npx ts-node src/test/sglang-p2a.test.ts
 */

let passed = 0, failed = 0;
const failures: string[] = [];

function test(name: string, fn: () => void): void {
  try { fn(); passed++; console.log("  \u2713 " + name); }
  catch (e) {
    failed++;
    const msg = e instanceof Error ? e.message : String(e);
    failures.push(name + ": " + msg);
    console.log("  \u2717 " + name + " — " + msg);
  }
}

// ==========================================
// DPRankState 测试
// ==========================================

// T1: DPRankState 初始化
test("T1 DPRankState init", () => {
  const s = new DPRankState(0, 100);
  assert.strictEqual(s.rank, 0);
  assert.strictEqual(s.pages_capacity, 100);
  assert.strictEqual(s.load, 0);
  assert.strictEqual(s.pages_allocated, 0);
});

// T2: pages_available 计算
test("T2 pages_available calculation", () => {
  const s = new DPRankState(0, 100);
  s.pages_allocated = 30;
  assert.strictEqual(s.pages_available, 70);
});

// ==========================================
// DataParallelController 基础测试
// ==========================================

// T3: dp_size=1 退化 noop
test("T3 dp_size=1 degenerate noop", () => {
  const c = new DataParallelController(1, 100, "round_robin");
  const r = c.select_rank_for_request(10);
  assert.ok(r !== null);
  assert.strictEqual(r!.rank, 0);
  assert.strictEqual(r!.pages_allocated, 10);
  assert.strictEqual(r!.load, 1);
});

// T4: dp_size=1 页不足返回 null
test("T4 dp_size=1 insufficient pages returns null", () => {
  const c = new DataParallelController(1, 5, "round_robin");
  const r = c.select_rank_for_request(10);
  assert.strictEqual(r, null);
});

// T5: pages_per_rank 均分（divEven 语义）
test("T5 pages_per_rank divEven distribution", () => {
  const c = new DataParallelController(3, 100, "round_robin");
  assert.strictEqual(c.ranks[0].pages_capacity, 34);
  assert.strictEqual(c.ranks[1].pages_capacity, 33);
  assert.strictEqual(c.ranks[2].pages_capacity, 33);
});

// T6: allocate_pages 成功
test("T6 allocate_pages success", () => {
  const c = new DataParallelController(2, 100, "round_robin");
  const ok = c.allocate_pages(0, 30);
  assert.strictEqual(ok, true);
  assert.strictEqual(c.ranks[0].pages_allocated, 30);
  assert.strictEqual(c.ranks[0].pages_available, 20); // 50 - 30
});

// T7: allocate_pages 失败
test("T7 allocate_pages failure", () => {
  const c = new DataParallelController(2, 100, "round_robin");
  const ok = c.allocate_pages(0, 60); // rank 0 capacity = 50
  assert.strictEqual(ok, false);
  assert.strictEqual(c.ranks[0].pages_allocated, 0);
});

// T8: free_pages 回写
test("T8 free_pages writeback", () => {
  const c = new DataParallelController(2, 100, "round_robin");
  c.allocate_pages(0, 30);
  c.ranks[0].load = 2;
  c.free_pages(0, 20);
  assert.strictEqual(c.ranks[0].pages_allocated, 10);
  assert.strictEqual(c.ranks[0].load, 1);
});

// T9: free_pages 越界保护
test("T9 free_pages boundary protection", () => {
  const c = new DataParallelController(2, 100, "round_robin");
  c.ranks[0].pages_allocated = 5;
  c.ranks[0].load = 1;
  c.free_pages(0, 999);
  assert.strictEqual(c.ranks[0].pages_allocated, 0);
  assert.strictEqual(c.ranks[0].load, 0);
});

// ==========================================
// round_robin 策略测试
// ==========================================

// T10: round_robin 轮询均匀
test("T10 round_robin even distribution", () => {
  const c = new DataParallelController(4, 400, "round_robin");
  const rankSeq: number[] = [];
  for (let i = 0; i < 8; i++) {
    const r = c.select_rank_for_request(1)!;
    rankSeq.push(r.rank);
  }
  assert.deepStrictEqual(rankSeq, [0, 1, 2, 3, 0, 1, 2, 3]);
});

// T11: round_robin 分配失败不影响轮询索引
test("T11 round_robin allocation failure does not affect index", () => {
  // dp_size=4, total=4 → each rank capacity=1
  const c = new DataParallelController(4, 4, "round_robin");
  // Consume rank 2's page
  c.allocate_pages(2, 1);
  // Now round_robin: idx=0→rank0, idx=1→rank1, idx=2→rank2(oom), idx=3→rank3
  c.select_rank_for_request(1); // rank 0, idx→1
  c.select_rank_for_request(1); // rank 1, idx→2
  const r2 = c.select_rank_for_request(1); // rank 2 but OOM → null, idx→3
  assert.strictEqual(r2, null);
  const r3 = c.select_rank_for_request(1); // rank 3
  assert.strictEqual(r3!.rank, 3);
});

// ==========================================
// shortest_queue 策略测试
// ==========================================

// T12: shortest_queue 选最小负载
test("T12 shortest_queue selects min load", () => {
  const c = new DataParallelController(4, 400, "shortest_queue");
  // Manually set loads
  c.ranks[0].load = 3;
  c.ranks[0].pages_allocated = 3;
  c.ranks[1].load = 1;
  c.ranks[1].pages_allocated = 1;
  c.ranks[2].load = 2;
  c.ranks[2].pages_allocated = 2;
  c.ranks[3].load = 1;
  c.ranks[3].pages_allocated = 1;
  // rank 1 and rank 3 both have load=1, should pick rank 1 (first min)
  const r = c.select_rank_for_request(1)!;
  assert.strictEqual(r.rank, 1);
  assert.strictEqual(r.load, 2);
});

// T13: shortest_queue 负载均衡
test("T13 shortest_queue load balancing", () => {
  const c = new DataParallelController(4, 400, "shortest_queue");
  // 4 requests should go to ranks 0,1,2,3 in order
  const rankSeq: number[] = [];
  for (let i = 0; i < 4; i++) {
    const r = c.select_rank_for_request(1)!;
    rankSeq.push(r.rank);
  }
  assert.deepStrictEqual(rankSeq, [0, 1, 2, 3]);
  // All ranks should have load=1
  for (let i = 0; i < 4; i++) {
    assert.strictEqual(c.ranks[i].load, 1);
  }
});

// ==========================================
// 分配失败 reject 测试
// ==========================================

// T14: 所有副本页不足返回 null
test("T14 all ranks insufficient pages returns null", () => {
  const c = new DataParallelController(2, 4, "round_robin"); // each capacity=2
  // Exhaust all pages
  c.allocate_pages(0, 2);
  c.allocate_pages(1, 2);
  const r0 = c.select_rank_for_request(1);
  const r1 = c.select_rank_for_request(1);
  assert.strictEqual(r0, null);
  assert.strictEqual(r1, null);
});

// T15: 部分副本页不足仍可分配
test("T15 partial ranks full still allocatable", () => {
  // dp_size=2, total=4 → each capacity=2
  const c = new DataParallelController(2, 4, "round_robin");
  // Exhaust rank 0's pages
  c.allocate_pages(0, 2);
  // round_robin starts at rank 0 (full) → null
  const r0 = c.select_rank_for_request(1);
  assert.strictEqual(r0, null); // rank 0 has no pages
  // Next round_robin is rank 1 → should succeed
  const r1 = c.select_rank_for_request(1);
  assert.strictEqual(r1!.rank, 1);
});

// ==========================================
// 综合测试
// ==========================================

// T16: 完整生命周期
test("T16 full lifecycle select-allocate-free", () => {
  const c = new DataParallelController(2, 100, "round_robin");
  // Select
  const r = c.select_rank_for_request(10)!;
  assert.strictEqual(r.rank, 0);
  assert.strictEqual(r.pages_allocated, 10);
  assert.strictEqual(r.load, 1);
  // Free
  c.free_pages(0, 10);
  assert.strictEqual(c.ranks[0].pages_allocated, 0);
  assert.strictEqual(c.ranks[0].load, 0);
  assert.strictEqual(c.ranks[0].pages_available, 50);
});

// T17: 大量请求压力测试
test("T17 stress test 1000 requests round_robin", () => {
  const c = new DataParallelController(8, 80000, "round_robin");
  const counts = new Array(8).fill(0);
  for (let i = 0; i < 1000; i++) {
    const r = c.select_rank_for_request(1)!;
    counts[r.rank]++;
  }
  // Each rank should get 125 requests
  for (let i = 0; i < 8; i++) {
    assert.strictEqual(counts[i], 125, `rank ${i} expected 125, got ${counts[i]}`);
  }
});

// T18: ParallelMetrics 回填验证
test("T18 ParallelMetrics backfill verification", () => {
  const c = new DataParallelController(4, 400, "shortest_queue");
  c.select_rank_for_request(10);
  c.select_rank_for_request(5);
  c.select_rank_for_request(3);

  // Simulate backfill
  const dpRankLoad = c.ranks.map(r => r.load);
  const dpAllocatePagesPerRank = c.ranks.map(r => r.pages_allocated);

  assert.strictEqual(dpRankLoad.reduce((a, b) => a + b, 0), 3);
  assert.strictEqual(dpAllocatePagesPerRank.reduce((a, b) => a + b, 0), 18);
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: dp_size=0 抛出 Error
test("B1 dp_size=0 throws Error", () => {
  assert.throws(
    () => new DataParallelController(0, 100, "round_robin"),
    /dp_size must be >= 1/
  );
});

// B2: total_num_pages=0 任何分配返回 null
test("B2 total_num_pages=0 all allocations fail", () => {
  const c = new DataParallelController(2, 0, "round_robin");
  assert.strictEqual(c.ranks[0].pages_capacity, 0);
  assert.strictEqual(c.ranks[1].pages_capacity, 0);
  const r = c.select_rank_for_request(1);
  assert.strictEqual(r, null);
});

// B3: needed_pages=0 select 成功，load+1，pages_allocated 不变
test("B3 needed_pages=0 select succeeds, load+1, pages unchanged", () => {
  const c = new DataParallelController(2, 100, "round_robin");
  const r = c.select_rank_for_request(0)!;
  assert.strictEqual(r.rank, 0);
  assert.strictEqual(r.load, 1);
  assert.strictEqual(r.pages_allocated, 0);
});

// B4: freed_pages=0 不改变 pages_allocated，但 load-1
test("B4 freed_pages=0 does not change pages_allocated, but load-1", () => {
  const c = new DataParallelController(2, 100, "round_robin");
  c.ranks[0].pages_allocated = 10;
  c.ranks[0].load = 2;
  c.free_pages(0, 0);
  assert.strictEqual(c.ranks[0].pages_allocated, 10);
  assert.strictEqual(c.ranks[0].load, 1);
});

// B5: round_robin_idx 溢出（使用 % dp_size）
test("B5 round_robin_idx overflow uses modulo", () => {
  const c = new DataParallelController(3, 30000, "round_robin");
  // Issue many requests to push idx past MAX_SAFE_INTEGER-like boundaries
  // We just verify the pattern repeats correctly after many iterations
  for (let i = 0; i < 300; i++) {
    const r = c.select_rank_for_request(1)!;
    assert.strictEqual(r.rank, i % 3);
  }
});

// B6: shortest_queue 多副本 load 相同选 index 最小
test("B6 shortest_queue same load picks lowest index", () => {
  const c = new DataParallelController(4, 400, "shortest_queue");
  // All loads are 0, should pick rank 0
  const r = c.select_rank_for_request(1)!;
  assert.strictEqual(r.rank, 0);
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P2a 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P2a 验收测试通过 \u2713");
  process.exit(0);
}
