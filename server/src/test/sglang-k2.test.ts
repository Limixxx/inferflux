import assert from "assert";
import {
  DEFAULT_MODEL_CONFIG,
  DEFAULT_SIMULATOR_CONFIG,
  calculateMemoryBudget,
  CacheSizeInfo,
  BaseCacheHandle,
  BaseKVCachePool,
  BasePrefixCache,
  MatchResult,
  InsertResult,
  MockKVCachePool,
  PageAllocation,
  NaivePrefixCache,
  NaiveCacheHandle,
} from "../sglang";

/**
 * Issue #13 验收测试 — K2: MockKVCachePool + NaivePrefixCache (§3.4.2 / §9.3b)
 *
 * Run with:  npx ts-node src/test/sglang-k2.test.ts
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

// Helper: minimal ModelConfig for testing
const testModelConfig = {
  ...DEFAULT_MODEL_CONFIG,
  numLayers: 32,
  hiddenSize: 4096,
  numKvHeads: 8,
  headDim: 128,
};

// ===== T1: MockKVCachePool 构造 =====
test("T1 MockKVCachePool construction", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  assert.strictEqual(pool.numPages, 100);
  assert.strictEqual(pool.pageSize, 16);
  assert.strictEqual(pool.totalCapacity, 1600);
  assert.strictEqual(pool.usedCapacity, 0);
  assert.strictEqual(pool.usedPages, 0);
  assert.strictEqual(pool.freePages, 100);
  assert.strictEqual(pool.freePagesPool.length, 100);
  // freePagesPool should be [0, 16, 32, ..., 99*16]
  assert.strictEqual(pool.freePagesPool[0], 0);
  assert.strictEqual(pool.freePagesPool[1], 16);
  assert.strictEqual(pool.freePagesPool[99], 99 * 16);
});

// ===== T2: MockKVCachePool.cachePerPage 公式 =====
test("T2 MockKVCachePool.cachePerPage formula", () => {
  const config = { ...DEFAULT_SIMULATOR_CONFIG, dtypeSize: 2, tpSize: 1 };
  const pool = new MockKVCachePool(testModelConfig, 100, 16, config);
  // Verify against calculateMemoryBudget's formula
  const budget = calculateMemoryBudget(config, testModelConfig, 80 * 1024 ** 3);
  // Both use the same formula: 2 * headDim * kvHeadsPerGpu * pageSize * dtypeSize * numLayers
  // calculateMemoryBudget uses config.pageSize, pool uses its own pageSize param
  // They should match when pageSize is the same
  const expectedCachePerPage =
    2 * testModelConfig.headDim * testModelConfig.numKvHeads * 16 * 2 * testModelConfig.numLayers;
  assert.strictEqual(pool.cachePerPage, expectedCachePerPage);
});

// ===== T3: MockKVCachePool.storeKV noop =====
test("T3 MockKVCachePool.storeKV noop", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  const beforeUsed = pool.usedPages;
  const beforeFree = pool.freePages;
  pool.storeKV([1, 2, 3], [4, 5, 6], [0, 1, 2], 0);
  assert.strictEqual(pool.usedPages, beforeUsed);
  assert.strictEqual(pool.freePages, beforeFree);
});

// ===== T4: MockKVCachePool.allocatePaged 基本分配 =====
test("T4 MockKVCachePool.allocatePaged basic allocation", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  const alloc = pool.allocatePaged(1);
  assert.ok(alloc instanceof PageAllocation);
  assert.strictEqual(alloc.pages.length, 1);
  assert.strictEqual(alloc.slots.length, 16);
  assert.strictEqual(alloc.slotCount, 16);
  // Slots should be contiguous within the page
  const pageStart = alloc.pages[0];
  assert.strictEqual(alloc.slots[0], pageStart);
  assert.strictEqual(alloc.slots[15], pageStart + 15);
});

// ===== T5: MockKVCachePool.allocatePaged 多页分配 =====
test("T5 MockKVCachePool.allocatePaged multi-page allocation", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  const alloc = pool.allocatePaged(3);
  assert.strictEqual(alloc.pages.length, 3);
  assert.strictEqual(alloc.slotCount, 48);  // 3 * 16
  assert.strictEqual(pool.usedPages, 3);
  assert.strictEqual(pool.freePages, 97);
});

// ===== T6: MockKVCachePool.allocatePaged 不足抛错 =====
test("T6 MockKVCachePool.allocatePaged insufficient pages throws Error", () => {
  const pool = new MockKVCachePool(testModelConfig, 5, 16);
  assert.throws(
    () => pool.allocatePaged(6),
    /allocatePaged failed/
  );
});

// ===== T7: MockKVCachePool.deallocatePageAllocation =====
test("T7 MockKVCachePool.deallocatePageAllocation", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  const alloc = pool.allocatePaged(3);
  assert.strictEqual(pool.usedPages, 3);
  assert.strictEqual(pool.freePages, 97);
  pool.deallocatePageAllocation(alloc);
  assert.strictEqual(pool.usedPages, 0);
  assert.strictEqual(pool.freePages, 100);
});

// ===== T8: MockKVCachePool allocate-deallocate 循环 =====
test("T8 MockKVCachePool allocate-deallocate cycle", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  const alloc1 = pool.allocatePaged(10);
  assert.strictEqual(pool.usedPages, 10);
  pool.deallocatePageAllocation(alloc1);
  assert.strictEqual(pool.usedPages, 0);
  const alloc2 = pool.allocatePaged(5);
  assert.strictEqual(pool.usedPages, 5);
  assert.strictEqual(pool.freePages, 95);
  pool.deallocatePageAllocation(alloc2);
  assert.strictEqual(pool.usedPages, 0);
  assert.strictEqual(pool.freePages, 100);
});

// ===== T9: MockKVCachePool 页数守恒 =====
test("T9 MockKVCachePool page count invariant", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  const invariant = () => pool.usedPages + pool.freePages === pool.numPages;
  assert.ok(invariant());
  const a1 = pool.allocatePaged(10);
  assert.ok(invariant());
  const a2 = pool.allocatePaged(20);
  assert.ok(invariant());
  pool.deallocatePageAllocation(a1);
  assert.ok(invariant());
  pool.deallocatePageAllocation(a2);
  assert.ok(invariant());
});

// ===== T10: MockKVCachePool.decodeStepLatency =====
test("T10 MockKVCachePool.decodeStepLatency", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  assert.strictEqual(pool.decodeStepLatency(5, 2, 3), 13);  // 5*2 + 3
  assert.strictEqual(pool.decodeStepLatency(10, 1, 0), 10);  // 10*1 + 0
  assert.strictEqual(pool.decodeStepLatency(0, 1, 5), 5);    // 0*1 + 5
});

// ===== T11: MockKVCachePool.usedCapacity/totalCapacity =====
test("T11 MockKVCachePool.usedCapacity/totalCapacity", () => {
  const pool = new MockKVCachePool(testModelConfig, 50, 16);
  assert.strictEqual(pool.totalCapacity, 800);  // 50 * 16
  assert.strictEqual(pool.usedCapacity, 0);
  const alloc = pool.allocatePaged(3);
  assert.strictEqual(pool.usedCapacity, 48);  // 3 * 16
  pool.deallocatePageAllocation(alloc);
  assert.strictEqual(pool.usedCapacity, 0);
});

// ===== T12: NaiveCacheHandle 构造 =====
test("T12 NaiveCacheHandle construction", () => {
  const handle = new NaiveCacheHandle();
  assert.strictEqual(handle.cachedLen, 0);
  assert.deepStrictEqual(handle.getMatchedIndices(), []);
});

// ===== T13: NaivePrefixCache.matchPrefix 全 miss =====
test("T13 NaivePrefixCache.matchPrefix always miss", () => {
  const cache = new NaivePrefixCache(100, 16);
  const result = cache.matchPrefix([1, 2, 3, 4, 5]);
  assert.ok(result instanceof MatchResult);
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
  assert.ok(result.cudaHandle instanceof NaiveCacheHandle);
});

// ===== T14: NaivePrefixCache.insertPrefix =====
test("T14 NaivePrefixCache.insertPrefix", () => {
  const cache = new NaivePrefixCache(100, 16);
  const result = cache.insertPrefix([1, 2, 3], [0, 1, 2]);
  assert.ok(result instanceof InsertResult);
  assert.strictEqual(result.cachedLen, 0);
  assert.ok(result.cudaHandle instanceof NaiveCacheHandle);
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
});

// ===== T15: NaivePrefixCache.lockHandle noop =====
test("T15 NaivePrefixCache.lockHandle noop", () => {
  const cache = new NaivePrefixCache(100, 16);
  const handle = new NaiveCacheHandle();
  // Should not throw
  cache.lockHandle(handle);
  cache.lockHandle(handle, true);
  cache.lockHandle(handle, false);
});

// ===== T16: NaivePrefixCache.evict 空返回 =====
test("T16 NaivePrefixCache.evict empty return", () => {
  const cache = new NaivePrefixCache(100, 16);
  assert.deepStrictEqual(cache.evict(100), []);
});

// ===== T17: NaivePrefixCache.sizeInfo =====
test("T17 NaivePrefixCache.sizeInfo always (0, 0)", () => {
  const cache = new NaivePrefixCache(100, 16);
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
  assert.strictEqual(cache.sizeInfo.protectedSize, 0);
  assert.strictEqual(cache.sizeInfo.totalSize, 0);
});

// ===== T18: NaivePrefixCache.reset =====
test("T18 NaivePrefixCache.reset noop", () => {
  const cache = new NaivePrefixCache(100, 16);
  cache.reset();
  assert.strictEqual(cache.sizeInfo.totalSize, 0);
});

// ===== T19: NaivePrefixCache.checkIntegrity =====
test("T19 NaivePrefixCache.checkIntegrity noop", () => {
  const cache = new NaivePrefixCache(100, 16);
  // Should not throw
  cache.checkIntegrity();
});

// ===== Boundary: B1 — 分配全部页后耗尽 =====
test("B1 MockKVCachePool allocate all pages then exhaust", () => {
  const pool = new MockKVCachePool(testModelConfig, 5, 16);
  pool.allocatePaged(5);
  assert.strictEqual(pool.freePages, 0);
  assert.throws(
    () => pool.allocatePaged(1),
    /allocatePaged failed/
  );
});

// ===== Boundary: B2 — deallocate 后重新 allocate =====
test("B2 MockKVCachePool deallocate then allocate", () => {
  const pool = new MockKVCachePool(testModelConfig, 10, 16);
  const alloc = pool.allocatePaged(5);
  pool.deallocatePageAllocation(alloc);
  assert.strictEqual(pool.freePages, 10);
  const alloc2 = pool.allocatePaged(3);
  assert.strictEqual(alloc2.pages.length, 3);
  assert.strictEqual(pool.usedPages, 3);
});

// ===== Boundary: B3 — numPages=0 =====
test("B3 MockKVCachePool(numPages=0)", () => {
  const pool = new MockKVCachePool(testModelConfig, 0, 16);
  assert.strictEqual(pool.freePagesPool.length, 0);
  assert.strictEqual(pool.totalCapacity, 0);
  assert.strictEqual(pool.usedCapacity, 0);
});

// ===== Boundary: B4 — numPages=1 =====
test("B4 MockKVCachePool(numPages=1)", () => {
  const pool = new MockKVCachePool(testModelConfig, 1, 16);
  const alloc = pool.allocatePaged(1);
  assert.strictEqual(alloc.pages.length, 1);
  assert.strictEqual(alloc.slotCount, 16);
  assert.strictEqual(pool.freePages, 0);
  assert.throws(
    () => pool.allocatePaged(1),
    /allocatePaged failed/
  );
  pool.deallocatePageAllocation(alloc);
  assert.strictEqual(pool.freePages, 1);
});

// ===== Boundary: B5 — allocatePaged(0) =====
test("B5 MockKVCachePool.allocatePaged(0)", () => {
  const pool = new MockKVCachePool(testModelConfig, 10, 16);
  const alloc = pool.allocatePaged(0);
  assert.strictEqual(alloc.pages.length, 0);
  assert.strictEqual(alloc.slots.length, 0);
  assert.strictEqual(alloc.slotCount, 0);
  assert.strictEqual(pool.usedPages, 0);
});

// ===== Boundary: B6 — NaiveCacheHandle 非 0 构造 =====
test("B6 NaiveCacheHandle with non-zero cachedLen", () => {
  const handle = new NaiveCacheHandle(5);
  assert.strictEqual(handle.cachedLen, 5);
  assert.deepStrictEqual(handle.getMatchedIndices(), []);
});

// ===== Boundary: B7 — NaivePrefixCache 多次 match/insert =====
test("B7 NaivePrefixCache repeated match/insert stays miss", () => {
  const cache = new NaivePrefixCache(100, 16);
  for (let i = 0; i < 5; i++) {
    const mr = cache.matchPrefix([1, 2, 3]);
    assert.strictEqual(mr.cudaHandle.cachedLen, 0);
    const ir = cache.insertPrefix([1, 2, 3], [0, 1, 2]);
    assert.strictEqual(ir.cachedLen, 0);
  }
  assert.strictEqual(cache.sizeInfo.totalSize, 0);
});

// ===== Boundary: B8 — pageSize=16 的 slot 展开 =====
test("B8 MockKVCachePool pageSize=16 slot expansion", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  const alloc = pool.allocatePaged(2);
  assert.strictEqual(alloc.slots.length, 32);
  // Each page's slots should be contiguous
  const page0 = alloc.pages[0];
  const page1 = alloc.pages[1];
  const slots0 = alloc.slots.slice(0, 16);
  const slots1 = alloc.slots.slice(16, 32);
  assert.deepStrictEqual(slots0, Array.from({ length: 16 }, (_, i) => page0 + i));
  assert.deepStrictEqual(slots1, Array.from({ length: 16 }, (_, i) => page1 + i));
});

// ===== Boundary: B9 — decodeStepLatency numDecodeTokens=0 =====
test("B9 MockKVCachePool.decodeStepLatency numDecodeTokens=0", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  assert.strictEqual(pool.decodeStepLatency(0, 1, 7), 7);  // 0*1 + 7
});

// ===== Type hierarchy verification =====
test("T_extra MockKVCachePool extends BaseKVCachePool", () => {
  const pool = new MockKVCachePool(testModelConfig, 100, 16);
  assert.ok(pool instanceof BaseKVCachePool);
});

test("T_extra NaivePrefixCache extends BasePrefixCache", () => {
  const cache = new NaivePrefixCache(100, 16);
  assert.ok(cache instanceof BasePrefixCache);
});

test("T_extra NaiveCacheHandle extends BaseCacheHandle", () => {
  const handle = new NaiveCacheHandle();
  assert.ok(handle instanceof BaseCacheHandle);
});

// Summary
console.log("\n=== K2 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 K2 验收测试通过 \u2713");
  process.exit(0);
}
