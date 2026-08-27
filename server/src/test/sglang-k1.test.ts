import assert from "assert";
import {
  CacheSizeInfo,
  BaseCacheHandle,
  MatchResult,
  InsertResult,
  BaseKVCachePool,
  BasePrefixCache,
  TableManager,
} from "../sglang";

/**
 * Issue #11 验收测试 — K1: KVCache 基础抽象层
 *
 * Run with:  npx ts-node src/test/sglang-k1.test.ts
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

// ===== T1: CacheSizeInfo 构造 =====
test("T1 CacheSizeInfo construction", () => {
  const info1 = new CacheSizeInfo();
  assert.strictEqual(info1.evictableSize, 0);
  assert.strictEqual(info1.protectedSize, 0);

  const info2 = new CacheSizeInfo(100, 50);
  assert.strictEqual(info2.evictableSize, 100);
  assert.strictEqual(info2.protectedSize, 50);
});

// ===== T2: CacheSizeInfo.totalSize =====
test("T2 CacheSizeInfo.totalSize", () => {
  const info = new CacheSizeInfo(100, 50);
  assert.strictEqual(info.totalSize, 150);
});

// ===== T3: MatchResult 构造 =====
test("T3 MatchResult construction", () => {
  class TestHandle extends BaseCacheHandle {
    get cachedLen(): number { return 5; }
    getMatchedIndices(): number[] { return [0, 1, 2]; }
  }
  const handle = new TestHandle();
  const result = new MatchResult(handle);
  assert.strictEqual(result.cudaHandle, handle);
  assert.strictEqual(result.cudaHandle.cachedLen, 5);
});

// ===== T4: MatchResult 不可变 =====
test("T4 MatchResult readonly fields", () => {
  class TestHandle extends BaseCacheHandle {
    get cachedLen(): number { return 0; }
    getMatchedIndices(): number[] { return []; }
  }
  const handle = new TestHandle();
  const result = new MatchResult(handle);
  // TypeScript readonly is a compile-time constraint;
  // verify the handle reference remains stable after construction
  assert.strictEqual(result.cudaHandle, handle);
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
});

// ===== T5: InsertResult 构造 =====
test("T5 InsertResult construction", () => {
  class TestHandle extends BaseCacheHandle {
    get cachedLen(): number { return 10; }
    getMatchedIndices(): number[] { return [0, 1]; }
  }
  const handle = new TestHandle();
  const result = new InsertResult(8, handle);
  assert.strictEqual(result.cachedLen, 8);
  assert.strictEqual(result.cudaHandle, handle);
});

// ===== T6: InsertResult 不可变 =====
test("T6 InsertResult readonly fields", () => {
  class TestHandle extends BaseCacheHandle {
    get cachedLen(): number { return 0; }
    getMatchedIndices(): number[] { return []; }
  }
  const handle = new TestHandle();
  const result = new InsertResult(0, handle);
  // TypeScript readonly is a compile-time constraint;
  // verify values remain stable after construction
  assert.strictEqual(result.cachedLen, 0);
  assert.strictEqual(result.cudaHandle, handle);
});

// ===== T7: BaseCacheHandle 子类化 =====
test("T7 BaseCacheHandle subclassing", () => {
  class TestHandle extends BaseCacheHandle {
    private _len: number;
    private _indices: number[];
    constructor(len: number, indices: number[]) {
      super();
      this._len = len;
      this._indices = indices;
    }
    get cachedLen(): number { return this._len; }
    getMatchedIndices(): number[] { return this._indices; }
  }
  const h = new TestHandle(7, [0, 1, 2, 3, 4, 5, 6]);
  assert.strictEqual(h.cachedLen, 7);
  assert.deepStrictEqual(h.getMatchedIndices(), [0, 1, 2, 3, 4, 5, 6]);
});

// ===== T8: BaseKVCachePool 子类化 =====
test("T8 BaseKVCachePool subclassing", () => {
  class MockPool extends BaseKVCachePool {
    get numPages(): number { return 100; }
    get pageSize(): number { return 16; }
    get totalCapacity(): number { return 1600; }
    get usedCapacity(): number { return 500; }
    storeKV(_k: number[], _v: number[], _outLoc: number[], _layerId: number): void {}
  }
  const pool = new MockPool();
  assert.strictEqual(pool.numPages, 100);
  assert.strictEqual(pool.pageSize, 16);
  assert.strictEqual(pool.totalCapacity, 1600);
  assert.strictEqual(pool.usedCapacity, 500);
  // storeKV should not throw
  pool.storeKV([], [], [], 0);
});

// ===== T9: BasePrefixCache 子类化 =====
test("T9 BasePrefixCache subclassing", () => {
  class TestHandle extends BaseCacheHandle {
    get cachedLen(): number { return 0; }
    getMatchedIndices(): number[] { return []; }
  }
  const dummyHandle = new TestHandle();
  class MockCache extends BasePrefixCache {
    private _sizeInfo = new CacheSizeInfo(50, 30);
    get sizeInfo(): CacheSizeInfo { return this._sizeInfo; }
    matchPrefix(_inputIds: number[]): MatchResult { return new MatchResult(dummyHandle); }
    insertPrefix(_inputIds: number[], _indices: number[]): InsertResult { return new InsertResult(0, dummyHandle); }
    lockHandle(_handle: BaseCacheHandle, _unlock?: boolean): void {}
    evict(_size: number): number[] { return []; }
    reset(): void { this._sizeInfo.evictableSize = 0; this._sizeInfo.protectedSize = 0; }
    checkIntegrity(): void {}
  }
  const cache = new MockCache();
  assert.strictEqual(cache.sizeInfo.totalSize, 80);
  const matchResult = cache.matchPrefix([1, 2, 3]);
  assert.ok(matchResult instanceof MatchResult);
  const insertResult = cache.insertPrefix([1, 2, 3], [0, 1, 2]);
  assert.ok(insertResult instanceof InsertResult);
  cache.lockHandle(dummyHandle);
  cache.lockHandle(dummyHandle, true);
  assert.deepStrictEqual(cache.evict(10), []);
  cache.reset();
  assert.strictEqual(cache.sizeInfo.totalSize, 0);
  cache.checkIntegrity();
});

// ===== T10: TableManager 构造 =====
test("T10 TableManager construction", () => {
  const pageTable = Array.from({ length: 5 }, () => new Array(4).fill(0));
  const tm = new TableManager(4, pageTable);
  assert.strictEqual(tm.maxRunningReq, 4);
  assert.deepStrictEqual(tm.freeTableIndices, [0, 1, 2, 3]);
  assert.strictEqual(tm.availableSize, 4);
});

// ===== T11: TableManager.allocate =====
test("T11 TableManager.allocate", () => {
  const pageTable = Array.from({ length: 5 }, () => new Array(4).fill(0));
  const tm = new TableManager(4, pageTable);
  const idx = tm.allocate();
  assert.strictEqual(idx, 3); // pop from end
  assert.strictEqual(tm.availableSize, 3);
  const idx2 = tm.allocate();
  assert.strictEqual(idx2, 2);
});

// ===== T12: TableManager.free =====
test("T12 TableManager.free", () => {
  const pageTable = Array.from({ length: 5 }, () => new Array(4).fill(0));
  const tm = new TableManager(4, pageTable);
  const idx = tm.allocate();
  assert.strictEqual(tm.availableSize, 3);
  tm.free(idx);
  assert.strictEqual(tm.availableSize, 4);
});

// ===== T13: TableManager.availableSize =====
test("T13 TableManager.availableSize", () => {
  const pageTable = Array.from({ length: 5 }, () => new Array(4).fill(0));
  const tm = new TableManager(4, pageTable);
  assert.strictEqual(tm.availableSize, 4);
  tm.allocate();
  assert.strictEqual(tm.availableSize, 3);
  tm.allocate();
  assert.strictEqual(tm.availableSize, 2);
  tm.free(3);
  assert.strictEqual(tm.availableSize, 3);
});

// ===== T14: TableManager 分配耗尽 =====
test("T14 TableManager allocation exhaustion throws Error", () => {
  const pageTable = Array.from({ length: 2 }, () => new Array(4).fill(0));
  const tm = new TableManager(1, pageTable);
  tm.allocate();
  assert.throws(() => tm.allocate(), /No available table indices/);
});

// ===== T15: TableManager.tokenPool =====
test("T15 TableManager.tokenPool", () => {
  const pageTable = Array.from({ length: 5 }, () => new Array(4).fill(0));
  const tm = new TableManager(4, pageTable);
  assert.strictEqual(tm.tokenPool.length, 5); // maxRunningReq + 1
  assert.strictEqual(tm.tokenPool[0].length, 4);
  assert.strictEqual(tm.tokenPool[4].length, 4);
  // all zeros
  assert.ok(tm.tokenPool.every(row => row.every(v => v === 0)));
});

// ===== T16: TableManager 循环分配释放 =====
test("T16 TableManager allocate-free-allocate cycle", () => {
  const pageTable = Array.from({ length: 5 }, () => new Array(4).fill(0));
  const tm = new TableManager(4, pageTable);
  const idx1 = tm.allocate(); // 3
  const idx2 = tm.allocate(); // 2
  tm.free(idx1);              // push 3
  const idx3 = tm.allocate(); // should get 3 back
  assert.strictEqual(idx3, 3);
  assert.strictEqual(tm.availableSize, 2);
});

// ===== Boundary: B1 — CacheSizeInfo(0, 0) =====
test("B1 CacheSizeInfo(0, 0) totalSize = 0", () => {
  const info = new CacheSizeInfo(0, 0);
  assert.strictEqual(info.totalSize, 0);
});

// ===== Boundary: B2 — TableManager(maxRunningReq=1) =====
test("B2 TableManager(maxRunningReq=1) only one available row", () => {
  const pageTable = Array.from({ length: 2 }, () => new Array(4).fill(0));
  const tm = new TableManager(1, pageTable);
  assert.strictEqual(tm.availableSize, 1);
  assert.strictEqual(tm.tokenPool.length, 2); // 1 + 1 for dummy
});

// ===== Boundary: B3 — TableManager 连续 allocate 直到耗尽 =====
test("B3 TableManager allocate until exhaustion", () => {
  const pageTable = Array.from({ length: 3 }, () => new Array(4).fill(0));
  const tm = new TableManager(2, pageTable);
  tm.allocate();
  tm.allocate();
  assert.throws(() => tm.allocate(), /No available table indices/);
});

// ===== Boundary: B4 — TableManager free 后重新 allocate =====
test("B4 TableManager free then allocate returns freed index", () => {
  const pageTable = Array.from({ length: 3 }, () => new Array(4).fill(0));
  const tm = new TableManager(2, pageTable);
  const idx1 = tm.allocate();
  const idx2 = tm.allocate();
  tm.free(idx1);
  const idx3 = tm.allocate();
  assert.strictEqual(idx3, idx1);
});

// ===== Boundary: B5 — MatchResult handle.cachedLen = 0 =====
test("B5 MatchResult with zero-length handle", () => {
  class ZeroHandle extends BaseCacheHandle {
    get cachedLen(): number { return 0; }
    getMatchedIndices(): number[] { return []; }
  }
  const handle = new ZeroHandle();
  const result = new MatchResult(handle);
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
  assert.deepStrictEqual(result.cudaHandle.getMatchedIndices(), []);
});

// ===== Boundary: B6 — InsertResult cachedLen = 0 =====
test("B6 InsertResult with cachedLen = 0", () => {
  class TestHandle extends BaseCacheHandle {
    get cachedLen(): number { return 0; }
    getMatchedIndices(): number[] { return []; }
  }
  const handle = new TestHandle();
  const result = new InsertResult(0, handle);
  assert.strictEqual(result.cachedLen, 0);
});

// ===== Boundary: B7 — BasePrefixCache.lockHandle 默认 unlock=false =====
test("B7 BasePrefixCache.lockHandle default unlock=false", () => {
  let unlockReceived: boolean | undefined = undefined;
  class TestHandle extends BaseCacheHandle {
    get cachedLen(): number { return 0; }
    getMatchedIndices(): number[] { return []; }
  }
  const dummyHandle = new TestHandle();
  class MockCache extends BasePrefixCache {
    private _sizeInfo = new CacheSizeInfo();
    get sizeInfo(): CacheSizeInfo { return this._sizeInfo; }
    matchPrefix(_inputIds: number[]): MatchResult { return new MatchResult(dummyHandle); }
    insertPrefix(_inputIds: number[], _indices: number[]): InsertResult { return new InsertResult(0, dummyHandle); }
    lockHandle(_handle: BaseCacheHandle, unlock?: boolean): void { unlockReceived = unlock; }
    evict(_size: number): number[] { return []; }
    reset(): void {}
    checkIntegrity(): void {}
  }
  const cache = new MockCache();
  cache.lockHandle(dummyHandle);
  assert.strictEqual(unlockReceived, undefined);
  cache.lockHandle(dummyHandle, false);
  assert.strictEqual(unlockReceived, false);
  cache.lockHandle(dummyHandle, true);
  assert.strictEqual(unlockReceived, true);
});

// Summary
console.log("\n=== K1 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 K1 验收测试通过 \u2713");
  process.exit(0);
}
