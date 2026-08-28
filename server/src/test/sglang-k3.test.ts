import assert from "assert";
import {
  CacheManager,
  NaiveCacheHandle,
  NaivePrefixCache,
  BaseCacheHandle,
  BasePrefixCache,
  MatchResult,
  InsertResult,
} from "../sglang";

/**
 * Issue #14 验收测试 — K3: CacheManager（naive backend）§9.11
 *
 * Run with:  npx ts-node src/test/sglang-k3.test.ts
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

// Helper: create a CacheManager with a fresh pageTable
function makeCacheManager(numPages: number, pageSize: number): CacheManager {
  const maxSlots = numPages * pageSize;
  const pageTable: number[][] = [new Array(maxSlots).fill(0)];
  return new CacheManager(numPages, pageSize, pageTable);
}

// ===== T1: CacheManager 构造（naive backend）=====
test("T1 CacheManager construction (naive backend)", () => {
  const cm = makeCacheManager(100, 16);
  assert.strictEqual(cm.numPages, 100);
  assert.strictEqual(cm.pageSize, 16);
  assert.strictEqual(cm.freeSlots.length, 100);
  assert.strictEqual(cm.freeSlots[0], 0);
  assert.strictEqual(cm.freeSlots[1], 16);
  assert.strictEqual(cm.freeSlots[99], 99 * 16);
  assert.ok(cm.prefixCache instanceof NaivePrefixCache);
});

// ===== T2: CacheManager.availableSize =====
test("T2 CacheManager.availableSize", () => {
  const cm = makeCacheManager(100, 16);
  // naive: evictableSize=0, so availableSize = 0 + 100*16 = 1600
  assert.strictEqual(cm.availableSize, 1600);
});

// ===== T3: CacheManager.matchReq =====
test("T3 CacheManager.matchReq — naive always miss", () => {
  const cm = makeCacheManager(100, 16);
  const result = cm.matchReq({ inputIds: [1, 2, 3, 4, 5], inputLen: 5 });
  assert.ok(result instanceof MatchResult);
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
  assert.ok(result.cudaHandle instanceof NaiveCacheHandle);
});

// ===== T4: CacheManager.matchReq → lock → unlock 完整链路 =====
test("T4 CacheManager.matchReq → lock → unlock chain", () => {
  const cm = makeCacheManager(100, 16);
  const result = cm.matchReq({ inputIds: [1, 2, 3, 4, 5], inputLen: 5 });
  // lock/unlock should not throw (naive noop)
  assert.doesNotThrow(() => cm.lock(result.cudaHandle));
  assert.doesNotThrow(() => cm.unlock(result.cudaHandle));
});

// ===== T5: CacheManager.lock/unlock =====
test("T5 CacheManager.lock/unlock — naive noop", () => {
  const cm = makeCacheManager(100, 16);
  const handle = new NaiveCacheHandle(0);
  assert.doesNotThrow(() => cm.lock(handle));
  assert.doesNotThrow(() => cm.unlock(handle));
});

// ===== T6: CacheManager.allocatePaged 基本分配 =====
test("T6 CacheManager.allocatePaged basic allocation", () => {
  const numPages = 100, pageSize = 16;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);
  cm.allocatePaged({ deviceLen: 16, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 99);
  // pageTable[0] should have the allocated page physical position at [0..15]
  const pageIdx = pageTable[0][0];
  assert.ok(pageIdx >= 0);
  // All positions within the page should have the same physical position
  for (let i = 0; i < 16; i++) {
    assert.strictEqual(pageTable[0][i], pageIdx);
  }
});

// ===== T7: CacheManager.allocatePaged 多页分配 =====
test("T7 CacheManager.allocatePaged multi-page allocation", () => {
  const numPages = 100, pageSize = 16;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);
  cm.allocatePaged({ deviceLen: 48, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 97);
});

// ===== T8: CacheManager.allocatePaged 不足触发 eviction =====
test("T8 CacheManager.allocatePaged insufficient — eviction returns empty, break", () => {
  const numPages = 3, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);
  // Need 4 pages but only 3 available; naive evict returns []
  cm.allocatePaged({ deviceLen: 16, cachedLen: 0, tableIdx: 0 });
  // Only 3 pages available, so break after 3 allocations
  assert.strictEqual(cm.freeSlots.length, 0);
});

// ===== T9: CacheManager.cacheReq（finished=true）=====
test("T9 CacheManager.cacheReq finished=true — releases all pages", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // First allocate pages
  cm.allocatePaged({ deviceLen: 8, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 8); // 10 - 2 pages

  // Cache req with finished=true should release all pages
  const req = {
    inputIds: [1, 2, 3, 4, 5, 6, 7, 8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  // All pages freed (naive: newHandle.cachedLen=0, so _free(pageIndices[0:]) frees everything)
  assert.strictEqual(cm.freeSlots.length, 10);
});

// ===== T10: CacheManager.cacheReq（finished=false）=====
test("T10 CacheManager.cacheReq finished=false — retains pages, updates handle", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  cm.allocatePaged({ deviceLen: 8, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 8);

  const req = {
    inputIds: [1, 2, 3, 4, 5, 6, 7, 8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, false);
  // Pages NOT freed when finished=false
  assert.strictEqual(cm.freeSlots.length, 8);
  // Handle updated
  assert.ok(req.cacheHandle !== null);
  assert.ok(req.cacheHandle instanceof NaiveCacheHandle);
});

// ===== T11: CacheManager._free 页对齐 Set 去重 =====
test("T11 CacheManager._free — Set dedup for non-page-aligned slice (pageSize=4)", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // Manually set up pageTable with known values: [0,0,0,0, 4,4,4,4, 8,8,8,8]
  pageTable[0] = [0,0,0,0, 4,4,4,4, 8,8,8,8];

  // Allocate 3 pages first (consumes from freeSlots)
  cm.allocatePaged({ deviceLen: 12, cachedLen: 0, tableIdx: 0 });
  const freeAfterAlloc = cm.freeSlots.length;

  // Now cacheReq finished=true with cachedLen=12 (aligned)
  // pageIndices = [0,0,0,0, 4,4,4,4, 8,8,8,8]
  // _free(pageIndices.slice(0)) → Set dedup → [0, 4, 8]
  const req = {
    inputIds: [1,2,3,4,5,6,7,8,9,10,11,12],
    cachedLen: 12,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);

  // Should have recovered exactly 3 page positions (0, 4, 8)
  assert.strictEqual(cm.freeSlots.length, freeAfterAlloc + 3);
});

// ===== T12: CacheManager._free 多页重复释放验证 =====
test("T12 CacheManager._free — multi-page duplicate release verification", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // Setup: pageTable with duplicate physical positions within a page
  pageTable[0] = [0,0,0,0, 4,4,4,4];

  // Simulate a non-aligned slice that would cause duplicates
  // If we free pageIndices[2:6] = [0,0,4,4], Set dedup gives [0,4] (2 pages)
  // We test this via cacheReq with a manual setup

  // First allocate to consume pages
  cm.allocatePaged({ deviceLen: 8, cachedLen: 0, tableIdx: 0 });
  const freeAfterAlloc = cm.freeSlots.length;

  // Set up oldHandle with cachedLen=2 to create non-aligned slice
  // pageIndices = [0,0,0,0, 4,4,4,4], alignedLen=8
  // We simulate oldHandle.cachedLen=2 by creating a NaiveCacheHandle(2)
  // But since NaiveCacheHandle.cachedLen is set at construction...
  // The _free call will be: pageIndices.slice(oldHandle.cachedLen, cachedLen)
  // = pageIndices.slice(2, 0) = [] (since naive cachedLen=0 always)
  // So for naive backend this is always an empty operation

  // Instead, test indirectly: after finished=true, all pages are freed correctly
  const req = {
    inputIds: [1,2,3,4,5,6,7,8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  // Set dedup should free [0, 4] — 2 unique page positions
  assert.strictEqual(cm.freeSlots.length, freeAfterAlloc + 2);

  // Verify page count invariant
  cm.checkIntegrity();
});

// ===== T13: CacheManager.beginLazyFree/endLazyFree =====
test("T13 CacheManager.beginLazyFree/endLazyFree", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  cm.allocatePaged({ deviceLen: 8, cachedLen: 0, tableIdx: 0 });
  const freeBefore = cm.freeSlots.length;

  cm.beginLazyFree();
  // cacheReq finished=true inside lazy free context
  const req = {
    inputIds: [1,2,3,4,5,6,7,8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  // Pages should NOT go to freeSlots yet
  assert.strictEqual(cm.freeSlots.length, freeBefore);
  // They should be in lazyFreeList
  assert.ok(cm.lazyFreeList.length > 0);

  cm.endLazyFree();
  // Now pages should be in freeSlots
  assert.strictEqual(cm.freeSlots.length, freeBefore + 2); // 2 pages freed
  assert.strictEqual(cm.lazyFreeList.length, 0);
});

// ===== T14: CacheManager.checkIntegrity =====
test("T14 CacheManager.checkIntegrity — naive initial state", () => {
  const cm = makeCacheManager(100, 16);
  // Should not throw — initial state has 0 allocated, 0 in cache
  assert.doesNotThrow(() => cm.checkIntegrity());
});

// ===== T15: CacheManager.checkIntegrity 失败 =====
test("T15 CacheManager.checkIntegrity — failure on corrupted freeSlots", () => {
  const cm = makeCacheManager(100, 16);
  // Corrupt: remove some free slots to simulate allocation without cache tracking
  cm.freeSlots.splice(0, 5); // Now 95 free but 0 in cache → mismatch
  assert.throws(() => cm.checkIntegrity(), /integrity check failed/);
});

// ===== T16: CacheManager.freeCache =====
test("T16 CacheManager.freeCache — semantics equal to cacheReq(finished=true)", () => {
  const numPages = 10, pageSize = 4;
  const pageTable1: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm1 = new CacheManager(numPages, pageSize, pageTable1);

  const numPages2 = 10, pageSize2 = 4;
  const pageTable2: number[][] = [new Array(numPages2 * pageSize2).fill(0)];
  const cm2 = new CacheManager(numPages2, pageSize2, pageTable2);

  // Both allocate and then free
  cm1.allocatePaged({ deviceLen: 8, cachedLen: 0, tableIdx: 0 });
  cm2.allocatePaged({ deviceLen: 8, cachedLen: 0, tableIdx: 0 });

  const req1 = {
    inputIds: [1,2,3,4,5,6,7,8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  const req2 = {
    inputIds: [1,2,3,4,5,6,7,8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };

  cm1.cacheReq(req1, true);
  cm2.freeCache(req2);

  // Both should have same number of free slots
  assert.strictEqual(cm1.freeSlots.length, cm2.freeSlots.length);
});

// ===== T17: 短 prompt 测试 =====
test("T17 Short prompt — inputIds.length < pageSize", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // inputIds shorter than pageSize
  cm.allocatePaged({ deviceLen: 2, cachedLen: 0, tableIdx: 0 });
  // divCeil(2,4) = 1, divCeil(0,4) = 0, need 1 page
  assert.strictEqual(cm.freeSlots.length, 9);

  const req = {
    inputIds: [1, 2],
    cachedLen: 2,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  // alignedLen = floor(2/4)*4 = 0, so pageIndices = [] (empty slice)
  // _free([]) is noop
  cm.cacheReq(req, true);
  // No pages freed via cacheReq (alignedLen=0 → pageIndices empty)
  // But the page was allocated, so it stays allocated
  assert.strictEqual(cm.freeSlots.length, 9);
});

// ===== T18: 中 prompt 测试 =====
test("T18 Medium prompt — inputIds.length = 3 * pageSize", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  cm.allocatePaged({ deviceLen: 12, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 7);

  const req = {
    inputIds: [1,2,3,4,5,6,7,8,9,10,11,12],
    cachedLen: 12,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  // 3 pages freed (Set dedup from pageIndices)
  assert.strictEqual(cm.freeSlots.length, 10);
});

// ===== T19: 长 prompt 测试 =====
test("T19 Long prompt — inputIds.length = 10 * pageSize", () => {
  const numPages = 20, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  cm.allocatePaged({ deviceLen: 40, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 10);

  const req = {
    inputIds: Array.from({ length: 40 }, (_, i) => i + 1),
    cachedLen: 40,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  assert.strictEqual(cm.freeSlots.length, 20);
});

// ===== T20: matched < computed 场景 =====
test("T20 matched < computed — oldHandle.cachedLen < insertResult.cachedLen", () => {
  // In naive backend, insertResult.cachedLen is always 0,
  // so we cannot have matched < computed naturally.
  // We simulate by using cacheReq with an existing oldHandle.
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  cm.allocatePaged({ deviceLen: 8, cachedLen: 0, tableIdx: 0 });

  // First cacheReq with finished=false to get a handle
  const req1 = {
    inputIds: [1,2,3,4,5,6,7,8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req1, false);
  const handle = req1.cacheHandle;
  assert.ok(handle !== null);

  // Second cacheReq with the old handle — naive always returns cachedLen=0
  // so region 2 = pageIndices.slice(oldHandle.cachedLen, 0) = empty
  const req2 = {
    inputIds: [1,2,3,4,5,6,7,8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: handle,
  };
  // This should not throw
  assert.doesNotThrow(() => cm.cacheReq(req2, true));
});

// ===== T21: prefix 增长场景 =====
test("T21 Prefix growth — availableSize monotonically decreases", () => {
  const numPages = 20, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  const sizes: number[] = [cm.availableSize];
  for (let i = 1; i <= 5; i++) {
    cm.allocatePaged({ deviceLen: i * 4, cachedLen: 0, tableIdx: 0 });
    sizes.push(cm.availableSize);
  }
  // availableSize should be monotonically decreasing
  for (let i = 1; i < sizes.length; i++) {
    assert.ok(sizes[i] <= sizes[i - 1], `availableSize not decreasing: ${sizes[i - 1]} -> ${sizes[i]}`);
  }
});

// ===== T22: lazyFreeRegion 正确计数 =====
test("T22 lazyFreeRegion correct count", () => {
  const numPages = 20, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // Allocate 2 separate requests
  cm.allocatePaged({ deviceLen: 4, cachedLen: 0, tableIdx: 0 });
  cm.allocatePaged({ deviceLen: 8, cachedLen: 4, tableIdx: 0 });
  const freeBefore = cm.freeSlots.length;

  cm.beginLazyFree();
  const req1 = {
    inputIds: [1,2,3,4],
    cachedLen: 4,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req1, true);

  const req2 = {
    inputIds: [1,2,3,4,5,6,7,8],
    cachedLen: 8,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req2, true);

  // lazyFreeList should contain freed page indices from both requests
  const lazyCount = cm.lazyFreeList.length;
  assert.ok(lazyCount > 0, "lazyFreeList should not be empty");
  assert.strictEqual(cm.freeSlots.length, freeBefore, "freeSlots unchanged during lazy mode");

  cm.endLazyFree();
  assert.strictEqual(cm.freeSlots.length, freeBefore + lazyCount);
  assert.strictEqual(cm.lazyFreeList.length, 0);
});

// ===== T23: availableSize 不越界 =====
test("T23 availableSize within bounds [0, numPages * pageSize]", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  const maxSize = numPages * pageSize;
  assert.ok(cm.availableSize >= 0);
  assert.ok(cm.availableSize <= maxSize);

  // Allocate all pages
  cm.allocatePaged({ deviceLen: 40, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 0);
  assert.ok(cm.availableSize >= 0);
  assert.ok(cm.availableSize <= maxSize);
});

// ===== T24: _free Set 去重后的页数守恒 =====
test("T24 _free Set dedup page count invariant after non-aligned frees", () => {
  const numPages = 10, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // Allocate some pages
  cm.allocatePaged({ deviceLen: 12, cachedLen: 0, tableIdx: 0 });
  cm.checkIntegrity(); // Should pass

  // Free via cacheReq
  const req = {
    inputIds: [1,2,3,4,5,6,7,8,9,10,11,12],
    cachedLen: 12,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  cm.checkIntegrity(); // Should pass after freeing

  assert.strictEqual(cm.freeSlots.length, 10); // All recovered
});

// ===== Boundary: B1 — CacheManager(numPages=0) =====
test("B1 CacheManager(numPages=0)", () => {
  const cm = makeCacheManager(0, 16);
  assert.strictEqual(cm.freeSlots.length, 0);
  assert.strictEqual(cm.availableSize, 0);
});

// ===== Boundary: B2 — CacheManager(numPages=1, pageSize=1) =====
test("B2 CacheManager(numPages=1, pageSize=1)", () => {
  const pageTable: number[][] = [new Array(1).fill(0)];
  const cm = new CacheManager(1, 1, pageTable);

  cm.allocatePaged({ deviceLen: 1, cachedLen: 0, tableIdx: 0 });
  assert.strictEqual(cm.freeSlots.length, 0);

  const req = {
    inputIds: [1],
    cachedLen: 1,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  assert.strictEqual(cm.freeSlots.length, 1);
});

// ===== Boundary: B3 — cacheReq with alignedLen=0 =====
test("B3 cacheReq with alignedLen=0", () => {
  const cm = makeCacheManager(10, 4);
  const req = {
    inputIds: [1, 2],
    cachedLen: 2, // floor(2/4)*4 = 0
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  // Should not throw, pageIndices is empty
  assert.doesNotThrow(() => cm.cacheReq(req, true));
});

// ===== Boundary: B4 — allocatePaged with neededPages=0 =====
test("B4 allocatePaged with neededPages=0", () => {
  const cm = makeCacheManager(10, 4);
  const freeBefore = cm.freeSlots.length;
  cm.allocatePaged({ deviceLen: 3, cachedLen: 3, tableIdx: 0 });
  // divCeil(3,4)=1, divCeil(3,4)=1, need 0 pages
  assert.strictEqual(cm.freeSlots.length, freeBefore);
});

// ===== Boundary: B5 — endLazyFree without beginLazyFree =====
test("B5 endLazyFree without beginLazyFree", () => {
  const cm = makeCacheManager(10, 4);
  const freeBefore = cm.freeSlots.length;
  // _inLazyFree is false, calling endLazyFree should merge empty lazyFreeList
  assert.doesNotThrow(() => cm.endLazyFree());
  assert.strictEqual(cm.freeSlots.length, freeBefore);
});

// ===== Boundary: B6 — pageSize=16 的 _free Set 去重 =====
test("B6 pageSize=16 _free Set dedup", () => {
  const numPages = 10, pageSize = 16;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // pageTable with pageSize=16: each page has 16 identical entries
  for (let p = 0; p < 3; p++) {
    for (let i = 0; i < 16; i++) {
      pageTable[0][p * 16 + i] = p * 16;
    }
  }

  cm.allocatePaged({ deviceLen: 48, cachedLen: 0, tableIdx: 0 });
  const freeAfterAlloc = cm.freeSlots.length;

  const req = {
    inputIds: Array.from({ length: 48 }, (_, i) => i),
    cachedLen: 48,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  // Should recover 3 unique page positions
  assert.strictEqual(cm.freeSlots.length, freeAfterAlloc + 3);
  cm.checkIntegrity();
});

// ===== Boundary: B7 — matchReq with inputLen=1 =====
test("B7 matchReq with inputLen=1", () => {
  const cm = makeCacheManager(10, 4);
  // inputLen=1 → slice(0, 0) → matchPrefix([]) → NaiveCacheHandle(0)
  const result = cm.matchReq({ inputIds: [1], inputLen: 1 });
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
});

// ===== Boundary: B8 — 连续 allocatePaged 全部页耗尽 =====
test("B8 allocatePaged exhaust all pages then break", () => {
  const numPages = 3, pageSize = 4;
  const pageTable: number[][] = [new Array(numPages * pageSize).fill(0)];
  const cm = new CacheManager(numPages, pageSize, pageTable);

  // Request more pages than available
  cm.allocatePaged({ deviceLen: 16, cachedLen: 0, tableIdx: 0 });
  // Only 3 pages available, breaks after 3
  assert.strictEqual(cm.freeSlots.length, 0);
});

// ===== Boundary: B9 — _free 传入空数组 =====
test("B9 _free with empty array — noop", () => {
  const cm = makeCacheManager(10, 4);
  const freeBefore = cm.freeSlots.length;

  // cacheReq with alignedLen=0 results in _free([])
  const req = {
    inputIds: [1],
    cachedLen: 1,
    tableIdx: 0,
    cacheHandle: null as BaseCacheHandle | null,
  };
  cm.cacheReq(req, true);
  assert.strictEqual(cm.freeSlots.length, freeBefore);
});

// ===== Type hierarchy verification =====
test("T_extra CacheManager is exported as class", () => {
  const cm = makeCacheManager(10, 4);
  assert.ok(cm instanceof CacheManager);
});

test("T_extra CacheManager.constructor rejects radix cacheType", () => {
  const pageTable: number[][] = [[]];
  assert.throws(
    () => new CacheManager(10, 4, pageTable, "radix"),
    /RadixPrefixCache not implemented/,
  );
});

// Summary
console.log("\n=== K3 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 K3 验收测试通过 \u2713");
  process.exit(0);
}
