import assert from "assert";
import {
  SamplingParams,
  Req,
  Batch,
} from "../sglang/core";
import {
  ChunkedReq,
  PendingReq,
} from "../sglang/entities";
import {
  PrefillAdder,
  PrefillManager,
  DecodeManager,
  TableManager,
} from "../sglang/scheduler";
import {
  CacheManager,
  NaivePrefixCache,
  NaiveCacheHandle,
  BaseCacheHandle,
} from "../sglang/cache";

/**
 * Issue #16 验收测试 — S2: PrefillAdder + PrefillManager + DecodeManager
 *
 * Run with:  npx ts-node src/test/sglang-s2.test.ts
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

// ===== 辅助函数 =====

/** 创建测试用 CacheManager（naive backend，大容量） */
function makeCacheManager(numPages = 1024, pageSize = 1): {
  cm: CacheManager;
  pageTable: number[][];
} {
  const maxSeqLen = 8192;
  const maxRunningReq = 128;
  const pageTable = Array.from(
    { length: maxRunningReq + 1 },
    () => new Array(maxSeqLen).fill(0)
  );
  const cm = new CacheManager(numPages, pageSize, pageTable, "naive");
  return { cm, pageTable };
}

/** 创建测试用 TableManager */
function makeTableManager(maxRunningReq = 128, maxSeqLen = 8192): TableManager {
  const pageTable = Array.from(
    { length: maxRunningReq + 1 },
    () => new Array(maxSeqLen).fill(0)
  );
  return new TableManager(maxRunningReq, pageTable);
}

/** 创建测试用 DecodeManager */
function makeDecodeManager(pageSize = 1): DecodeManager {
  return new DecodeManager(pageSize);
}

/** 创建 SimRequestMsg */
function makeMsg(uid: number, inputIds: number[], outputLen = 100): {
  tag: "req_in";
  uid: number;
  inputIds: number[];
  samplingParams: SamplingParams;
  outputLen: number;
} {
  return {
    tag: "req_in",
    uid,
    inputIds,
    samplingParams: new SamplingParams({ maxNewTokens: outputLen }),
    outputLen,
  };
}

// ===== T1: PrefillAdder 构造 =====
test("T1 PrefillAdder construction", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(100, cm, tm, dm);
  assert.strictEqual(adder.consumedTokens, 0);
  assert.strictEqual(adder.remainingBudget, 100);
  assert.strictEqual(adder.reservedSize, 0);
});

// ===== T2: PrefillAdder.tryAddOne - 短 prompt 一次性 prefill =====
test("T2 PrefillAdder.tryAddOne - short prompt one-shot prefill", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(100, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.ok(result instanceof Req);
  assert.strictEqual(result!.rid, 1);
  assert.strictEqual(result!.deviceLen, 3);
  assert.strictEqual(adder.consumedTokens, 3);
  assert.strictEqual(dm.runningReqs.size, 1);
});

// ===== T3: PrefillAdder.tryAddOne - token budget 不足 -> ChunkedReq =====
test("T3 PrefillAdder.tryAddOne - token budget insufficient -> ChunkedReq", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(2, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.ok(result instanceof ChunkedReq, "should return ChunkedReq");
  assert.strictEqual((result as ChunkedReq).rid, 1);
  assert.strictEqual((result as ChunkedReq).deviceLen, 2);
  assert.strictEqual((result as ChunkedReq).canDecode, false);
  assert.strictEqual(adder.consumedTokens, 2);
  assert.strictEqual(dm.runningReqs.size, 0);
});

// ===== T4: PrefillAdder.tryAddOne - 第一次 available_size 检查失败 =====
test("T4 PrefillAdder.tryAddOne - first available_size check fail", () => {
  // 极小容量的 CacheManager
  const { cm } = makeCacheManager(1, 1);
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 先添加一个 decode 请求消耗 reservedSize
  const sp0 = new SamplingParams({ maxNewTokens: 10 });
  const decReq = new Req({ rid: 100, inputIds: [1], samplingParams: sp0 });
  dm.addReq(decReq);

  const adder = new PrefillAdder(100, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  // naive backend: availableSize = 1, estimatedLen=5, reservedSize>0
  assert.strictEqual(result, null);
});

// ===== T5: PrefillAdder.tryAddOne - 第二次 available_size 检查失败 =====
test("T5 PrefillAdder.tryAddOne - second available_size check fail (lock reduces available)", () => {
  // 在 naive backend 下，lock 是 noop，所以第二次检查结果与第一次相同
  // 为了测试此路径，我们需要一个 mock cacheManager 使得 lock 后 availableSize 减小
  // 使用自定义对象来模拟
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 创建一个包装器，lock 后 availableSize 减小
  let lockCount = 0;
  const { cm: realCm, pageTable } = makeCacheManager(1024, 1);
  const wrappedCm = {
    get availableSize() { return lockCount > 0 ? 0 : 1000; },
    matchReq: realCm.matchReq.bind(realCm),
    lock(_handle: BaseCacheHandle) { lockCount++; realCm.lock(_handle); },
    unlock(handle: BaseCacheHandle) { lockCount = Math.max(0, lockCount - 1); realCm.unlock(handle); },
    prefixCache: realCm.prefixCache,
    pageTable: realCm.pageTable,
    freeSlots: realCm.freeSlots,
    pageSize: realCm.pageSize,
    numPages: realCm.numPages,
    lazyFreeList: realCm.lazyFreeList,
    allocatePaged: realCm.allocatePaged.bind(realCm),
    cacheReq: realCm.cacheReq.bind(realCm),
    freeCache: realCm.freeCache.bind(realCm),
    beginLazyFree: realCm.beginLazyFree.bind(realCm),
    endLazyFree: realCm.endLazyFree.bind(realCm),
    checkIntegrity: realCm.checkIntegrity.bind(realCm),
  } as unknown as CacheManager;

  const adder = new PrefillAdder(100, wrappedCm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.strictEqual(result, null);
});

// ===== T6: PrefillAdder._tryAddOneChunked 续接 =====
test("T6 PrefillAdder._tryAddOneChunked - continuation", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 第一轮: budget=2, prompt 有 5 个 token -> ChunkedReq
  const adder1 = new PrefillAdder(2, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [10, 20, 30, 40, 50], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);

  assert.ok(result1 instanceof ChunkedReq);
  assert.strictEqual((result1 as ChunkedReq).deviceLen, 2);

  // 第二轮: 续接
  const pr2 = new PendingReq({
    rid: 1,
    inputIds: [10, 20, 30, 40, 50],
    samplingParams: sp,
    chunkedReq: result1 as ChunkedReq,
  });
  const adder2 = new PrefillAdder(100, cm, tm, dm);
  const result2 = adder2.tryAddOne(pr2);

  assert.ok(result2 instanceof Req, "continuation should return Req");
  assert.strictEqual((result2 as Req).rid, 1);
  assert.strictEqual((result2 as Req).deviceLen, 5);
  assert.strictEqual(adder2.consumedTokens, 3);
});

// ===== T7: PrefillAdder - 两次 available_size 检查行为一致性 =====
test("T7 PrefillAdder - two available_size checks consistency", () => {
  // 在 naive backend 下两次检查结果相同（lock 是 noop）
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 第一次：宽松通过 -> lock -> 第二次也通过 -> 正常分配
  const adder = new PrefillAdder(100, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);
  assert.ok(result instanceof Req);

  // 第二次测试：通过 mock 使第二次检查失败
  let lockCount = 0;
  const wrappedCm = {
    get availableSize() { return lockCount > 0 ? 0 : 1000; },
    matchReq: cm.matchReq.bind(cm),
    lock(_h: BaseCacheHandle) { lockCount++; cm.lock(_h); },
    unlock(h: BaseCacheHandle) { lockCount = Math.max(0, lockCount - 1); cm.unlock(h); },
    prefixCache: cm.prefixCache,
    pageTable: cm.pageTable,
    freeSlots: cm.freeSlots,
    pageSize: cm.pageSize,
    numPages: cm.numPages,
    lazyFreeList: cm.lazyFreeList,
    allocatePaged: cm.allocatePaged.bind(cm),
    cacheReq: cm.cacheReq.bind(cm),
    freeCache: cm.freeCache.bind(cm),
    beginLazyFree: cm.beginLazyFree.bind(cm),
    endLazyFree: cm.endLazyFree.bind(cm),
    checkIntegrity: cm.checkIntegrity.bind(cm),
  } as unknown as CacheManager;

  const adder2 = new PrefillAdder(100, wrappedCm, tm, dm);
  const pr2 = new PendingReq({ rid: 2, inputIds: [4, 5, 6], samplingParams: sp });
  const result2 = adder2.tryAddOne(pr2);
  assert.strictEqual(result2, null, "second check fail should return null");
});

// ===== T8: PrefillManager.addOneReq =====
test("T8 PrefillManager.addOneReq", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  const msg = makeMsg(1, [1, 2, 3]);
  pm.addOneReq(msg);

  assert.strictEqual(pm.pendingList.length, 1);
  assert.strictEqual(pm.pendingList[0].rid, 1);
  assert.deepStrictEqual(pm.pendingList[0].inputIds, [1, 2, 3]);
});

// ===== T9: PrefillManager.addBatch =====
test("T9 PrefillManager.addBatch", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  const msgs = [makeMsg(1, [1]), makeMsg(2, [2]), makeMsg(3, [3])];
  pm.addBatch(msgs);

  assert.strictEqual(pm.pendingList.length, 3);
  assert.strictEqual(pm.pendingList[0].rid, 1);
  assert.strictEqual(pm.pendingList[1].rid, 2);
  assert.strictEqual(pm.pendingList[2].rid, 3);
});

// ===== T10: PrefillManager.scheduleNextBatch - 空队列 =====
test("T10 PrefillManager.scheduleNextBatch - empty queue", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  const result = pm.scheduleNextBatch(100);
  assert.strictEqual(result, null);
});

// ===== T11: PrefillManager.scheduleNextBatch - 短 prompt 一次性 =====
test("T11 PrefillManager.scheduleNextBatch - short prompt one-shot", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  pm.addOneReq(makeMsg(1, [1, 2, 3]));
  const result = pm.scheduleNextBatch(100);

  assert.ok(result !== null);
  assert.strictEqual(result!.reqs.size, 1);
  assert.strictEqual(result!.reqs.get(1)!.rid, 1);
  assert.strictEqual(result!.extendInputTokens, 3);
  assert.strictEqual(pm.pendingList.length, 0);
  assert.strictEqual(dm.runningReqs.size, 1);
});

// ===== T12: PrefillManager.scheduleNextBatch - 长 prompt 分块 =====
test("T12 PrefillManager.scheduleNextBatch - long prompt chunked then continuation", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  pm.addOneReq(makeMsg(1, [1, 2, 3, 4, 5]));

  // 第一次 schedule: budget=2 -> chunked
  const batch1 = pm.scheduleNextBatch(2);
  assert.ok(batch1 !== null);
  assert.strictEqual(batch1!.reqs.size, 1);
  const creq = batch1!.reqs.get(1)!;
  assert.ok(creq instanceof ChunkedReq);
  assert.strictEqual(creq.deviceLen, 2);
  assert.strictEqual(pm.pendingList.length, 1);
  assert.ok(pm.pendingList[0].chunkedReq !== null);

  // 第二次 schedule: 续接完成
  const batch2 = pm.scheduleNextBatch(100);
  assert.ok(batch2 !== null);
  assert.strictEqual(batch2!.reqs.size, 1);
  const req = batch2!.reqs.get(1)!;
  assert.ok(req instanceof Req);
  assert.strictEqual(req.deviceLen, 5);
  assert.strictEqual(pm.pendingList.length, 0);
  assert.strictEqual(dm.runningReqs.size, 1);
});

// ===== T13: PrefillManager.abortReq - 存在的 uid =====
test("T13 PrefillManager.abortReq - existing uid", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  pm.addOneReq(makeMsg(1, [1, 2, 3]));

  // 先 schedule 使其变成 chunked
  pm.scheduleNextBatch(2);
  assert.strictEqual(pm.pendingList.length, 1);
  const chunked = pm.pendingList[0].chunkedReq;
  assert.ok(chunked !== null);

  const aborted = pm.abortReq(1);
  assert.ok(aborted !== null);
  assert.strictEqual(aborted!.rid, 1);
  assert.strictEqual(pm.pendingList.length, 0);
});

// ===== T14: PrefillManager.abortReq - 不存在的 uid =====
test("T14 PrefillManager.abortReq - non-existing uid", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  const result = pm.abortReq(999);
  assert.strictEqual(result, null);
});

// ===== T15: DecodeManager.addReq/removeReq =====
test("T15 DecodeManager.addReq/removeReq", () => {
  const dm = makeDecodeManager();
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });

  dm.addReq(req);
  assert.strictEqual(dm.runningReqs.size, 1);
  assert.ok(dm.runningReqs.has(req));

  dm.removeReq(req);
  assert.strictEqual(dm.runningReqs.size, 0);
});

// ===== T16: DecodeManager.filterReqs =====
test("T16 DecodeManager.filterReqs", () => {
  const dm = makeDecodeManager();
  const sp = new SamplingParams({ maxNewTokens: 2 });
  const req1 = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const req2 = new Req({ rid: 2, inputIds: [4, 5], samplingParams: sp });

  dm.addReq(req1);
  dm.addReq(req2);

  // req1 有 2 个 remainLen, req2 也有 2 个
  assert.strictEqual(dm.runningReqs.size, 2);

  // 消耗 req1 的所有 remainLen
  req1.completeOne();
  req1.completeOne();
  assert.strictEqual(req1.canDecode, false);
  assert.strictEqual(req2.canDecode, true);

  // filterReqs: req1 should be removed, newReqs is empty
  dm.filterReqs([]);
  assert.strictEqual(dm.runningReqs.size, 1);
  assert.ok(dm.runningReqs.has(req2));
});

// ===== T17: DecodeManager.inflightTokens =====
test("T17 DecodeManager.inflightTokens", () => {
  const dm = makeDecodeManager(4); // pageSize=4
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req1 = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  // remainLen = 3+10 - 3 = 10
  dm.addReq(req1);
  // inflightTokens = 10 + (4-1)*1 = 13
  assert.strictEqual(dm.inflightTokens, 13);

  const req2 = new Req({ rid: 2, inputIds: [1, 2], samplingParams: sp });
  // remainLen = 2+10 - 2 = 10
  dm.addReq(req2);
  // inflightTokens = 10+10 + (4-1)*2 = 26
  assert.strictEqual(dm.inflightTokens, 26);
});

// ===== T18: DecodeManager.scheduleNextBatch - 空集 =====
test("T18 DecodeManager.scheduleNextBatch - empty set", () => {
  const dm = makeDecodeManager();
  assert.strictEqual(dm.scheduleNextBatch(), null);
});

// ===== T19: DecodeManager.scheduleNextBatch - 非空 =====
test("T19 DecodeManager.scheduleNextBatch - non-empty", () => {
  const dm = makeDecodeManager();
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req2 = new Req({ rid: 2, inputIds: [1], samplingParams: sp });
  const req1 = new Req({ rid: 1, inputIds: [2], samplingParams: sp });

  dm.addReq(req2);
  dm.addReq(req1);

  const batch = dm.scheduleNextBatch();
  assert.ok(batch !== null);
  // 按 rid 排序
  assert.strictEqual(batch!.readyIds[0], 1);
  assert.strictEqual(batch!.readyIds[1], 2);
  assert.strictEqual(batch!.numDecodeTokens, 2);
});

// ===== T20: DecodeManager.abortReq =====
test("T20 DecodeManager.abortReq", () => {
  const dm = makeDecodeManager();
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req = new Req({ rid: 1, inputIds: [1], samplingParams: sp });

  dm.addReq(req);
  const aborted = dm.abortReq(1);
  assert.ok(aborted !== null);
  assert.strictEqual(aborted!.rid, 1);
  assert.strictEqual(dm.runningReqs.size, 0);

  // abort non-existing
  const aborted2 = dm.abortReq(999);
  assert.strictEqual(aborted2, null);
});

// ===== T21: PrefillManager + DecodeManager 集成 - 短 prompt 全流程 =====
test("T21 PrefillManager + DecodeManager integration - short prompt full flow", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  pm.addOneReq(makeMsg(1, [1, 2, 3]));

  // Prefill batch
  const prefillBatch = pm.scheduleNextBatch(100);
  assert.ok(prefillBatch !== null);
  assert.strictEqual(prefillBatch!.reqs.size, 1);

  // Req should be in decodeManager
  assert.strictEqual(dm.runningReqs.size, 1);

  // Decode batch
  const decodeBatch = dm.scheduleNextBatch();
  assert.ok(decodeBatch !== null);
  assert.strictEqual(decodeBatch!.reqs.size, 1);
  assert.strictEqual(decodeBatch!.readyIds[0], 1);
});

// ===== T22: PrefillManager + DecodeManager 集成 - 长 prompt 两次 tick =====
test("T22 PrefillManager + DecodeManager integration - long prompt two ticks", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  pm.addOneReq(makeMsg(1, [1, 2, 3, 4, 5]));

  // Tick 1: budget=2 -> chunked
  const prefillBatch1 = pm.scheduleNextBatch(2);
  assert.ok(prefillBatch1 !== null);
  const creq = prefillBatch1!.reqs.get(1)!;
  assert.ok(creq instanceof ChunkedReq);
  assert.strictEqual(dm.runningReqs.size, 0);

  // Tick 2: 续接完成 -> Req -> 加入 decodeManager
  const prefillBatch2 = pm.scheduleNextBatch(100);
  assert.ok(prefillBatch2 !== null);
  const req = prefillBatch2!.reqs.get(1)!;
  assert.ok(req instanceof Req);
  assert.strictEqual(dm.runningReqs.size, 1);

  // Decode batch
  const decodeBatch = dm.scheduleNextBatch();
  assert.ok(decodeBatch !== null);
  assert.strictEqual(decodeBatch!.reqs.size, 1);
});

// ===== T23: PrefillAdder - lock 后 unlock 回滚 =====
test("T23 PrefillAdder - unlock rollback after second check fail", () => {
  // 验证 cache_handle 已被正确 unlock，后续相同请求可重新 lock
  let lockCount = 0;
  const { cm, pageTable } = makeCacheManager(1024, 1);
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  const wrappedCm = {
    get availableSize() { return lockCount > 0 ? 0 : 1000; },
    matchReq: cm.matchReq.bind(cm),
    lock(_h: BaseCacheHandle) { lockCount++; cm.lock(_h); },
    unlock(h: BaseCacheHandle) { lockCount = Math.max(0, lockCount - 1); cm.unlock(h); },
    prefixCache: cm.prefixCache,
    pageTable: cm.pageTable,
    freeSlots: cm.freeSlots,
    pageSize: cm.pageSize,
    numPages: cm.numPages,
    lazyFreeList: cm.lazyFreeList,
    allocatePaged: cm.allocatePaged.bind(cm),
    cacheReq: cm.cacheReq.bind(cm),
    freeCache: cm.freeCache.bind(cm),
    beginLazyFree: cm.beginLazyFree.bind(cm),
    endLazyFree: cm.endLazyFree.bind(cm),
    checkIntegrity: cm.checkIntegrity.bind(cm),
  } as unknown as CacheManager;

  const adder1 = new PrefillAdder(100, wrappedCm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.strictEqual(result1, null);

  // 验证 lock 已释放 -> lockCount 应该为 0
  assert.strictEqual(lockCount, 0, "lock should be released after second check fail");

  // 再次尝试，应该同样能进入 lock 阶段
  const adder2 = new PrefillAdder(100, wrappedCm, tm, dm);
  const pr2 = new PendingReq({ rid: 2, inputIds: [4, 5, 6], samplingParams: sp });
  const result2 = adder2.tryAddOne(pr2);
  assert.strictEqual(result2, null, "should still fail but lock should be acquired and released");
  assert.strictEqual(lockCount, 0, "lock should be released again");
});

// ===== T24: PrefillAdder - tableManager 分配失败 =====
test("T24 PrefillAdder - tableManager allocate failure", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager(0); // 0 slots -> allocate will fail
  const dm = makeDecodeManager();

  const adder = new PrefillAdder(100, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.strictEqual(result, null);
  // 在 naive backend 下 lock 是 noop，但逻辑上应已正确处理
});

// ===== T25: PrefillAdder - 连续多次 tryAddOne 的一致性 =====
test("T25 PrefillAdder - consecutive tryAddOne consistency", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(10, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 10 });

  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const r1 = adder.tryAddOne(pr1);
  assert.ok(r1 instanceof Req);
  assert.strictEqual(adder.consumedTokens, 3);

  const pr2 = new PendingReq({ rid: 2, inputIds: [4, 5], samplingParams: sp });
  const r2 = adder.tryAddOne(pr2);
  assert.ok(r2 instanceof Req);
  assert.strictEqual(adder.consumedTokens, 5);

  const pr3 = new PendingReq({ rid: 3, inputIds: [6, 7, 8, 9, 10, 11], samplingParams: sp });
  const r3 = adder.tryAddOne(pr3);
  assert.ok(r3 instanceof ChunkedReq);
  assert.strictEqual(adder.consumedTokens, 10);
  assert.strictEqual(adder.remainingBudget, 0);
});

// ===== T26: PrefillManager - 多个 chunked 请求续接优先级 =====
test("T26 PrefillManager - multiple chunked requests continuation priority", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  // 两个请求都会被 chunked
  pm.addOneReq(makeMsg(1, [1, 2, 3, 4, 5]));
  pm.addOneReq(makeMsg(2, [6, 7, 8, 9, 10]));

  // Tick 1: budget=2，只有第一个请求会被部分处理（chunked）
  const batch1 = pm.scheduleNextBatch(2);
  assert.ok(batch1 !== null);
  assert.strictEqual(batch1!.reqs.size, 1);
  assert.ok(batch1!.reqs.get(1) instanceof ChunkedReq);

  // chunked 请求应在队列头部
  assert.strictEqual(pm.pendingList.length, 2);
  assert.ok(pm.pendingList[0].chunkedReq !== null);
  assert.strictEqual(pm.pendingList[0].rid, 1);
  assert.strictEqual(pm.pendingList[1].rid, 2);

  // Tick 2: 续接优先
  const batch2 = pm.scheduleNextBatch(100);
  assert.ok(batch2 !== null);
  // 续接完成 rid=1 + 新处理 rid=2
  assert.ok(batch2!.reqs.get(1) instanceof Req);
});

// ===== T27: DecodeManager.filterReqs - 空 newReqs =====
test("T27 DecodeManager.filterReqs - empty newReqs", () => {
  const dm = makeDecodeManager();
  const sp = new SamplingParams({ maxNewTokens: 1 });
  const req1 = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });

  dm.addReq(req1);
  // 消耗所有 remain
  req1.completeOne(); // remainLen=0, canDecode=false

  dm.filterReqs([]);
  assert.strictEqual(dm.runningReqs.size, 0);
});

// ===== T28: DecodeManager - pageSize=1 时 tokens_reserved=0 =====
test("T28 DecodeManager - pageSize=1 tokens_reserved=0", () => {
  const dm = makeDecodeManager(1);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });

  dm.addReq(req);
  // inflightTokens = 10 + (1-1)*1 = 10
  assert.strictEqual(dm.inflightTokens, 10);
});

// ===== T29: PrefillAdder._tryAddOneChunked - 续接时资源不足 =====
test("T29 PrefillAdder._tryAddOneChunked - resource insufficient on continuation", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 第一轮: chunked
  const adder1 = new PrefillAdder(2, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.ok(result1 instanceof ChunkedReq);

  // 第二轮: 极小容量
  const smallCm = new CacheManager(1, 1, Array.from({ length: 129 }, () => new Array(8192).fill(0)), "naive");
  const adder2 = new PrefillAdder(100, smallCm, tm, dm);
  const pr2 = new PendingReq({
    rid: 1,
    inputIds: [1, 2, 3, 4, 5],
    samplingParams: sp,
    chunkedReq: result1 as ChunkedReq,
  });
  const result2 = adder2.tryAddOne(pr2);
  // extendLen=3, availableSize=1 -> resource insufficient
  assert.strictEqual(result2, null);
});

// ===== T30: PrefillAdder - 全缓存命中（extendLen=0） =====
test("T30 PrefillAdder - full cache hit (extendLen=0)", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(100, cm, tm, dm);

  // NaivePrefixCache 总是 miss (cachedLen=0)，所以 extendLen=inputLen
  // 为了测试全缓存命中，需要模拟一个 cachedLen=inputLen 的 matchResult
  // 使用 wrapper
  const fakeHandle = new NaiveCacheHandle(3); // cachedLen=3
  const wrappedCm: CacheManager = Object.create(cm);
  wrappedCm.matchReq = function(_req: { inputIds: number[]; inputLen: number }) {
    return { cudaHandle: fakeHandle };
  };

  const adder2 = new PrefillAdder(100, wrappedCm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder2.tryAddOne(pr);

  // extendLen = 3 - 3 = 0, chunk_size = 0, is_chunked = false
  assert.ok(result instanceof Req, "should return Req for full cache hit");
  assert.strictEqual(adder2.consumedTokens, 0, "consumedTokens should not increase for full cache hit");
});

// ===== Boundary: B1 — PrefillAdder.tokenBudget = 0 =====
test("B1 PrefillAdder.tokenBudget = 0 -> all tryAddOne return null", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(0, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.strictEqual(result, null);
  assert.strictEqual(adder.remainingBudget, 0);
});

// ===== Boundary: B2 — PrefillAdder - extendLen = 0（全缓存命中） =====
test("B2 PrefillAdder - extendLen = 0 (full cache hit)", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  const fakeHandle = new NaiveCacheHandle(3);
  const wrappedCm: CacheManager = Object.create(cm);
  wrappedCm.matchReq = function() { return { cudaHandle: fakeHandle }; };

  const adder = new PrefillAdder(100, wrappedCm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.ok(result instanceof Req);
  assert.strictEqual(adder.consumedTokens, 0);
});

// ===== Boundary: B3 — PrefillAdder - tableManager.availableSize = 0 =====
test("B3 PrefillAdder - tableManager.availableSize = 0 -> first check fail", () => {
  const { cm } = makeCacheManager(1, 1);
  const tm = makeTableManager(0);
  const dm = makeDecodeManager();

  // 极小 CacheManager 使第一次 available_size 检查失败
  const adder = new PrefillAdder(100, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  // 需要 estimatedLen > availableSize
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  // 对于 naive: availableSize=1, estimatedLen=5, reservedSize=0
  // 5 + 0 > 1 -> return null
  const result = adder.tryAddOne(pr);
  assert.strictEqual(result, null);
});

// ===== Boundary: B4 — PrefillManager - 多个 chunked 请求续接 =====
test("B4 PrefillManager - multiple chunked requests continuation order", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const pm = new PrefillManager(cm, tm, dm);

  pm.addOneReq(makeMsg(1, [1, 2, 3, 4, 5]));
  pm.addOneReq(makeMsg(2, [6, 7, 8, 9, 10]));

  // budget=2 -> 第一个请求 chunked
  const batch1 = pm.scheduleNextBatch(2);
  assert.ok(batch1 !== null);

  // chunked 请求应优先于新请求
  assert.strictEqual(pm.pendingList[0].chunkedReq !== null, true);
});

// ===== Boundary: B5 — DecodeManager - pageSize=1 时 tokens_reserved=0 =====
test("B5 DecodeManager - pageSize=1 tokens_reserved=0", () => {
  const dm = makeDecodeManager(1);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });

  dm.addReq(req);
  assert.strictEqual(dm.inflightTokens, 10);
});

// ===== Boundary: B6 — DecodeManager.filterReqs - 空 newReqs =====
test("B6 DecodeManager.filterReqs - empty newReqs filters only existing", () => {
  const dm = makeDecodeManager();
  const sp1 = new SamplingParams({ maxNewTokens: 1 });
  const sp2 = new SamplingParams({ maxNewTokens: 10 });
  const req1 = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp1 });
  const req2 = new Req({ rid: 2, inputIds: [4, 5], samplingParams: sp2 });

  dm.addReq(req1);
  dm.addReq(req2);

  req1.completeOne(); // canDecode=false
  dm.filterReqs([]);

  assert.strictEqual(dm.runningReqs.size, 1);
  assert.ok(dm.runningReqs.has(req2));
});

// ===== Boundary: B7 — PrefillAdder._tryAddOneChunked - 续接时资源不足 =====
test("B7 PrefillAdder._tryAddOneChunked - resource insufficient on continuation", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 第一轮: chunked
  const adder1 = new PrefillAdder(1, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.ok(result1 instanceof ChunkedReq);

  // 极小容量的 CacheManager
  const smallCm = new CacheManager(1, 1, Array.from({ length: 129 }, () => new Array(8192).fill(0)), "naive");
  const adder2 = new PrefillAdder(100, smallCm, tm, dm);
  const pr2 = new PendingReq({
    rid: 1,
    inputIds: [1, 2, 3, 4, 5],
    samplingParams: sp,
    chunkedReq: result1 as ChunkedReq,
  });
  const result2 = adder2.tryAddOne(pr2);
  assert.strictEqual(result2, null);
});

// ===== Boundary: B8 — PrefillAdder - 第二次 available_size 检查失败后重试 =====
test("B8 PrefillAdder - retry after second available_size check fail", () => {
  let lockCount = 0;
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  const wrappedCm = {
    get availableSize() { return lockCount > 0 ? 0 : 1000; },
    matchReq: cm.matchReq.bind(cm),
    lock(_h: BaseCacheHandle) { lockCount++; cm.lock(_h); },
    unlock(h: BaseCacheHandle) { lockCount = Math.max(0, lockCount - 1); cm.unlock(h); },
    prefixCache: cm.prefixCache,
    pageTable: cm.pageTable,
    freeSlots: cm.freeSlots,
    pageSize: cm.pageSize,
    numPages: cm.numPages,
    lazyFreeList: cm.lazyFreeList,
    allocatePaged: cm.allocatePaged.bind(cm),
    cacheReq: cm.cacheReq.bind(cm),
    freeCache: cm.freeCache.bind(cm),
    beginLazyFree: cm.beginLazyFree.bind(cm),
    endLazyFree: cm.endLazyFree.bind(cm),
    checkIntegrity: cm.checkIntegrity.bind(cm),
  } as unknown as CacheManager;

  // 第一次尝试：第二次检查失败
  const adder1 = new PrefillAdder(100, wrappedCm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.strictEqual(result1, null);
  assert.strictEqual(lockCount, 0);

  // 第二次尝试：使用正常 CacheManager，应成功
  const adder2 = new PrefillAdder(100, cm, tm, dm);
  const pr2 = new PendingReq({ rid: 2, inputIds: [4, 5, 6], samplingParams: sp });
  const result2 = adder2.tryAddOne(pr2);
  assert.ok(result2 instanceof Req, "retry with normal CM should succeed");
});

// ===== Boundary: B9 — tableManager 分配耗尽 =====
test("B9 tableManager allocate exhausted", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager(1); // 仅 1 个 slot
  const dm = makeDecodeManager();

  // 第一个请求成功
  const adder1 = new PrefillAdder(100, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.ok(result1 instanceof Req);

  // 第二个请求: allocate 失败
  const adder2 = new PrefillAdder(100, cm, tm, dm);
  const pr2 = new PendingReq({ rid: 2, inputIds: [4, 5, 6], samplingParams: sp });
  const result2 = adder2.tryAddOne(pr2);
  assert.strictEqual(result2, null);
});

// ===== Boundary: B10 — DecodeManager.inflightTokens - 无 running 请求 =====
test("B10 DecodeManager.inflightTokens - no running requests", () => {
  const dm = makeDecodeManager();
  assert.strictEqual(dm.inflightTokens, 0);
});

// ===== 驳回修复验证: estimatedLen 包含 outputLen =====

// ===== R1: estimatedLen 包含 outputLen — tryAddOne 第一次检查 =====
test("R1 estimatedLen includes outputLen - first check in tryAddOne", () => {
  // 构造场景：extendLen 能通过 availableSize 检查，但 extendLen + outputLen 不能
  // extendLen=3, outputLen=20, availableSize=15
  // 3 <= 15 但 3+20=23 > 15 -> 应该 return null
  const { cm } = makeCacheManager(15, 1);
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(100, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 20 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);
  // estimatedLen = 3 + 20 = 23 > 15 -> null
  assert.strictEqual(result, null, "should reject when estimatedLen (extend+output) exceeds availableSize");
});

// ===== R2: estimatedLen 包含 outputLen — tryAddOne 通过场景 =====
test("R2 estimatedLen includes outputLen - tryAddOne passes when fits", () => {
  // extendLen=3, outputLen=10, availableSize=1024 -> 通过
  const { cm } = makeCacheManager(1024, 1);
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(100, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);
  // estimatedLen = 3 + 10 = 13 <= 1024 -> passes
  assert.ok(result instanceof Req, "should succeed when estimatedLen fits in availableSize");
});

// ===== R3: estimatedLen 包含 outputLen — _tryAddOneChunked 资源检查 =====
test("R3 estimatedLen includes outputLen - _tryAddOneChunked resource check", () => {
  // 第一轮: 创建 ChunkedReq
  const { cm } = makeCacheManager(1024, 1);
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder1 = new PrefillAdder(2, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 20 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.ok(result1 instanceof ChunkedReq);

  // 第二轮: 极小容量，extendLen=3, outputLen=20, estimatedLen=23 > 1
  const smallCm = new CacheManager(1, 1, Array.from({ length: 129 }, () => new Array(8192).fill(0)), "naive");
  const adder2 = new PrefillAdder(100, smallCm, tm, dm);
  const pr2 = new PendingReq({
    rid: 1,
    inputIds: [1, 2, 3, 4, 5],
    samplingParams: sp,
    chunkedReq: result1 as ChunkedReq,
  });
  const result2 = adder2.tryAddOne(pr2);
  assert.strictEqual(result2, null, "should reject chunked continuation when estimatedLen exceeds availableSize");
});

// ===== R4: maxDeviceLen 显式设置 — ChunkedReq =====
test("R4 maxDeviceLen explicitly set - ChunkedReq from tryAddOne", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(2, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.ok(result instanceof ChunkedReq);
  // cachedLen=0 (naive always miss), chunkSize=min(5,2)=2
  // maxDeviceLen 应为 cachedLen + chunkSize = 0 + 2 = 2
  assert.strictEqual((result as ChunkedReq).maxDeviceLen, 2,
    "ChunkedReq.maxDeviceLen should equal cachedLen + chunkSize");
  assert.strictEqual((result as ChunkedReq).deviceLen, 2);
  assert.strictEqual((result as ChunkedReq).remainLen, 0,
    "ChunkedReq.remainLen should be 0 since maxDeviceLen == deviceLen");
});

// ===== R5: maxDeviceLen 显式设置 — Req from tryAddOne =====
test("R5 maxDeviceLen explicitly set - Req from tryAddOne", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(100, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);

  assert.ok(result instanceof Req);
  // cachedLen=0, extendLen=3, outputLen=10
  // maxDeviceLen = cachedLen + extendLen + outputLen = 0 + 3 + 10 = 13
  assert.strictEqual((result as Req).maxDeviceLen, 13,
    "Req.maxDeviceLen should equal cachedLen + extendLen + outputLen");
  assert.strictEqual((result as Req).remainLen, 10,
    "Req.remainLen = maxDeviceLen - deviceLen = 13 - 3 = 10");
});

// ===== R6: maxDeviceLen 显式设置 — Req from _tryAddOneChunked =====
test("R6 maxDeviceLen explicitly set - Req from _tryAddOneChunked", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 第一轮: chunked
  const adder1 = new PrefillAdder(2, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.ok(result1 instanceof ChunkedReq);

  // 第二轮: 续接完成 -> Req
  const pr2 = new PendingReq({
    rid: 1,
    inputIds: [1, 2, 3, 4, 5],
    samplingParams: sp,
    chunkedReq: result1 as ChunkedReq,
  });
  const adder2 = new PrefillAdder(100, cm, tm, dm);
  const result2 = adder2.tryAddOne(pr2);

  assert.ok(result2 instanceof Req);
  // inputLen=5, outputLen=10 -> maxDeviceLen = 5 + 10 = 15
  assert.strictEqual((result2 as Req).maxDeviceLen, 15,
    "Req.maxDeviceLen from continuation should equal inputLen + outputLen");
  assert.strictEqual((result2 as Req).remainLen, 10,
    "Req.remainLen = 15 - 5 = 10");
});

// ===== R7: maxDeviceLen 显式设置 — ChunkedReq from _tryAddOneChunked =====
test("R7 maxDeviceLen explicitly set - ChunkedReq from _tryAddOneChunked", () => {
  const { cm } = makeCacheManager();
  const tm = makeTableManager();
  const dm = makeDecodeManager();

  // 第一轮: chunked (budget=1 -> chunkSize=1)
  const adder1 = new PrefillAdder(1, cm, tm, dm);
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const pr1 = new PendingReq({ rid: 1, inputIds: [1, 2, 3, 4, 5], samplingParams: sp });
  const result1 = adder1.tryAddOne(pr1);
  assert.ok(result1 instanceof ChunkedReq);
  assert.strictEqual((result1 as ChunkedReq).maxDeviceLen, 1);

  // 第二轮: 仍 chunked (budget=1 -> chunkSize=1, extendLen=4 -> still chunked)
  const pr2 = new PendingReq({
    rid: 1,
    inputIds: [1, 2, 3, 4, 5],
    samplingParams: sp,
    chunkedReq: result1 as ChunkedReq,
  });
  const adder2 = new PrefillAdder(1, cm, tm, dm);
  const result2 = adder2.tryAddOne(pr2);
  assert.ok(result2 instanceof ChunkedReq);
  // cachedLen=1 (from prevReq.deviceLen), chunkSize=1
  // maxDeviceLen = 1 + 1 = 2
  assert.strictEqual((result2 as ChunkedReq).maxDeviceLen, 2,
    "ChunkedReq.maxDeviceLen from continuation should equal cachedLen + chunkSize");
  assert.strictEqual((result2 as ChunkedReq).remainLen, 0);
});

// ===== R8: outputLen=0 时 estimatedLen == extendLen =====
test("R8 estimatedLen with outputLen=0 equals extendLen", () => {
  // outputLen=0 时 estimatedLen = extendLen + 0 = extendLen
  // 此时资源检查退化为原始逻辑
  const { cm } = makeCacheManager(5, 1);
  const tm = makeTableManager();
  const dm = makeDecodeManager();
  const adder = new PrefillAdder(100, cm, tm, dm);

  const sp = new SamplingParams({ maxNewTokens: 0 });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  const result = adder.tryAddOne(pr);
  // estimatedLen = 3 + 0 = 3 <= 5 -> passes
  assert.ok(result instanceof Req, "should succeed when outputLen=0 and extendLen fits");
  assert.strictEqual((result as Req).maxDeviceLen, 3,
    "maxDeviceLen = cachedLen + extendLen + 0 = 3");
});

// Summary
console.log("\n=== S2 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 S2 验收测试通过 \u2713");
  process.exit(0);
}
