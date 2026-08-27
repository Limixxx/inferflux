import assert from "assert";
import {
  SamplingParams,
  Req,
  Batch,
  ChunkedReq,
  PendingReq,
  alignDown,
  divCeil,
  divEven,
  bytesPerElement,
} from "../sglang";

/**
 * Issue #10 验收测试 — S1: 核心数据结构 SamplingParams/Req/ChunkedReq/Batch/PendingReq + 工具函数
 *
 * Run with:  npx ts-node src/test/sglang-s1.test.ts
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

// ===== T1: SamplingParams 默认值 =====
test("T1 SamplingParams default values", () => {
  const sp = new SamplingParams();
  assert.strictEqual(sp.maxNewTokens, 1024);
  assert.strictEqual(sp.temperature, 0.0);
  assert.strictEqual(sp.topP, 1.0);
  assert.strictEqual(sp.topK, -1);
  assert.strictEqual(sp.frequencyPenalty, 0.0);
  assert.strictEqual(sp.repetitionPenalty, 1.0);
  assert.strictEqual(sp.minP, 0.0);
  assert.deepStrictEqual(sp.stopTokenIds, []);
  assert.strictEqual(sp.skipSpecialTokens, true);
  assert.strictEqual(sp.dtype, "float16");
});

// ===== T2: SamplingParams.isGreedy — greedy 场景 =====
test("T2 SamplingParams.isGreedy greedy cases", () => {
  // temperature ≤ 0 且 topP = 1.0
  assert.strictEqual(new SamplingParams({ temperature: 0.0 }).isGreedy, true);
  assert.strictEqual(new SamplingParams({ temperature: -0.5 }).isGreedy, true);
  // topK === 1 且 topP = 1.0
  assert.strictEqual(new SamplingParams({ topK: 1 }).isGreedy, true);
  // temperature=0 且 topK=1
  assert.strictEqual(new SamplingParams({ temperature: 0.0, topK: 1 }).isGreedy, true);
});

// ===== T3: SamplingParams.isGreedy — 非 greedy 场景 =====
test("T3 SamplingParams.isGreedy non-greedy cases", () => {
  // temperature > 0 且 topP ≠ 1.0
  assert.strictEqual(new SamplingParams({ temperature: 0.5, topP: 0.9 }).isGreedy, false);
  // temperature > 0, topP = 1.0
  assert.strictEqual(new SamplingParams({ temperature: 1.0 }).isGreedy, false);
  // temperature = 0, topP ≠ 1.0
  assert.strictEqual(new SamplingParams({ temperature: 0.0, topP: 0.9 }).isGreedy, false);
});

// ===== T4: Req 构造 =====
test("T4 Req construction", () => {
  const sp = new SamplingParams({ maxNewTokens: 50 });
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  assert.strictEqual(req.rid, 1);
  assert.deepStrictEqual(req.inputIds, [1, 2, 3]);
  assert.strictEqual(req.originInputLen, 3);
  assert.strictEqual(req.samplingParams, sp);
  assert.strictEqual(req.finished, false);
  assert.strictEqual(req.finishReason, null);
  assert.strictEqual(req.samplingCounter, 0);
  assert.strictEqual(req.maxNewTokens, 50);
  assert.strictEqual(req.dpRank, 0);
  assert.strictEqual(req.deviceLen, 3);
  assert.strictEqual(req.maxDeviceLen, 53);
  assert.deepStrictEqual(req.outputIds, []);
});

// ===== T5: Req.completeOne =====
test("T5 Req.completeOne increments deviceLen and samplingCounter", () => {
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  assert.strictEqual(req.deviceLen, 3);
  assert.strictEqual(req.samplingCounter, 0);
  req.completeOne();
  assert.strictEqual(req.deviceLen, 4);
  assert.strictEqual(req.samplingCounter, 1);
  req.completeOne();
  assert.strictEqual(req.deviceLen, 5);
  assert.strictEqual(req.samplingCounter, 2);
});

// ===== T6: Req.appendHost =====
test("T6 Req.appendHost appends token to inputIds and outputIds", () => {
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  req.appendHost(42);
  assert.deepStrictEqual(req.inputIds, [1, 2, 3, 42]);
  assert.deepStrictEqual(req.outputIds, [42]);
  req.appendHost(99);
  assert.deepStrictEqual(req.inputIds, [1, 2, 3, 42, 99]);
  assert.deepStrictEqual(req.outputIds, [42, 99]);
});

// ===== T7: Req.canDecode =====
test("T7 Req.canDecode remainLen>0 returns true, =0 returns false", () => {
  const sp = new SamplingParams({ maxNewTokens: 2 });
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  assert.strictEqual(req.remainLen, 2);
  assert.strictEqual(req.canDecode, true);
  req.completeOne();
  assert.strictEqual(req.remainLen, 1);
  assert.strictEqual(req.canDecode, true);
  req.completeOne();
  assert.strictEqual(req.remainLen, 0);
  assert.strictEqual(req.canDecode, false);
});

// ===== T8: ChunkedReq.canDecode =====
test("T8 ChunkedReq.canDecode always returns false", () => {
  const sp = new SamplingParams({ maxNewTokens: 100 });
  const creq = new ChunkedReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  assert.strictEqual(creq.canDecode, false);
  assert.strictEqual(creq.remainLen > 0, true); // still has remaining length logically
});

// ===== T9: ChunkedReq.appendHost =====
test("T9 ChunkedReq.appendHost throws error", () => {
  const sp = new SamplingParams({ maxNewTokens: 100 });
  const creq = new ChunkedReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp });
  assert.throws(() => creq.appendHost(42), /ChunkedReq should not be sampled/);
});

// ===== T10: Batch 构造与 reqs 管理 =====
test("T10 Batch construction and reqs management", () => {
  const batch = new Batch();
  assert.strictEqual(batch.reqs.size, 0);
  assert.strictEqual(batch.initLen, 0);
  assert.strictEqual(batch.promptTokens, 0);
  assert.strictEqual(batch.extendInputTokens, 0);
  assert.strictEqual(batch.extendOutputTokens, 0);
  assert.strictEqual(batch.numDecodeTokens, 0);
  assert.strictEqual(batch.hasIdleReqs, false);
  assert.deepStrictEqual(batch.readyIds, []);
  assert.strictEqual(batch.nextId, 0);
  assert.strictEqual(batch.schedulerThinkingBatch, false);

  const sp = new SamplingParams();
  const req1 = new Req({ rid: 1, inputIds: [1], samplingParams: sp });
  const req2 = new Req({ rid: 2, inputIds: [2], samplingParams: sp });
  batch.reqs.set(1, req1);
  batch.reqs.set(2, req2);
  batch.readyIds = [1, 2];
  assert.strictEqual(batch.reqs.size, 2);
  assert.strictEqual(batch.reqs.get(1)!.rid, 1);
  assert.strictEqual(batch.reqs.get(2)!.rid, 2);
});

// ===== T11: Batch.nextReadyReq/nextBatchReq =====
test("T11 Batch.nextReadyReq/nextBatchReq", () => {
  const batch = new Batch();
  const sp = new SamplingParams();
  const req1 = new Req({ rid: 1, inputIds: [1], samplingParams: sp });
  const req2 = new Req({ rid: 2, inputIds: [2], samplingParams: sp });
  batch.reqs.set(1, req1);
  batch.reqs.set(2, req2);
  batch.readyIds = [1, 2];

  // nextBatchReq 不移除
  const peeked = batch.nextBatchReq();
  assert.strictEqual(peeked!.rid, 1);
  assert.strictEqual(batch.readyIds.length, 2); // still 2

  // nextReadyReq 移除
  const taken = batch.nextReadyReq();
  assert.strictEqual(taken!.rid, 1);
  assert.strictEqual(batch.readyIds.length, 1);

  const taken2 = batch.nextReadyReq();
  assert.strictEqual(taken2!.rid, 2);
  assert.strictEqual(batch.readyIds.length, 0);
});

// ===== T12: PendingReq 构造 =====
test("T12 PendingReq construction", () => {
  const sp = new SamplingParams({ maxNewTokens: 100 });
  const pr = new PendingReq({ rid: 5, inputIds: [1, 2, 3, 4], samplingParams: sp });
  assert.strictEqual(pr.rid, 5);
  assert.deepStrictEqual(pr.inputIds, [1, 2, 3, 4]);
  assert.strictEqual(pr.samplingParams, sp);
  assert.strictEqual(pr.priority, 0);
  assert.strictEqual(pr.nextScheduledTime, 0);
  assert.strictEqual(pr.chunkedReq, null);
  assert.strictEqual(pr.inputLen, 4);
  assert.strictEqual(pr.outputLen, 100);
});

// ===== T13: PendingReq.chunkedReq 续接 =====
test("T13 PendingReq.chunkedReq continuation", () => {
  const sp = new SamplingParams({ maxNewTokens: 100 });
  const creq = new ChunkedReq({ rid: 1, inputIds: [1, 2], samplingParams: sp });
  const pr = new PendingReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: sp, chunkedReq: creq });
  assert.ok(pr.chunkedReq !== null);
  assert.strictEqual(pr.chunkedReq!.rid, 1);
});

// ===== T14: alignDown =====
test("T14 alignDown", () => {
  assert.strictEqual(alignDown(10, 3), 9);
  assert.strictEqual(alignDown(0, 5), 0);
  assert.strictEqual(alignDown(7, 1), 7);
  assert.strictEqual(alignDown(15, 4), 12);
  assert.strictEqual(alignDown(16, 4), 16);
});

// ===== T15: divCeil =====
test("T15 divCeil", () => {
  assert.strictEqual(divCeil(7, 3), 3);
  assert.strictEqual(divCeil(6, 3), 2);
  assert.strictEqual(divCeil(0, 5), 0);
  assert.strictEqual(divCeil(1, 1), 1);
  assert.strictEqual(divCeil(5, 2), 3);
});

// ===== T16: divEven 均分 =====
test("T16 divEven even distribution", () => {
  assert.deepStrictEqual(divEven(8, 3), [3, 3, 2]);
  assert.deepStrictEqual(divEven(6, 3), [2, 2, 2]);
  assert.deepStrictEqual(divEven(10, 4), [3, 3, 2, 2]);
  assert.deepStrictEqual(divEven(1, 1), [1]);
});

// ===== T17: divEven allowReplicate =====
test("T17 divEven allowReplicate", () => {
  assert.deepStrictEqual(divEven(2, 4, true), [1, 1, 0, 0]);
  assert.deepStrictEqual(divEven(3, 5, true), [1, 1, 1, 0, 0]);
});

// ===== T18: divEven 禁止复制时抛错 =====
test("T18 divEven throws when a < b and allowReplicate=false", () => {
  assert.throws(() => divEven(2, 4, false), /divEven\(2, 4\)/);
  assert.throws(() => divEven(2, 4), /divEven\(2, 4\)/);
});

// ===== T19: bytesPerElement =====
test("T19 bytesPerElement", () => {
  assert.strictEqual(bytesPerElement("float32"), 4);
  assert.strictEqual(bytesPerElement("float16"), 2);
  assert.strictEqual(bytesPerElement("bfloat16"), 2);
});

// ===== Boundary: B1 — alignDown(0, n) =====
test("B1 alignDown(0, n) returns 0", () => {
  assert.strictEqual(alignDown(0, 5), 0);
  assert.strictEqual(alignDown(0, 1), 0);
});

// ===== Boundary: B2 — divCeil(0, n) =====
test("B2 divCeil(0, n) returns 0", () => {
  assert.strictEqual(divCeil(0, 5), 0);
  assert.strictEqual(divCeil(0, 1), 0);
});

// ===== Boundary: B3 — divEven(0, n) =====
test("B3 divEven(0, n) returns [0]*n", () => {
  assert.deepStrictEqual(divEven(0, 3), [0, 0, 0]);
  assert.deepStrictEqual(divEven(0, 1), [0]);
});

// ===== Boundary: B4 — SamplingParams 全部默认构造 =====
test("B4 SamplingParams all-default construction is greedy", () => {
  const sp = new SamplingParams();
  assert.strictEqual(sp.isGreedy, true);
});

// ===== Boundary: B5 — Req 空 inputIds =====
test("B5 Req with empty inputIds", () => {
  const sp = new SamplingParams({ maxNewTokens: 5 });
  const req = new Req({ rid: 1, inputIds: [], samplingParams: sp });
  assert.strictEqual(req.deviceLen, 0);
  assert.strictEqual(req.maxDeviceLen, 5);
  assert.strictEqual(req.remainLen, 5);
});

// ===== Boundary: B6 — PendingReq 无 chunkedReq =====
test("B6 PendingReq without chunkedReq is null", () => {
  const sp = new SamplingParams();
  const pr = new PendingReq({ rid: 1, inputIds: [1], samplingParams: sp });
  assert.strictEqual(pr.chunkedReq, null);
});

// ===== Boundary: B7 — Batch 无 readyIds =====
test("B7 Batch with empty readyIds returns undefined", () => {
  const batch = new Batch();
  assert.strictEqual(batch.nextReadyReq(), undefined);
  assert.strictEqual(batch.nextBatchReq(), undefined);
});

// Summary
console.log("\n=== S1 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 S1 验收测试通过 \u2713");
  process.exit(0);
}
