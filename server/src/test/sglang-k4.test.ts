import assert from "assert";
import {
  BasePrefixCache,
  BaseCacheHandle,
  CacheSizeInfo,
  MatchResult,
  InsertResult,
  RadixTreeNode,
  RadixCacheHandle,
  RadixPrefixCache,
} from "../sglang";

import type { KeyFn } from "../sglang";

/**
 * Issue #15 验收测试 — K4: RadixPrefixCache（LRU 驱逐堆 + split_at + lock/unlock）(§9.8 / §3.3.5)
 *
 * Run with:  npx ts-node src/test/sglang-k4.test.ts
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

// Helper: page_size=1 keyFn
const keyFnPs1: KeyFn = (tokens: number[]) => tokens[0];

// ===== T1: RadixTreeNode 构造与基本属性 =====
test("T1 RadixTreeNode construction and basic properties", () => {
  const node = new RadixTreeNode(keyFnPs1, 42);
  assert.strictEqual(node.timestamp, 42);
  assert.strictEqual(node.refCount, 0);
  assert.strictEqual(node.children.size, 0);
  assert.strictEqual(node.parent, null);
  assert.strictEqual(node.length, 0);
  assert.deepStrictEqual(node.value, []);
  assert.strictEqual(node.keyFn, keyFnPs1);
});

// ===== T2: RadixTreeNode.setKeyValue =====
test("T2 RadixTreeNode.setKeyValue", () => {
  const node = new RadixTreeNode(keyFnPs1);
  node.setKeyValue([1, 2, 3], [10, 20, 30]);
  assert.strictEqual(node.length, 3);
  assert.deepStrictEqual(node.value, [10, 20, 30]);
});

// ===== T3: RadixTreeNode.getMatchLen =====
test("T3 RadixTreeNode.getMatchLen", () => {
  const node = new RadixTreeNode(keyFnPs1);
  node.setKeyValue([1, 2, 3, 4], [10, 20, 30, 40]);
  // 完全匹配
  assert.strictEqual(node.getMatchLen([1, 2, 3, 4]), 4);
  // 部分匹配
  assert.strictEqual(node.getMatchLen([1, 2, 5, 6]), 2);
  // 不匹配
  assert.strictEqual(node.getMatchLen([9, 9, 9]), 0);
  // input 更短
  assert.strictEqual(node.getMatchLen([1, 2]), 2);
});

// ===== T4: RadixTreeNode.splitAt 基本分裂 =====
test("T4 RadixTreeNode.splitAt basic split", () => {
  const parent = new RadixTreeNode(keyFnPs1, 0);
  const node = new RadixTreeNode(keyFnPs1, 10);
  node.setKeyValue([1, 2, 3, 4], [10, 20, 30, 40]);
  node.setParent(parent);

  const newNode = node.splitAt(2);
  // 新节点持有 [1,2] / [10,20]
  assert.strictEqual(newNode.length, 2);
  assert.deepStrictEqual(newNode.value, [10, 20]);
  // 原节点缩进到 [3,4] / [30,40]
  assert.strictEqual(node.length, 2);
  assert.deepStrictEqual(node.value, [30, 40]);
  // 原节点成为新节点的子节点
  assert.strictEqual(node.parent, newNode);
  // 新节点取代原节点在 parent 中的位置
  assert.strictEqual(newNode.parent, parent);
  assert.ok(parent.children.has(1)); // keyFn([1,2,3,4]) first key = 1 for page_size=1
});

// ===== T5: RadixTreeNode.splitAt refCount 继承 =====
test("T5 RadixTreeNode.splitAt refCount inheritance", () => {
  const parent = new RadixTreeNode(keyFnPs1, 0);
  const node = new RadixTreeNode(keyFnPs1, 10);
  node.setKeyValue([1, 2, 3], [10, 20, 30]);
  node.setParent(parent);
  node.refCount = 2;

  const newNode = node.splitAt(1);
  assert.strictEqual(newNode.refCount, 2);
});

// ===== T6: RadixTreeNode.setParent =====
test("T6 RadixTreeNode.setParent bidirectional reference", () => {
  const parent = new RadixTreeNode(keyFnPs1, 0);
  const child = new RadixTreeNode(keyFnPs1, 1);
  child.setKeyValue([5], [50]);
  child.setParent(parent);

  assert.strictEqual(child.parent, parent);
  assert.ok(parent.children.has(5)); // keyFn([5]) = 5
  assert.strictEqual(parent.children.get(5), child);
});

// ===== T7: RadixTreeNode.isLeaf/isRoot =====
test("T7 RadixTreeNode.isLeaf/isRoot", () => {
  const root = new RadixTreeNode(keyFnPs1, 0);
  assert.ok(root.isRoot());
  assert.ok(root.isLeaf());

  const child = new RadixTreeNode(keyFnPs1, 1);
  child.setKeyValue([1], [10]);
  child.setParent(root);
  assert.ok(!root.isLeaf());
  assert.ok(!child.isRoot());
  assert.ok(child.isLeaf());
});

// ===== T8: RadixPrefixCache 构造 =====
test("T8 RadixPrefixCache construction", () => {
  const cache = new RadixPrefixCache(100, 1);
  assert.strictEqual(cache.rootNode.refCount, 1);
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
  assert.strictEqual(cache.sizeInfo.protectedSize, 0);
  assert.strictEqual(cache.sizeInfo.totalSize, 0);
});

// ===== T9: matchPrefix 完全命中 =====
test("T9 matchPrefix full hit", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  const result = cache.matchPrefix([1, 2, 3]);
  assert.strictEqual(result.cudaHandle.cachedLen, 3);
});

// ===== T10: matchPrefix 部分命中 =====
test("T10 matchPrefix partial hit", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  const result = cache.matchPrefix([1, 2, 4, 5]);
  assert.strictEqual(result.cudaHandle.cachedLen, 2);
});

// ===== T11: matchPrefix 未命中 =====
test("T11 matchPrefix miss", () => {
  const cache = new RadixPrefixCache(100, 1);
  const result = cache.matchPrefix([1, 2, 3]);
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
});

// ===== T12: insertPrefix 新序列插入 =====
test("T12 insertPrefix new sequence", () => {
  const cache = new RadixPrefixCache(100, 1);
  const result = cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  assert.strictEqual(result.cachedLen, 0); // 插入前无缓存
  assert.strictEqual(result.cudaHandle.cachedLen, 3); // 插入后总缓存
  assert.strictEqual(cache.sizeInfo.evictableSize, 3);
});

// ===== T13: insertPrefix 已存在序列 =====
test("T13 insertPrefix existing sequence", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  const result = cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  assert.strictEqual(result.cachedLen, 3); // 已全部缓存
  assert.strictEqual(result.cudaHandle.cachedLen, 3);
  // sizeInfo 不应增长（已有节点不再增加 evictableSize）
  assert.strictEqual(cache.sizeInfo.evictableSize, 3);
});

// ===== T14: 同一前缀重放命中率 100% =====
test("T14 same prefix replay 100% hit rate", () => {
  const cache = new RadixPrefixCache(100, 1);
  const tokens = [1, 2, 3, 4, 5];
  const indices = [10, 20, 30, 40, 50];
  cache.insertPrefix(tokens, indices);

  // 多次 match 应该全部命中
  for (let i = 0; i < 5; i++) {
    const mr = cache.matchPrefix([...tokens]);
    assert.strictEqual(mr.cudaHandle.cachedLen, 5, `iteration ${i}: cachedLen should be 5`);
  }
});

// ===== T15: split_at 正确性 =====
test("T15 split_at correctness", () => {
  const cache = new RadixPrefixCache(100, 1);
  // 插入 [1,2,3,4,5]
  cache.insertPrefix([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]);
  // 插入 [1,2,6,7] — 应触发 [1,2,3,4,5] 的 split_at(2)
  cache.insertPrefix([1, 2, 6, 7], [10, 20, 60, 70]);

  // 现在 [1,2] 是共享前缀节点
  const mr1 = cache.matchPrefix([1, 2, 3, 4, 5]);
  assert.strictEqual(mr1.cudaHandle.cachedLen, 5);

  const mr2 = cache.matchPrefix([1, 2, 6, 7]);
  assert.strictEqual(mr2.cudaHandle.cachedLen, 4);

  const mr3 = cache.matchPrefix([1, 2, 8, 9]);
  assert.strictEqual(mr3.cudaHandle.cachedLen, 2); // 只匹配到共享前缀 [1,2]
});

// ===== T16: lockHandle 锁定 =====
test("T16 lockHandle lock", () => {
  const cache = new RadixPrefixCache(100, 1);
  const ir = cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.lockHandle(ir.cudaHandle);

  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
  assert.strictEqual(cache.sizeInfo.protectedSize, 3);
  assert.strictEqual((ir.cudaHandle as RadixCacheHandle).node.refCount, 1);
});

// ===== T17: lockHandle 解锁 =====
test("T17 lockHandle unlock", () => {
  const cache = new RadixPrefixCache(100, 1);
  const ir = cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.lockHandle(ir.cudaHandle);
  cache.lockHandle(ir.cudaHandle, true);

  assert.strictEqual(cache.sizeInfo.evictableSize, 3);
  assert.strictEqual(cache.sizeInfo.protectedSize, 0);
  assert.strictEqual((ir.cudaHandle as RadixCacheHandle).node.refCount, 0);
});

// ===== T18: lockHandle 多次锁定 =====
test("T18 lockHandle multiple locks", () => {
  const cache = new RadixPrefixCache(100, 1);
  const ir = cache.insertPrefix([1, 2, 3], [10, 20, 30]);

  // 第一次 lock: 0→1, evictable→protected
  cache.lockHandle(ir.cudaHandle);
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
  assert.strictEqual(cache.sizeInfo.protectedSize, 3);

  // 第二次 lock: 1→2, sizeInfo 不变（已经 protected）
  cache.lockHandle(ir.cudaHandle);
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
  assert.strictEqual(cache.sizeInfo.protectedSize, 3);
  assert.strictEqual((ir.cudaHandle as RadixCacheHandle).node.refCount, 2);

  // 第一次 unlock: 2→1, 仍为 protected
  cache.lockHandle(ir.cudaHandle, true);
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
  assert.strictEqual(cache.sizeInfo.protectedSize, 3);

  // 第二次 unlock: 1→0, protected→evictable
  cache.lockHandle(ir.cudaHandle, true);
  assert.strictEqual(cache.sizeInfo.evictableSize, 3);
  assert.strictEqual(cache.sizeInfo.protectedSize, 0);
});

// ===== T19: evict 基本驱逐 =====
test("T19 evict basic eviction", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  assert.strictEqual(cache.sizeInfo.evictableSize, 3);

  const evicted = cache.evict(3);
  assert.strictEqual(evicted.length, 3);
  assert.deepStrictEqual(evicted, [10, 20, 30]);
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
});

// ===== T20: evict 驱逐不影响命中节点 =====
test("T20 evict does not affect locked nodes", () => {
  const cache = new RadixPrefixCache(100, 1);
  const ir = cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.lockHandle(ir.cudaHandle); // 锁定

  const evicted = cache.evict(3);
  assert.strictEqual(evicted.length, 0); // 无法驱逐
  assert.strictEqual(cache.sizeInfo.protectedSize, 3);
});

// ===== T21: evict 父节点合并 =====
test("T21 evict parent merge after child eviction", () => {
  const cache = new RadixPrefixCache(100, 1);
  // 插入两个共享前缀的序列
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.insertPrefix([1, 2, 4], [10, 20, 40]);

  // 树结构: root -> [1,2] -> [3], [4]
  // evictableSize 应为 4 ([1,2]=2 + [3]=1 + [4]=1)
  assert.strictEqual(cache.sizeInfo.evictableSize, 4);

  // 驱逐叶子 [3]（timestamp 更早）
  const evicted = cache.evict(1);
  assert.strictEqual(evicted.length, 1);
  assert.deepStrictEqual(evicted, [30]);

  // [1,2] 仍存在，因为 [4] 仍是它的子节点
  const mr = cache.matchPrefix([1, 2, 4]);
  assert.strictEqual(mr.cudaHandle.cachedLen, 3);

  // 驱逐 [4] 后，[1,2] 变叶子且可驱逐
  const evicted2 = cache.evict(10);
  assert.ok(evicted2.length >= 2); // [4] + [1,2]
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
});

// ===== T22: handle lock 阻止驱逐 =====
test("T22 handle lock prevents eviction", () => {
  const cache = new RadixPrefixCache(100, 1);
  const ir = cache.insertPrefix([5, 6, 7], [50, 60, 70]);
  cache.lockHandle(ir.cudaHandle);

  // 多次 evict 尝试
  for (let i = 0; i < 3; i++) {
    const evicted = cache.evict(10);
    assert.strictEqual(evicted.length, 0);
  }
  assert.strictEqual(cache.sizeInfo.protectedSize, 3);
});

// ===== T23: sizeInfo 一致性 =====
test("T23 sizeInfo consistency", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.insertPrefix([1, 2, 4], [10, 20, 40]);

  const ir = cache.insertPrefix([5, 6], [50, 60]);
  cache.lockHandle(ir.cudaHandle);

  // totalSize 应等于树中非 root 节点 token 总数
  const total = cache.sizeInfo.totalSize;
  assert.ok(total > 0);
  assert.strictEqual(total, cache.sizeInfo.evictableSize + cache.sizeInfo.protectedSize);
});

// ===== T24: RadixCacheHandle.getMatchedIndices =====
test("T24 RadixCacheHandle.getMatchedIndices", () => {
  const cache = new RadixPrefixCache(100, 1);
  // 插入共享前缀序列
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.insertPrefix([1, 2, 4], [10, 20, 40]);

  // match [1,2,3] 应沿 root -> [1,2] -> [3] 路径拼合 indices
  const mr = cache.matchPrefix([1, 2, 3]);
  const indices = mr.cudaHandle.getMatchedIndices();
  // [1,2] 的 value = [10,20], [3] 的 value = [30]
  assert.ok(indices.length >= 3);
  assert.strictEqual(indices[0], 10);
  assert.strictEqual(indices[1], 20);
  assert.strictEqual(indices[2], 30);
});

// ===== T25: reset 重置缓存 =====
test("T25 reset clears cache", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.reset();
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
  assert.strictEqual(cache.sizeInfo.protectedSize, 0);
  assert.strictEqual(cache.sizeInfo.totalSize, 0);

  // 匹配应返回 0
  const mr = cache.matchPrefix([1, 2, 3]);
  assert.strictEqual(mr.cudaHandle.cachedLen, 0);
});

// ===== T26: checkIntegrity 正常 =====
test("T26 checkIntegrity normal", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  const ir = cache.insertPrefix([4, 5], [40, 50]);
  cache.lockHandle(ir.cudaHandle);
  // Should not throw
  cache.checkIntegrity();
});

// ===== T27: checkIntegrity 检测不一致 =====
test("T27 checkIntegrity detects inconsistency", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  // 手动破坏 sizeInfo
  cache.sizeInfo.evictableSize = 999;
  assert.throws(
    () => cache.checkIntegrity(),
    /Integrity check failed/
  );
});

// ===== T28: pageSize=1 的 keyFn =====
test("T28 pageSize=1 keyFn uses number keys", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([42], [100]);
  // root 的 children 应该有 key=42 (number 类型)
  const root = cache.rootNode;
  assert.ok(root.children.has(42));
  assert.strictEqual(typeof Array.from(root.children.keys())[0], "number");
});

// ===== T29: pageSize>1 的 keyFn =====
test("T29 pageSize>1 keyFn uses string keys", () => {
  const cache = new RadixPrefixCache(100, 2);
  cache.insertPrefix([1, 2, 3, 4], [10, 20, 30, 40]);
  // root 的 children 应该有 key="[1,2]" (string 类型)
  const root = cache.rootNode;
  assert.ok(root.children.has("[1,2]"));
  assert.strictEqual(typeof Array.from(root.children.keys())[0], "string");

  // 匹配测试（页对齐）
  const mr = cache.matchPrefix([1, 2, 3, 4]);
  assert.strictEqual(mr.cudaHandle.cachedLen, 4);

  // 非页对齐匹配
  const mr2 = cache.matchPrefix([1, 2, 3, 5]);
  assert.strictEqual(mr2.cudaHandle.cachedLen, 2); // 只匹配 [1,2]，[3,5] 不匹配
});

// ===== T30: insertPrefix 空序列 =====
test("T30 insertPrefix empty sequence", () => {
  const cache = new RadixPrefixCache(100, 1);
  const result = cache.insertPrefix([], []);
  assert.strictEqual(result.cachedLen, 0);
  assert.strictEqual(result.cudaHandle.cachedLen, 0);
});

// ===== Boundary: B1 — 空树 matchPrefix =====
test("B1 empty tree matchPrefix returns 0", () => {
  const cache = new RadixPrefixCache(100, 1);
  const mr = cache.matchPrefix([1, 2, 3]);
  assert.strictEqual(mr.cudaHandle.cachedLen, 0);
});

// ===== Boundary: B2 — 单 token 插入与匹配 =====
test("B2 single token insert and match (pageSize=1)", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([7], [70]);
  const mr = cache.matchPrefix([7]);
  assert.strictEqual(mr.cudaHandle.cachedLen, 1);
});

// ===== Boundary: B3 — evict 请求量大于可驱逐量 =====
test("B3 evict more than available only evicts what's possible", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2], [10, 20]);
  const evicted = cache.evict(100);
  assert.strictEqual(evicted.length, 2);
  assert.strictEqual(cache.sizeInfo.evictableSize, 0);
});

// ===== Boundary: B4 — evict(0) =====
test("B4 evict(0) returns empty array", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  const evicted = cache.evict(0);
  assert.deepStrictEqual(evicted, []);
  assert.strictEqual(cache.sizeInfo.evictableSize, 3);
});

// ===== Boundary: B5 — 连续 lock/unlock 配对 =====
test("B5 consecutive lock/unlock pairs", () => {
  const cache = new RadixPrefixCache(100, 1);
  const ir = cache.insertPrefix([1, 2], [10, 20]);

  for (let i = 0; i < 5; i++) {
    cache.lockHandle(ir.cudaHandle);
    cache.lockHandle(ir.cudaHandle, true);
  }
  assert.strictEqual((ir.cudaHandle as RadixCacheHandle).node.refCount, 0);
  assert.strictEqual(cache.sizeInfo.evictableSize, 2);
  assert.strictEqual(cache.sizeInfo.protectedSize, 0);
});

// ===== Boundary: B6 — 未 unlock 就再次 lock =====
test("B6 lock without intermediate unlock", () => {
  const cache = new RadixPrefixCache(100, 1);
  const ir = cache.insertPrefix([1], [10]);

  cache.lockHandle(ir.cudaHandle);
  cache.lockHandle(ir.cudaHandle);
  cache.lockHandle(ir.cudaHandle);
  assert.strictEqual((ir.cudaHandle as RadixCacheHandle).node.refCount, 3);
  assert.strictEqual(cache.sizeInfo.protectedSize, 1);

  // 逐步 unlock
  cache.lockHandle(ir.cudaHandle, true);
  assert.strictEqual((ir.cudaHandle as RadixCacheHandle).node.refCount, 2);
  assert.strictEqual(cache.sizeInfo.protectedSize, 1); // 仍为 protected
});

// ===== Boundary: B7 — 插入长度非页对齐 =====
test("B7 insertPrefix non-page-aligned length", () => {
  const cache = new RadixPrefixCache(100, 2);
  // 插入 3 个 token，pageSize=2，insertLen = alignDown(3, 2) = 2
  const result = cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  assert.strictEqual(result.cudaHandle.cachedLen, 2); // 只插入了 2 个
  assert.strictEqual(cache.sizeInfo.evictableSize, 2);
});

// ===== Boundary: B8 — 完全相同序列重复插入 =====
test("B8 identical sequence repeated insert no new node", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  const sizeBefore = cache.sizeInfo.evictableSize;
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  assert.strictEqual(cache.sizeInfo.evictableSize, sizeBefore);
});

// ===== Boundary: B9 — 多条共享前缀的序列 =====
test("B9 multiple sequences sharing prefix", () => {
  const cache = new RadixPrefixCache(100, 1);
  cache.insertPrefix([1, 2, 3], [10, 20, 30]);
  cache.insertPrefix([1, 2, 4], [10, 20, 40]);
  cache.insertPrefix([1, 2, 5], [10, 20, 50]);

  // [1,2] 是共享前缀
  const mr = cache.matchPrefix([1, 2]);
  assert.strictEqual(mr.cudaHandle.cachedLen, 2);

  // 各分支都能匹配
  assert.strictEqual(cache.matchPrefix([1, 2, 3]).cudaHandle.cachedLen, 3);
  assert.strictEqual(cache.matchPrefix([1, 2, 4]).cudaHandle.cachedLen, 3);
  assert.strictEqual(cache.matchPrefix([1, 2, 5]).cudaHandle.cachedLen, 3);

  // 不存在的分支只匹配到前缀
  assert.strictEqual(cache.matchPrefix([1, 2, 9]).cudaHandle.cachedLen, 2);
});

// ===== Type hierarchy verification =====
test("T_extra RadixPrefixCache extends BasePrefixCache", () => {
  const cache = new RadixPrefixCache(100, 1);
  assert.ok(cache instanceof BasePrefixCache);
});

test("T_extra RadixCacheHandle extends BaseCacheHandle", () => {
  const node = new RadixTreeNode(keyFnPs1);
  const handle = new RadixCacheHandle(0, node);
  assert.ok(handle instanceof BaseCacheHandle);
});

// Summary
console.log("\n=== K4 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 K4 验收测试通过 \u2713");
  process.exit(0);
}
