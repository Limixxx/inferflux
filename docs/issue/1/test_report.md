---
issue_number: 1
issue_type: Feature
test_date: 2026-08-20
test_result: pass
---

# Issue #1 测试报告

## 验收测试结果

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | 默认启动模式为 pd-disagg，pList 非空、wList 为空 | ✅ pass |
| T2 | 切换为 agg 模式后 wList 创建（numWorkers 个），pList/dList 为空 | ✅ pass |
| T3 | agg 非分块 prefill 仿真：请求完成，TTFT/E2E 为正 | ✅ pass |
| T4 | agg 分块 prefill 仿真：chunkedPrefill=true 时请求完成 | ✅ pass |
| T5 | agg KV 容量约束：kvUsed 永不超 maxTokens | ✅ pass |
| T7 | agg breakdown 为 4 列（tokenize/queue/prefill/detok） | ✅ pass |
| T9 | agg gauges：wQueue/kvW 正确，pd-disagg 专用 gauge 置零 | ✅ pass |
| T10 | RadixCache 前缀复用：uncachedLen ≤ inputLen，inputLen=cachedLen+uncachedLen | ✅ pass |
| T11 | BlockManager 预分配：准入后 kvUsed 含完整 inputLen | ✅ pass |
| T12 | make_batch 混合批处理：running 中可混合 prefill/decode | ✅ pass |
| T13 | 切换回 pd-disagg：wList 清空，pList/dList 重建 | ✅ pass |
| T14 | agg 预设加载：4 个 agg 预设 mode=agg，pd-disagg 预设不变 | ✅ pass |
| T15 | agg getRenderState：wList 序列化正确，含 id/waitingQ/running/kvUsed/maxTokens | ✅ pass |
| B1 | 边界：numWorkers=1 单 worker 正常工作 | ✅ pass |
| B3 | 边界：kvGb=1 极小显存饱和不崩溃，kvUsed 仍受限 | ✅ pass |
| B9 | 边界：cacheHitRate=1.0 时 uncachedLen 接近 0（≥1 因 inputLen-1 上限） | ✅ pass |

## 类型检查

- 结果: **pass**
- 命令: `node node_modules/typescript/bin/tsc --noEmit`
- 说明: 含新增 `src/test/agg.test.ts` 在内的全部 TypeScript 源码在 strict 模式下编译通过，零错误零警告。

## 测试运行方式

```bash
cd server
npx ts-node src/test/agg.test.ts
```

测试使用 Node.js 内置 `assert` 模块，无外部测试框架依赖。共 16 个用例全部通过。

## 失败用例详情

无。

## 边界条件覆盖

- B1: `numWorkers=1` 单 worker 场景 — 所有请求进同一 worker，正常完成
- B3: `kvGb=1` 极小 KV 预算 — KV 快速饱和，大量排队/retract，但引擎不崩溃，kvUsed 不超限
- B9: `cacheHitRate=1.0` 全命中 — uncachedLen 趋近 0（因 `cachedLen = Math.min(inputLen-1, ...)` 上限，恒 ≥1），prefill 计算量极小
- 模式切换（agg↔pd-disagg）— reset() 正确清空状态、重建拓扑
- 分块 vs 非分块 prefill — 两种子模式均正确完成请求生命周期
