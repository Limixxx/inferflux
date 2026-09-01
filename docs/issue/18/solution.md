---
title: "Issue #18 解决方案"
issue_number: 18
issue_type: Feature
created: 2026-09-01
updated: 2026-09-01
status: draft
review_round: 1
---

# Issue #18 解决方案

## 需求分析

- **问题描述**：Issue #18 要求实现 SGLang 仿真器 S4 阶段的 `SimGraphRunner` 组件，包含 CUDA Graph 的 batch size 分桶策略、`pad_batch` 补 padding 请求至分桶边界、`graph_replay_cost_ticks` 与 `eager_forward_cost_ticks` 成本模型，以及与 `MockEngine.forward_batch` 的集成（当 `can_use_cuda_graph=True` 时走 graph_replay_cost 路径，否则走 eager 路径，返回 `isGraphCapture` 标识字段）。

- **能力目标**：
  1. **SimGraphRunner**（§9.7 / §3.3.7）：替代当前 `GraphRunner`（S1 桩实现），提供完整的 CUDA Graph 仿真功能
  2. **`can_use_cuda_graph(batch, bs, is_decode_bs=True)`**：判断 batch 是否可使用 CUDA Graph replay——bs 须在 `cuda_graph_bs_list` 中且为 decode 阶段
  3. **`pad_batch_to_bs(batch, target_bs)`**：补 padding req（dummyReq）至分桶边界，padding 不计 latency
  4. **`graph_replay_cost_ticks(bs)`**：模拟 CUDA Graph replay 的 GPU ticks 开销，公式 ≈ `cuda_graph_replay_overhead_us × (1 + 0.05 × bs / 128)`
  5. **`eager_forward_cost_ticks(bs, tokens_per_seq)`**：模拟 eager forward 的 GPU ticks 开销，prefill 与 decode 分开成本模型
  6. **与 MockEngine 的集成**：`forward_batch` 中当 `can_use_cuda_graph=True` 时走 `graph_replay_cost_ticks`，否则走 `eager_forward_cost_ticks`；返回 `isGraphCapture=True/False` 字段
  7. **单元测试**：分桶边界（bs=31→32）、eager 与 graph 切换一致、`pad_batch` 不干扰 KV/KVPool 分配计数

- **影响范围**：仅修改 `server/src/sglang/engine/index.ts`（新增 `SimGraphRunner` class）、`server/src/sglang/scheduler/index.ts`（`_prepareBatch` 改用 `simGraphRunner`）、`server/src/sglang/index.ts`（更新 re-export），以及 `server/src/test/` 目录（新增 `sglang-s4.test.ts`）。不修改已有测试代码。

- **依赖 Issue**：
  - #17 S3: MockEngine + SimScheduler normal_tick（已完成）

- **阻塞 Issue**：
  - S5: Overlap Scheduling（S5 的 `_overlap_tick` 要用到 `graph_replay` vs `eager` 的 `last_data` 差异）

## 改造方案

### 总体思路

按照 §9.7 / §3.3.7 / §9.11 的规格，将当前 S1 阶段的 `GraphRunner` 桩实现升级为完整的 `SimGraphRunner`，核心变更如下：

1. **新增 `SimGraphRunner` class**：保留原 `GraphRunner` 不变（供 P3a/P4/P5 测试使用），新增 `SimGraphRunner` class 作为完整实现
2. **新增 `determineCudaGraphBs` 静态方法**：根据配置计算分桶列表 `[1, 2, 4] + range(8, max_bs+1, 8)`
3. **新增 `canUseCudaGraph(batch)`**：完整实现 decode 判断 + bs 分桶判断
4. **新增 `padBatchToBs(batch, targetBs)`**：使用 dummyReq 填充至指定分桶边界
5. **新增 `graphReplayCostTicks(bs)`**：公式化计算 graph replay 成本
6. **新增 `eagerForwardCostTicks(bs, tokensPerSeq)`**：公式化计算 eager forward 成本
7. **更新 `MockEngine.forward_batch`**：使用 `SimGraphRunner` 的时间模型替代当前简单的 `_computePrefillTime`/`_computeDecodeTime`
8. **更新 `SimScheduler._prepareBatch`**：`padBatch` 调用改用 `simGraphRunner`

核心设计决策：

1. **SimGraphRunner 与 GraphRunner 并存**：当前 `GraphRunner` 被 P3a/P4/P5 测试引用，为避免破坏已有测试，保留 `GraphRunner` 不做改动。`SimGraphRunner` 为独立 class，API 更丰富（新增 `graphReplayCostTicks`/`eagerForwardCostTicks`/`padBatchToBs`/`determineCudaGraphBs`）。
2. **SimGraphRunner 注入到 MockEngine**：`MockEngine` 新增 `simGraphRunner` 属性，`forward_batch` 使用 `simGraphRunner`；原 `graphRunner` 保留不变供 P4 使用。
3. **时间模型公式**：`graphReplayCostTicks(bs) = config.graphReplayCostTicks × (1 + 0.05 × bs / 128)`，对齐 Issue 描述中的公式；`eagerForwardCostTicks(bs, tokensPerSeq)` 区分 prefill 和 decode：prefill 按 `total_tokens = bs × tokensPerSeq` 线性增长，decode 按 bs 有轻微超线性增长。
4. **padBatchToBs 不干扰 KV 分配**：dummyReq 的 tableIdx = maxRunningReq，pageTable 最后一行填充 numTokens（标记全部已用），因此 dummyReq 不会触发真实的 KV 页分配。`_prepareBatch` 中 `cacheManager.allocatePaged(req)` 仅遍历 `batch.reqs`（不含 padded 的 dummyReq）。

### 详细设计

#### 1. SimGraphRunner 类

对齐 §9.7 / §3.3.7 / §9.11 L3590-3623。

**构造函数**：接收 `config: SimulatorConfig`, `modelConfig: ModelConfig`, `dummyReq: Req`。初始化 `enableCudaGraph`, `graphBsList`（通过 `determineCudaGraphBs` 计算）, `maxGraphBs`, `vocabSize`, `dummyReq`, `config`。

**`determineCudaGraphBs`（静态方法）**：
- 用户指定 `cudaGraphBs` → 直接返回
- 自动计算：`totalGpuMemory > 80GiB → maxBs=256`，否则 `maxBs=160`
- `maxBs < 1` → 返回空列表
- 生成 `[1, 2, 4, 8, 16, 24, 32, ..., maxBs]`（即 `[1,2,4] + range(8, maxBs+1, 8)`）

**`canUseCudaGraph(batch: Batch): boolean`**：
- `enableCudaGraph === false` → 返回 `false`
- `batch.numDecodeTokens > 0 && batch.extendInputTokens === 0`（decode 阶段）且 `batch.reqs.size <= maxGraphBs` → 返回 `true`
- 否则返回 `false`

**`padBatch(batch: Batch): void`**：
- 若 `canUseCudaGraph(batch)` → 找 `graphBsList` 中最小 `>= batch.reqs.size` 的值作为 `targetBs`
- 否则 → `targetBs = batch.reqs.size`
- 调用 `padBatchToBs(batch, targetBs)`

**`padBatchToBs(batch: Batch, targetBs: number): void`**：
- `dummyCount = max(0, targetBs - batch.reqs.size)`
- `batch.paddedReqs = [...batch.reqs.values(), ...Array(dummyCount).fill(dummyReq)]`

**`graphReplayCostTicks(bs: number): number`**：
- `return ceil(config.graphReplayCostTicks × (1 + 0.05 × bs / 128))`

**`eagerForwardCostTicks(bs: number, tokensPerSeq: number): number`**：
- prefill（`tokensPerSeq > 1`）：`return config.eagerForwardCostTicks × tokensPerSeq`（按总 token 数线性增长）
- decode（`tokensPerSeq === 1`）：`return ceil(config.eagerForwardCostTicks × (1 + 0.1 × (bs - 1) / 128))`（按 bs 轻微超线性增长）

**`replay(batch: Batch): number[][]`**：
- 返回行数 = `batch.reqs.size`（不含 padding），列数 = `vocabSize`
- 内容为全零（与现有 `GraphRunner.replay` 一致）

**`destroyCudaGraphs(): void`**：
- 仿真中为 noop

#### 2. MockEngine 更新

**新增属性**：`readonly simGraphRunner: SimGraphRunner`

**构造函数**：在现有逻辑后新增 `this.simGraphRunner = new SimGraphRunner(config, this.modelConfig, this.dummyReq)`

**`forward_batch` 更新**：
1. CUDA Graph 判断改用 `this.simGraphRunner.canUseCudaGraph(batch)`
2. Graph replay 改用 `this.simGraphRunner.replay(batch)`
3. 时间模型改用 `this.simGraphRunner.graphReplayCostTicks(bs)` 和 `this.simGraphRunner.eagerForwardCostTicks(bs, tokensPerSeq)`

#### 3. SimScheduler._prepareBatch 更新

将 `this.engine.graphRunner.padBatch(batch)` 改为 `this.engine.simGraphRunner.padBatch(batch)`。方法签名兼容，行为等价。

#### 4. index.ts 更新

新增 `SimGraphRunner` 的 re-export，保留原有 `GraphRunner` export 不变。

### 接口变更

1. **`engine/index.ts`**：新增 `SimGraphRunner` class；`MockEngine` 新增 `simGraphRunner` 属性；`MockEngine.forward_batch` 更新时间模型
2. **`scheduler/index.ts`**：`_prepareBatch` 中 `padBatch` 调用改用 `engine.simGraphRunner`
3. **`index.ts`**：新增 `SimGraphRunner` re-export

### 数据结构改动

1. **新增 `SimGraphRunner` class** — CUDA Graph 仿真运行器
2. **`MockEngine` 新增 `simGraphRunner` 属性** — 指向 `SimGraphRunner` 实例

### 修改点清单

1. **修改 `server/src/sglang/engine/index.ts`**：新增 `SimGraphRunner` class（含 `determineCudaGraphBs`/`canUseCudaGraph`/`padBatch`/`padBatchToBs`/`graphReplayCostTicks`/`eagerForwardCostTicks`/`replay`/`destroyCudaGraphs` 方法）；`MockEngine` 新增 `simGraphRunner` 属性；`MockEngine.forward_batch` 更新时间模型使用 `simGraphRunner`
2. **修改 `server/src/sglang/scheduler/index.ts`**：`_prepareBatch` 中 `padBatch` 调用改用 `engine.simGraphRunner.padBatch(batch)`
3. **修改 `server/src/sglang/index.ts`**：新增 `SimGraphRunner` re-export
4. **新建 `server/src/test/sglang-s4.test.ts`**：S4 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | SimGraphRunner 构造 | enableCudaGraph/graphBsList/maxGraphBs/vocabSize 正确初始化 |
| T2 | determineCudaGraphBs - 用户指定 | 传入 cudaGraphBs 时直接返回用户列表 |
| T3 | determineCudaGraphBs - 自动计算 | cudaGraphBs=null 时生成 [1,2,4,8,16,...,maxBs] |
| T4 | determineCudaGraphBs - 大显存 | totalGpuMemory>80GiB 时 maxBs=256 |
| T5 | determineCudaGraphBs - 小显存 | totalGpuMemory<=80GiB 时 maxBs=160 |
| T6 | determineCudaGraphBs - 禁用 | cudaGraphMaxBs<1 时返回空列表 |
| T7 | canUseCudaGraph - 禁用 | enableCudaGraph=false 时返回 false |
| T8 | canUseCudaGraph - decode batch | decode batch 且 bs<=maxGraphBs 时返回 true |
| T9 | canUseCudaGraph - prefill batch | prefill batch 时返回 false |
| T10 | canUseCudaGraph - bs 超限 | bs>maxGraphBs 时返回 false |
| T11 | padBatch - decode batch pad 到分桶 | bs=3 时 pad 到 4（cudaGraphBs=[1,2,4,8]） |
| T12 | padBatch - prefill batch 不 pad | prefill batch 时 paddedReqs 长度 = batch.reqs.size |
| T13 | padBatchToBs - 显式指定目标 | padBatchToBs(batch, 8) 将 bs=5 的 batch pad 到 8 |
| T14 | padBatchToBs - 使用 dummyReq | padding 使用 dummyReq 而非 null |
| T15 | padBatchToBs - 不干扰 KV 分配计数 | padBatch 后 cacheManager.freeSlots 不变 |
| T16 | graphReplayCostTicks - 基本值 | bs=1 时 ≈ graphReplayCostTicks × (1 + 0.05/128) |
| T17 | graphReplayCostTicks - 随 bs 增长 | bs=128 时 ≈ graphReplayCostTicks × 1.05 |
| T18 | graphReplayCostTicks - 大 bs | bs=256 时 ≈ graphReplayCostTicks × 1.1 |
| T19 | eagerForwardCostTicks - prefill | tokensPerSeq=100 时 = eagerForwardCostTicks × 100 |
| T20 | eagerForwardCostTicks - decode | tokensPerSeq=1 时按 bs 公式计算 |
| T21 | replay - 返回正确行数 | 返回行数 = batch.reqs.size（不含 padding） |
| T22 | replay - 返回正确列数 | 返回列数 = vocabSize |
| T23 | MockEngine.simGraphRunner 属性 | MockEngine 构造后 simGraphRunner 非 undefined |
| T24 | MockEngine.forward_batch - graph replay 时间 | decode batch 走 CUDA Graph 时 decodeBatchTime 使用 graphReplayCostTicks 公式 |
| T25 | MockEngine.forward_batch - eager 时间 | prefill batch 时 prefillBatchTime 使用 eagerForwardCostTicks 公式 |
| T26 | MockEngine.forward_batch - isGraphCapture 标识 | 使用 SimGraphRunner 判断 isGraphCapture |
| T27 | SimScheduler._prepareBatch 使用 simGraphRunner | padBatch 使用 simGraphRunner |
| T28 | 分桶边界 bs=31→32 | bs=31 时 pad 到 32（cudaGraphBs 含 32） |
| T29 | eager 与 graph 切换一致 | 同一 batch 在 enableCudaGraph 开关切换下 isGraphCapture 一致性 |
| T30 | destroyCudaGraphs 为 noop | 调用不抛异常 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | cudaGraphBs 为空列表 | canUseCudaGraph 返回 false，maxGraphBs=0 |
| B2 | bs=0 的空 batch | canUseCudaGraph 返回 false，padBatch 不 pad |
| B3 | bs 恰好等于分桶值 | padBatch 不添加 dummy（pad=0） |
| B4 | bs=1 的 decode batch | canUseCudaGraph 返回 true，pad 到 1 |
| B5 | chunked prefill batch | canUseCudaGraph 返回 false（extendInputTokens>0） |
| B6 | graphReplayCostTicks=0 | graphReplayCostTicks(bs) 返回 0 |
| B7 | eagerForwardCostTicks=0 | eagerForwardCostTicks 返回 0 |
| B8 | 多次 padBatch 调用 | 每次重新计算 paddedReqs，不累积 |

## 风险与注意事项

- **兼容性影响**：`MockEngine` 新增 `simGraphRunner` 属性，不删除原有 `graphRunner`，P3a/P4/P5 测试不受影响。`_prepareBatch` 改用 `simGraphRunner.padBatch`，但方法签名兼容（同名、同参数），SimScheduler 行为不变。
- **性能影响**：`SimGraphRunner` 方法均为纯计算，开销可忽略。
- **回滚方案**：所有改动在 `issue-18` 分支，合并前可安全回滚。
- **与现有 GraphRunner 的关系**：`GraphRunner`（S1）保留为独立 class，P4 的 `forwardBatchPP` 仍使用它。`SimGraphRunner` 是 S4 新增 class，供 `MockEngine.forward_batch` 和 `SimScheduler._prepareBatch` 使用。二者 API 有重叠但并非继承关系，避免 S1 改动影响 P4 测试。
- **forward_batch 时间模型的一致性**：S4 使用公式化模型后，prefillBatchTime 和 decodeBatchTime 的值可能与 S3 有微小差异（S3 使用简单的 `tokens × costTicks`，S4 使用 `costTicks × (1 + 0.05 × bs / 128)` 等），但这是预期的精度提升，不影响调度正确性。
- **padBatchToBs 不干扰 KV 分配**：dummyReq 的 tableIdx = maxRunningReq（最后一行），pageTable 最后一行填充 numTokens 标记所有页已使用，因此 `_prepareBatch` 中 `cacheManager.allocatePaged(req)` 仅遍历 `batch.reqs`（不含 padded 的 dummyReq），不会对 dummyReq 执行页分配。
