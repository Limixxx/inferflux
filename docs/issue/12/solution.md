---
title: "Issue #12 解决方案"
issue_number: 12
issue_type: Feature
created: 2026-08-27
updated: 2026-08-27
status: draft
review_round: 1
---

# Issue #12 解决方案

## 需求分析

### 问题描述

当前 `server/src/sglang/` 模块已具备骨架（S0 #9）和核心数据结构（S1 #10），但缺少内存预算计算功能。SGLang 仿真器需要忠实模拟 SGLang 的显存分配策略（mem_fraction_static），即根据模型配置、GPU 总显存、mem_fraction、page_size 等参数，计算出可分配的 KV cache 页数（num_pages），这是后续 MockKVCachePool、CacheManager、RadixCache 等组件的基础输入。

本 Issue（K5）需在 `server/src/sglang/cache/` 下实现：

1. `calculateMemoryBudget(config, modelConfig, totalMemory)`：核心公式，返回 `{ numPages, modelMemory, graphBuffer }`（§3.3.9）
2. `estimateModelMemory(modelConfig, dtypeSize)`：估算模型权重占用的显存（bytes）
3. `estimateGraphBuffer(cudaGraphBs, modelConfig)`：估算 CUDA Graph buffer 占用的显存（bytes）
4. OOM / 不足页警告逻辑
5. `_divEven` 按 TP head 复制的本地 KV head 数计算（已有 `divEven` 工具函数，K5 直接复用）

### 能力目标

1. 实现权重内存估算：`hidden × hidden × 12 × layers × dtypeSize`（对应 §3.3.9 `estimate_model_memory` 公式）
2. 实现 CUDA Graph buffer 估算：`max_bs × hidden × layers × 4`（对应 §3.3.9 `estimate_graph_buffer` 公式）
3. 实现 mem_fraction_static 核心公式：
   - `available = memoryRatio × totalGpuMemory - modelMemory - graphBuffer`
   - `kvHeadsPerGpu = sum(divEven(numKvHeads, tpSize, allowReplicate=true))`
   - `cachePerPage = 2 × headDim × kvHeadsPerGpu × pageSize × dtypeSize × numLayers`
   - `numPages = max(0, available ÷ cachePerPage)`
4. 返回值结构体 `{ numPages, modelMemory, graphBuffer }`，供 MockEngine._calculateNumPages() 调用
5. 当 `numPages < 1` 时输出 OOM 警告（console.warn）
6. 零运行时依赖，纯 TypeScript strict 实现

### 影响范围

| 层 | 路径 | 影响程度 |
|---|---|---|
| 新文件 | `server/src/sglang/cache/budget.ts` | 高 — 核心公式实现 |
| 修改 | `server/src/sglang/cache/index.ts` | 中 — re-export budget 模块 |
| 修改 | `server/src/sglang/index.ts` | 低 — re-export calculateMemoryBudget 等 |
| 依赖 | `server/src/sglang/types.ts` | 无变更 — 只读取 SimulatorConfig/ModelConfig |
| 依赖 | `server/src/sglang/core/index.ts` | 无变更 — 复用 divEven 工具函数 |
| 现有代码 | `server/src/sglang/engine/` | 无变更 — 本 Issue 仅提供函数，Engine 在后续 Issue 调用 |

## 改造方案

### 总体思路

在 `cache/` 子模块下新建 `budget.ts`，实现 §3.3.9 定义的三个函数（`estimateModelMemory`、`estimateGraphBuffer`、`calculateMemoryBudget`）。这些函数为纯函数（pure function），输入配置参数，输出计算结果，无副作用（仅 OOM 时 console.warn）。函数签名与返回值类型与 §3.3.9 Python 伪代码精确对齐，转为 TypeScript 风格。

### 详细设计

#### 1. 返回值类型

```typescript
/** calculateMemoryBudget 返回值结构 */
export interface MemoryBudgetResult {
  /** 可分配的 KV cache 页数（≥0，0 表示 OOM） */
  numPages: number;
  /** 模型权重占用的显存（bytes） */
  modelMemory: number;
  /** CUDA Graph buffer 占用的显存（bytes） */
  graphBuffer: number;
}
```

#### 2. estimateModelMemory

```typescript
/**
 * 估算模型权重占用的显存（bytes）
 * 对应 §3.3.9 estimate_model_memory
 *
 * 粗略估算：每层参数量 ≈ hidden² × 12（QKV + FFN + embed）
 * 公式：numLayers × hiddenSize × hiddenSize × 12 × dtypeSize
 */
export function estimateModelMemory(
  modelConfig: ModelConfig,
  dtypeSize: number,
): number {
  return modelConfig.numLayers * modelConfig.hiddenSize * modelConfig.hiddenSize * 12 * dtypeSize;
}
```

> **设计说明**：公式中 `12` 是一个粗略系数，涵盖每层 attention 的 Q/K/V/O 四个权重矩阵加上 FFN 的 gate/up/down 三个权重矩阵以及 layer norm 等参数。这与 SGLang 源码中 `mem_fraction_static` 的估算逻辑一致（参见 `Awesome-ML-SYS-Tutorial/sglang/kvcache-code-walk-through/mem-fraction-static.md`）。

#### 3. estimateGraphBuffer

```typescript
/**
 * 估算 CUDA Graph buffer 占用的显存（bytes）
 * 对应 §3.3.9 estimate_graph_buffer
 *
 * 每层 buffer ≈ max_bs × hidden × 4（中间激活、logits 等）
 * 当 cudaGraphBs 为空或 null 时返回 0
 */
export function estimateGraphBuffer(
  cudaGraphBs: number[] | null,
  modelConfig: ModelConfig,
): number {
  if (!cudaGraphBs || cudaGraphBs.length === 0) {
    return 0;
  }
  const maxBs = Math.max(...cudaGraphBs);
  return maxBs * modelConfig.hiddenSize * modelConfig.numLayers * 4;
}
```

#### 4. calculateMemoryBudget

```typescript
/**
 * 计算可分配的 KV cache 页数
 * 对应 §3.3.9 calculate_memory_budget / mem_fraction_static
 *
 * @param config - SimulatorConfig（含 memoryRatio, pageSize, dtypeSize, tpSize, cudaGraphBs）
 * @param modelConfig - ModelConfig（含 numLayers, hiddenSize, numKvHeads, headDim）
 * @param totalMemory - GPU 总显存（bytes）
 * @returns MemoryBudgetResult
 */
export function calculateMemoryBudget(
  config: SimulatorConfig,
  modelConfig: ModelConfig,
  totalMemory: number,
): MemoryBudgetResult {
  // 模型权重占用
  const modelMemory = estimateModelMemory(modelConfig, config.dtypeSize);
  // CUDA Graph buffer 占用
  const graphBuffer = estimateGraphBuffer(config.cudaGraphBs, modelConfig);

  // 剩余可用 = 比例预算 - 模型权重 - graph buffer
  const available = Math.floor(config.memoryRatio * totalMemory) - modelMemory - graphBuffer;

  // KV cache 每页大小
  // divEven 返回每 GPU 的 KV head 分布列表，sum 后得到总 KV head 数
  const kvHeadsPerGpu = divEven(modelConfig.numKvHeads, config.tpSize, true)
    .reduce((sum, v) => sum + v, 0);
  const cachePerPage =
    2 *                    // key + value
    modelConfig.headDim *
    kvHeadsPerGpu *
    config.pageSize *
    config.dtypeSize *     // float16=2, bfloat16=2, float8=1
    modelConfig.numLayers;

  // 可分配的页数（OOM 保护：负数时返回 0，由调用方触发 OOM 处理）
  const numPages = Math.max(0, Math.floor(available / cachePerPage));

  // OOM 预测警告
  if (numPages < 1) {
    console.warn(
      `[calculateMemoryBudget] OOM: numPages=0, available=${available}, ` +
      `modelMemory=${modelMemory}, graphBuffer=${graphBuffer}, cachePerPage=${cachePerPage}`
    );
  }

  return { numPages, modelMemory, graphBuffer };
}
```

> **关键设计决策**：
> 1. `kvHeadsPerGpu` 使用 `sum(divEven(...))` 而非简单除法，是因为 TP 下 KV heads 可能不整除，`divEven` 的 `allowReplicate=true` 会复制余数，使得总和可能大于原始 `numKvHeads`（如 `divEven(8, 3, true) → [3, 3, 2]`，sum=8；`divEven(1, 2, true) → [1, 0]`，sum=1）。这与 SGLang 源码中 `_div_even` 的语义一致。
> 2. `available` 使用 `Math.floor` 确保整数运算，与 Python `int(...)` 行为一致。
> 3. OOM 警告使用 `console.warn` 而非 `throw`，与 §3.3.9 的"由调用方触发 OOM 处理"语义一致——函数仅返回 0 页，不中断执行流。

#### 5. 导出与 re-export

**cache/budget.ts** 导出：
- `MemoryBudgetResult` 接口
- `estimateModelMemory` 函数
- `estimateGraphBuffer` 函数
- `calculateMemoryBudget` 函数

**cache/index.ts** 从 `budget.ts` re-export 以上所有导出。

**sglang/index.ts** 新增从 `./cache` re-export `MemoryBudgetResult`、`estimateModelMemory`、`estimateGraphBuffer`、`calculateMemoryBudget`。

### 修改点清单

1. **新建** `server/src/sglang/cache/budget.ts` — 实现 `MemoryBudgetResult` 接口、`estimateModelMemory`、`estimateGraphBuffer`、`calculateMemoryBudget`
2. **修改** `server/src/sglang/cache/index.ts` — 从 `./budget` re-export 所有公共 API
3. **修改** `server/src/sglang/index.ts` — 从 `./cache` re-export `MemoryBudgetResult`、`estimateModelMemory`、`estimateGraphBuffer`、`calculateMemoryBudget`
4. **验证** `npx tsc --noEmit` 通过 strict 编译零错误

## 测试设计

### 验收测试用例清单

| 编号 | 测试场景 | 预期结果 |
|------|---------|---------|
| T1 | `estimateModelMemory` 使用默认 ModelConfig + dtypeSize=2 | `32 × 4096 × 4096 × 12 × 2 = 12,884,901,888` (≈12 GiB) |
| T2 | `estimateGraphBuffer` cudaGraphBs=null 时返回 0 | `0` |
| T3 | `estimateGraphBuffer` cudaGraphBs=[1,2,4,8] + 默认 ModelConfig | `8 × 4096 × 32 × 4 = 4,194,304` |
| T4 | `estimateGraphBuffer` cudaGraphBs=[] 空数组返回 0 | `0` |
| T5 | `calculateMemoryBudget` 默认配置（80GiB, 0.88, tp=1） | numPages > 0, modelMemory > 0 |
| T6 | `calculateMemoryBudget` 返回值类型正确 | `{ numPages: number, modelMemory: number, graphBuffer: number }` |
| T7 | `calculateMemoryBudget` tp=2 时 numPages 应小于 tp=1（权重不变，但 KV heads 减半后每页更小→实际页数增大，验证趋势正确） | numPages(tp=2) > numPages(tp=1)（因为 cachePerPage 更小） |
| T8 | `calculateMemoryBudget` 模型过大（设 totalGpuMemory=1）时 numPages=0 且输出 OOM 警告 | numPages=0, console.warn 被调用 |
| T9 | `calculateMemoryBudget` numPages 显式设定时不影响计算（numPages 参数在 SimulatorConfig 中，calculateMemoryBudget 本身不读取 numPages） | 函数仅依据公式计算，与 config.numPages 无关 |
| T10 | `calculateMemoryBudget` pageSize=16 时 numPages 应小于 pageSize=1（每页更大→页数更少） | numPages(ps=16) < numPages(ps=1) |
| T11 | `divEven(8, 3, true)` 在 calculateMemoryBudget 中正确使用 | kvHeadsPerGpu = 8, cachePerPage 计算正确 |

### 边界条件覆盖

- `totalGpuMemory` 极小（1 byte）→ `numPages = 0`，OOM 警告
- `memoryRatio = 0` → `available = 0 - modelMemory - graphBuffer < 0` → `numPages = 0`
- `memoryRatio = 1.0` → 所有显存可用于 KV cache（减去权重和 graph buffer）
- `dtypeSize = 1`（float8）→ 每页更小，页数更多
- `cudaGraphBs = null` vs `[]` vs `[1,2,4]` — graphBuffer 分别为 0, 0, 正数
- `tpSize > numKvHeads`（如 tp=4, numKvHeads=2）→ `divEven(2, 4, true)=[1,1,0,0]`，sum=2
- `numKvHeads = 1`（MLA 场景）→ `divEven(1, tpSize, true)` 正确分配
- `numLayers = 0` → `modelMemory = 0`，`cachePerPage = 0` → 除零保护（`Math.floor(available / 0) = NaN` → `max(0, NaN) = NaN`）→ 需额外保护 `cachePerPage = 0` 时返回 `numPages = 0`

## 风险与注意事项

### 兼容性影响

- 新增 `cache/budget.ts` 为纯新增文件，修改 `cache/index.ts` 和 `sglang/index.ts` 仅新增 re-export，**零兼容性破坏**
- 现有 S0/S1 测试不受影响（不导入 budget 模块）

### 性能影响

- 三个函数均为 O(1) 纯计算，无循环（除 `divEven` 内部的 O(tpSize) 分配），**零性能隐患**
- `calculateMemoryBudget` 在 MockEngine 初始化时调用一次，不在热路径上

### 回滚方案

- 删除 `cache/budget.ts`，恢复 `cache/index.ts` 和 `sglang/index.ts` 到修改前状态即可回滚
