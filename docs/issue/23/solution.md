---
title: "Issue #23 解决方案"
issue_number: 23
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #23 解决方案

## 需求分析

### 问题描述

Issue #23 要求在 `server/src/sglang/parallel/**` 下实现两个核心函数：

1. **`calculate_memory_budget_parallel`**（§10.7.2）：组合多种并行维度的内存预算修正，基于已有的 `calculateMemoryBudget`（K5/Issue #12），叠加 TP/DP/EP/PP/CP 五种并行维度对权重和 KV cache 的影响
2. **`validate_parallel_config`**（§10.7.3）：7 条 assert + 1 条警告的并行配置合法性验证，返回 `{ok, errors, warnings}` 结构化结果

### 能力目标

- **组合内存预算**：当多种并行同时启用时，正确修正权重占用和 KV cache 页数计算：
  - TP 修正：权重 ÷ tp_size；KV heads 按 `divEven` 分配到各 TP rank
  - DP 修正：标准 DP 无修正（每 rank 独立 KV pool）；DP Attention 时 MLA KV cache 在 dp group 间 all-gather → `kv_per_tok_bytes ×= dp_size`
  - EP 修正（MoE）：FFN 权重按 ep_size 切分（`weight_bytes/ep_size`），但 experts_count × MoE 矩阵附加开销；KV 不修正
  - PP 修正：weight_bytes 按 stage 切分（`weight_bytes ÷ pp_size`），KV 不修正
  - CP 修正（§10.8）：`kv_per_tok_bytes × cp_size`（KV all-gather 导致每 rank 保留 cp_size 份完整 KV）
- **返回值**：与基础 `calculateMemoryBudget` 同样结构（`MemoryBudgetResult`），附加并行修正字段
- **配置验证**：7 条约束 + 1 条警告，返回结构化结果而非抛出异常；集成 MockEngine constructor 校验

### 影响范围

| 层 | 路径 | 影响程度 |
|---|---|---|
| 新文件 | `server/src/sglang/parallel/budget.ts` | 高 — 组合内存预算核心实现 |
| 新文件 | `server/src/sglang/parallel/validate.ts` | 高 — 并行配置验证核心实现 |
| 修改 | `server/src/sglang/parallel/index.ts` | 中 — re-export budget/validate 模块 |
| 修改 | `server/src/sglang/index.ts` | 低 — re-export 新增导出 |
| 修改 | `server/src/sglang/cache/budget.ts` | 中 — MemoryBudgetResult 扩展并行修正字段 |
| 依赖 | `server/src/sglang/types.ts` | 无变更 — 只读取 SimulatorConfig/ModelConfig |
| 依赖 | `server/src/sglang/core/index.ts` | 无变更 — 复用 divEven 工具函数 |
| 依赖 | `server/src/sglang/parallel/topology.ts` | 无变更 — 复用 ParallelTopology.ppStageLayers |

## 改造方案

### 总体思路

在 `parallel/` 子模块下新建两个文件：
- `budget.ts`：实现 `calculateMemoryBudgetParallel`，在基础 `calculateMemoryBudget` 之上叠加五种并行维度的修正
- `validate.ts`：实现 `validateParallelConfig`，按 §10.7.3 规格逐条检验并行配置合法性

扩展 `MemoryBudgetResult` 接口，增加并行修正字段（`tpCorrection`、`dpCorrection`、`epCorrection`、`ppCorrection`、`cpCorrection`），用于调试和指标追踪。

### 详细设计

#### 1. MemoryBudgetResult 扩展

```typescript
/** calculateMemoryBudget / calculateMemoryBudgetParallel 返回值结构 */
export interface MemoryBudgetResult {
  /** 可分配的 KV cache 页数（≥0，0 表示 OOM） */
  numPages: number;
  /** 模型权重占用的显存（bytes） */
  modelMemory: number;
  /** CUDA Graph buffer 占用的显存（bytes） */
  graphBuffer: number;
  /** 并行修正字段（仅 calculateMemoryBudgetParallel 填充） */
  parallelCorrections?: ParallelMemoryCorrections;
}

/** 并行维度内存修正明细 */
export interface ParallelMemoryCorrections {
  /** TP 修正：权重除以 tp_size */
  tpWeightDivisor: number;
  /** DP 修正：标准 DP 时 kv_budget_per_rank = kv_budget / dp_size；DP Attention 时 kv_per_tok_bytes ×= dp_size */
  dpKvMultiplier: number;
  /** EP 修正：MoE FFN 权重除以 ep_size（附加专家矩阵开销已在权重估算中） */
  epWeightDivisor: number;
  /** PP 修正：权重按 stage 切分，除以 pp_size */
  ppWeightDivisor: number;
  /** CP 修正：kv_per_tok_bytes ×= cp_size */
  cpKvMultiplier: number;
}
```

#### 2. calculateMemoryBudgetParallel

**文件**：`server/src/sglang/parallel/budget.ts`

**签名**：

```typescript
export function calculateMemoryBudgetParallel(
  config: SimulatorConfig,
  modelConfig: ModelConfig,
  totalMemory: number,
): MemoryBudgetResult
```

**核心逻辑**（对应 §10.7.2 Python 伪代码）：

```
1. 基础权重估算：
   - 不启用 DP Attention 时：weight_bytes = estimateModelMemory(modelConfig, dtypeSize) / tpSize
   - 启用 DP Attention 时：拆分 attention/MLP 权重
     - attn_weight = _estimateAttnWeightBytes(modelConfig, dtypeSize)  // 不除 tp（attention 权重复制）
     - mlp_weight = _estimateMlpWeightBytes(modelConfig, dtypeSize) / tpSize  // MLP 权重按 TP 切分

2. EP 修正：若 epSize > 1 且 isMoe，weight_bytes = weight_bytes / epSize
   注：MoE 专家矩阵的附加开销在 _estimateMlpWeightBytes 中已通过 numExperts × moeIntermediateSize 体现

3. PP 修正：weight_bytes = weight_bytes / ppSize

4. graphBuffer = estimateGraphBuffer(config.cudaGraphBs, modelConfig)

5. available = Math.floor(memoryRatio × totalMemory) - weight_bytes - graphBuffer

6. DP 修正：
   - 标准 DP：kv_budget_per_rank = available / dpSize（每 rank 独立 KV pool）
   - DP Attention：kv_budget_per_rank = available（不除 dpSize，但后面 kv_per_tok_bytes 会乘 dpSize）

7. KV per token per layer 计算：
   - local_kv_heads = sum(divEven(numKvHeads, tpSize, true)) （复用 K5 逻辑）
   - kv_per_tok_per_layer = 2 × local_kv_heads × headDim × dtypeSize

8. CP 修正：kv_per_tok_per_layer *= cpSize（KV all-gather 导致每 rank 保留 cp_size 份）

9. DP Attention 修正：kv_per_tok_per_layer *= dpSize（MLA KV cache 在 dp group 间 all-gather）

10. bytes_per_token = kv_per_tok_per_layer × numLayers

11. num_tokens = Math.floor(kv_budget_per_rank / Math.max(1, bytes_per_token))

12. num_pages = Math.max(0, Math.floor(num_tokens / pageSize))

13. 返回 MemoryBudgetResult 含 parallelCorrections
```

**辅助函数**：

```typescript
/**
 * 估算 attention 权重占用的字节数
 * 公式：numLayers × (numAttentionHeads × headDim × 3 + hiddenSize) × dtypeSize
 * 其中 3 = Q/K/V 三个权重矩阵，+ hiddenSize 为 output projection
 */
function _estimateAttnWeightBytes(modelConfig: ModelConfig, dtypeSize: number): number

/**
 * 估算 MLP 权重占用的字节数
 * - 非 MoE：numLayers × hiddenSize × intermediateSize × 3 × dtypeSize
 *   （gate/up/down 三个矩阵）
 * - MoE：numLayers × numExperts × hiddenSize × moeIntermediateSize × 3 × dtypeSize
 *   仅计算 MoE 层的专家矩阵权重
 */
function _estimateMlpWeightBytes(modelConfig: ModelConfig, dtypeSize: number): number
```

> **设计说明**：Issue 正文中的 Python 伪代码（§10.7.2）将权重拆分为 attn_weight + mlp_weight，本方案严格遵循此拆分逻辑。`estimateModelMemory` 的粗略公式（`hidden² × 12`）将在 `calculateMemoryBudgetParallel` 中被精确拆分替代，确保 DP Attention 场景下的准确性。

#### 3. validateParallelConfig

**文件**：`server/src/sglang/parallel/validate.ts`

**签名**：

```typescript
export interface ValidationResult {
  ok: boolean;
  errors: string[];
  warnings: string[];
}

export function validateParallelConfig(
  config: SimulatorConfig,
  modelConfig: ModelConfig,
): ValidationResult
```

**7 条约束 + 1 条警告**（对应 §10.7.3 + Issue 正文）：

| # | 约束 | 检查逻辑 | 失败类型 |
|---|------|---------|---------|
| 1 | `world_size === tp_size × dp_size × pp_size` | `config.tpSize * config.dpSize * config.ppSize === expectedWorldSize`（或通过 ParallelTopology.worldSize 计算） | error |
| 2 | `ep_size >= 1 && (ep_size === 1 || model.isMoe)` | `config.epSize === 1 || modelConfig.isMoe` | error |
| 3 | `tp_size % cp_size === 0` | `config.tpSize % config.cpSize === 0`（CP 整除 TP，attention 分 rank） | error |
| 4 | `(tp_size / cp_size) % ep_size === 0` | `(config.tpSize / config.cpSize) % config.epSize === 0`（剩余 TP/cp 整除 EP，专家分布） | error |
| 5 | `pp_size >= 1 && pp_stage_layers 所有阶段层数 >= 1` | 利用 `ParallelTopology.ppStageLayers(modelConfig.numLayers)` 验证每 stage 层数 ≥ 1 | error |
| 6 | `dp_size >= 1 && (enable_dp_attention → model.useMla)` | `!config.enableDpAttention || modelConfig.useMla` | error |
| 7 | `mem_fraction > 0 && mem_fraction <= 1` | `config.memoryRatio > 0 && config.memoryRatio <= 1` | error |
| W | `(kv_heads * cp_size) % tp_size !== 0` | `(modelConfig.numKvHeads * config.cpSize) % config.tpSize !== 0` | warning |

**返回值语义**：
- `ok = errors.length === 0`
- `errors`：所有约束违反的错误消息
- `warnings`：非整除等警告信息

> **与 §10.7.3 Python 版的差异**：Python 版使用 `assert` + `raise ValueError`，本方案改为收集所有错误后一次性返回结构化结果，便于调用方（如 MockEngine）统一处理。同时增加了 Issue 正文要求的约束 1（world_size 一致性）和约束 6（DP Attention → useMla），这些在 Python 伪代码中未显式列出但 Issue 正文明确要求。

#### 4. MockEngine 集成点

`validateParallelConfig` 将在 `MockEngine` 构造函数中调用（属于后续 Issue P6 的集成范围，本 Issue 仅提供函数）。调用模式：

```typescript
// MockEngine constructor 中（P6 实现）
const result = validateParallelConfig(config, modelConfig);
if (!result.ok) {
  throw new Error(`Invalid parallel config: ${result.errors.join("; ")}`);
}
for (const w of result.warnings) {
  console.warn(`[validateParallelConfig] ${w}`);
}
```

### 修改点清单

1. **新建** `server/src/sglang/parallel/budget.ts`
   - 导出 `ParallelMemoryCorrections` 接口
   - 导出 `calculateMemoryBudgetParallel` 函数
   - 内部 `_estimateAttnWeightBytes`、`_estimateMlpWeightBytes` 辅助函数

2. **新建** `server/src/sglang/parallel/validate.ts`
   - 导出 `ValidationResult` 接口
   - 导出 `validateParallelConfig` 函数

3. **修改** `server/src/sglang/cache/budget.ts`
   - `MemoryBudgetResult` 接口增加 `parallelCorrections?: ParallelMemoryCorrections` 可选字段
   - 导出 `ParallelMemoryCorrections` 接口（从 `parallel/budget.ts` re-export 或在此定义）

4. **修改** `server/src/sglang/parallel/index.ts`
   - 新增 re-export：`calculateMemoryBudgetParallel`、`ParallelMemoryCorrections`
   - 新增 re-export：`validateParallelConfig`、`ValidationResult`

5. **修改** `server/src/sglang/index.ts`
   - 新增 re-export：`calculateMemoryBudgetParallel`、`validateParallelConfig`、`ValidationResult`、`ParallelMemoryCorrections`

6. **修改** `server/src/sglang/cache/index.ts`
   - 确保 `ParallelMemoryCorrections` 随 `MemoryBudgetResult` 一起导出

## 测试设计

### 验收测试用例清单

#### calculateMemoryBudgetParallel 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T1 | 全 size=1 退化为基础 calculateMemoryBudget | tp=dp=ep=pp=cp=1 时，结果与 `calculateMemoryBudget` 一致 |
| T2 | TP>1 权重修正 | tp=2 时，modelMemory 约为基础的一半 |
| T3 | TP>1 KV heads 分割 | tp=2, numKvHeads=8 → local_kv_heads=4 |
| T4 | DP>1 KV budget 分割 | dp=2 时，每 rank KV budget 减半 |
| T5 | DP Attention KV 乘数 | enableDpAttention=true, dp=2 时，kv_per_tok ×= dpSize（MLA KV cache all-gather） |
| T6 | EP>1 MoE 权重修正 | ep=2, isMoe=true 时，weight_bytes / epSize |
| T7 | EP>1 非 MoE 不修正 | ep=2, isMoe=false 时，权重不除 epSize（validate 会报错，但 budget 函数仍应正常计算） |
| T8 | PP>1 权重修正 | pp=2 时，weight_bytes / ppSize |
| T9 | CP>1 KV 乘数 | cp=2 时，kv_per_tok_bytes ×= cpSize |
| T10 | 组合并行 tp=2,dp=2,ep=2,pp=2,cp=2 | world_size=8（tp×dp×pp），所有修正叠加正确 |
| T11 | parallelCorrections 字段填充 | 非 size=1 时，parallelCorrections 字段非 undefined |
| T12 | OOM 场景 | totalMemory 极小时 numPages=0 |
| T13 | DP Attention attention 权重复制 | enableDpAttention=true 时，attn_weight 不除 tpSize |

#### validateParallelConfig 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T14 | 合法配置通过 | tp=2,dp=2,ep=1,pp=1,cp=1,world_size=4 → ok=true, errors=[] |
| T15 | 约束 1：world_size 不匹配 | tp=2,dp=2,pp=2 但期望 world_size≠8 → error |
| T16 | 约束 2：EP>1 但非 MoE | ep=2, isMoe=false → error |
| T17 | 约束 3：cp_size 不整除 tp_size | tp=8, cp=3 → error |
| T18 | 约束 4：(tp/cp) 不整除 ep_size | tp=8, cp=2, ep=3 → error |
| T19 | 约束 5：pp_size > numLayers | pp=100, numLayers=32 → error（某 stage 层数=0） |
| T20 | 约束 6：DP Attention 但非 MLA | enableDpAttention=true, useMla=false → error |
| T21 | 约束 7：mem_fraction 越界 | memoryRatio=1.5 或 memoryRatio=0 → error |
| T22 | 警告：KV heads 不整除 | numKvHeads=7, tp=4, cp=1 → warning |
| T23 | 多错误同时返回 | 同时违反多条约束 → errors 包含多条消息 |
| T24 | 全默认配置通过 | tp=dp=ep=pp=cp=1 → ok=true |

### 边界条件覆盖

- `totalGpuMemory` 极小（1 byte）→ `numPages = 0`，OOM
- `memoryRatio = 0` → available 为负 → numPages = 0
- `numKvHeads = 1`（MLA 场景）→ divEven 正确处理
- `numLayers = 0` → cachePerPage = 0 → numPages = 0（除零保护）
- `epSize > 1` 但 `isMoe = false` → validate 报错，budget 不除 epSize
- `cpSize = tpSize` → attn_tp_size = 1（全部 rank 用于 CP 切分）
- `ppSize = numLayers` → 每 stage 恰好 1 层
- `ppSize > numLayers` → 某 stage 0 层 → validate 报错

## 风险与注意事项

### 兼容性影响

- 新增 `parallel/budget.ts` 和 `parallel/validate.ts` 为纯新增文件
- `MemoryBudgetResult` 增加 `parallelCorrections?` 可选字段，**零破坏性**：已有代码不访问此字段不受影响
- `parallel/index.ts` 和 `sglang/index.ts` 仅新增 re-export，不影响现有导入

### 性能影响

- `calculateMemoryBudgetParallel` 为 O(1) 纯计算（除 divEven 内部 O(tpSize)），与基础版本一致
- `validateParallelConfig` 为 O(ppSize)（ppStageLayers 遍历），仅在 MockEngine 初始化时调用一次，不在热路径上
- 两个函数均为纯函数，无副作用（仅 validate 可能通过 console.warn 输出警告）

### 回滚方案

- 删除 `parallel/budget.ts` 和 `parallel/validate.ts`
- 恢复 `MemoryBudgetResult` 接口（移除 `parallelCorrections` 可选字段）
- 恢复 `parallel/index.ts` 和 `sglang/index.ts` 到修改前状态即可回滚
