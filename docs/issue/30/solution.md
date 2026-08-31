---
title: "Issue #30 解决方案"
issue_number: 30
issue_type: Feature
created: 2026-08-31
updated: 2026-08-31
status: draft
review_round: 1
---

# Issue #30 解决方案

## 需求分析

### 问题描述

Issue #30 要求实现 `init_parallel_groups` 函数，将之前 8 个 Issue（#22~#29）独立实现的 9 个并行仿真组件整合到一个统一的初始化入口，并将其接入 `MockEngine`/`SimScheduler` 的调度循环，同时补充 `SimulationMetrics` 的并行指标汇总与 HTTP 暴露，最终通过 6 组端到端验收测试验证整个并行仿真栈的正确性。

### 能力目标

1. **统一初始化入口**：`initParallelGroups(topology, modelCfg, simCfg)` 按条件创建 9 组件并返回 `ParallelGroups` 对象
2. **MockEngine/SimScheduler 接入**：构造器接收 `ParallelGroups`，forward_batch 层循环中调用 TP/DP-Attn/MoE/CP 各自 simulate，tick 末尾调用 EPLB maybe_rebalance，add_request 前走 DP select_rank
3. **SimulationMetrics 合并**：tick 时收集 ParallelMetrics 全部字段汇总到 toJSON，HTTP `/internal/metrics` 展示
4. **端到端验收**：6 组 case 验证退化一致性、多并行吞吐、DP-Attn 性能、PP bubble 比例、配置校验、TypeScript 严格检查

### 影响范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `server/src/sglang/parallel/index.ts` | 修改 | 新增 `ParallelGroups` 接口与 `initParallelGroups` 函数导出 |
| `server/src/sglang/engine/index.ts` | 修改 | MockEngine 构造器接入 `ParallelGroups`，重构 forwardBatch 层循环 |
| `server/src/sglang/metrics/index.ts` | 修改 | SimulationMetrics 增加 `toJSON()` 方法暴露并行指标 |
| `server/src/sglang/http/HttpService.ts` 或新增路由 | 修改 | `/internal/metrics` 端点返回 SimulationMetrics |
| `server/src/test/sglang-p6.test.ts` | 新增 | 6 组端到端验收测试 |
| `server/src/sglang/index.ts` | 修改 | 新增导出 `ParallelGroups`、`initParallelGroups` |

---

## 改造方案

### 总体思路

当前各并行组件（TPSimulator、TPCommInfraSimulator、DataParallelController、DPAttentionSimulator、PPPipelineSimulator、CPSimulator、EPLBSimulator、SimMoeBackend）在 MockEngine 构造器中各自独立创建，组件间缺乏统一的初始化编排和依赖注入。本方案的核心是：

1. 定义 `ParallelGroups` 接口，封装全部 9 组件引用
2. 实现 `initParallelGroups()` 工厂函数，按 `validateParallelConfig` → `calculateMemoryBudgetParallel` → 条件创建组件的标准流程编排
3. 将 MockEngine 中分散的组件创建逻辑替换为接收 `ParallelGroups` 参数
4. 重构 `MockEngine.forwardBatch` 为完整的层循环，逐层注入各并行维度的通信仿真
5. 在 SimulationMetrics 中合并 ParallelMetrics 字段到 `toJSON()`
6. 新增 `/internal/metrics` HTTP 端点
7. 编写 6 组端到端验收测试

### 详细设计

#### 1. ParallelGroups 接口与 initParallelGroups 函数

**文件**: `server/src/sglang/parallel/index.ts`（或在 parallel 下新建 `groups.ts` 再从 index 导出）

```typescript
/** init_parallel_groups 返回的并行组件集合（§10.7.1） */
export interface ParallelGroups {
  readonly topology: ParallelTopology;
  readonly tpComm: TPCommInfraSimulator;
  readonly tpSim: TPSimulator;
  readonly dpController: DataParallelController;
  readonly dpAttnSim: DPAttentionSimulator | null;
  readonly ppSim: PPPipelineSimulator;
  readonly cpSim: CPSimulator | null;
  readonly eplbSim: EPLBSimulator | null;
  readonly moeBackend: SimMoeBackend | null;
}
```

`initParallelGroups` 函数签名与逻辑：

```typescript
export function initParallelGroups(
  config: SimulatorConfig,
  modelConfig: ModelConfig,
  numPages: number,  // 由 scheduler/budget 计算后填入
): ParallelGroups
```

执行步骤：
1. `validateParallelConfig(config, modelConfig)` — 失败直接 throw
2. `calculateMemoryBudgetParallel(config, modelConfig, totalGpuMemory)` — 计算 numPages（若调用方未提供则使用此值）
3. `new ParallelTopology(...)` — 创建拓扑
4. 按条件创建 9 组件：
   - `topology` — 已创建
   - `tpComm = new TPCommInfraSimulator(config, modelConfig)`
   - `tpSim = new TPSimulator(config, modelConfig)`
   - `dpController = new DataParallelController(dpSize, numPages, strategy)` — 使用 `divEven(numPages, dpSize)` 分配
   - `dpAttnSim = enableDpAttention ? new DPAttentionSimulator({...}) : null`
   - `ppSim = new PPPipelineSimulator(config, modelConfig)`
   - `cpSim = cpSize > 1 ? new CPSimulator(config, modelConfig) : null`
   - `eplbSim = enableEplb ? new EPLBSimulator({...}) : null`
   - `moeBackend = isMoe ? new SimMoeBackend({...}) : null`

**注意**: DataParallelController 构造器需要 `total_num_pages` 参数，目前 MockEngine 中未使用 DP 分配页面逻辑。此处 `numPages` 参数由外部计算后传入，若调用方未提供则使用 `calculateMemoryBudgetParallel` 的返回值。

#### 2. MockEngine 接入 ParallelGroups

**文件**: `server/src/sglang/engine/index.ts`

当前 MockEngine 构造器中各自创建 `ParallelTopology`、`PPPipelineSimulator`、`CPSimulator`、`SimMoeBackend`。重构为：

- 新增可选构造参数 `parallelGroups?: ParallelGroups`
- 若提供则直接使用；否则内部调用 `initParallelGroups` 创建（向后兼容）
- 合并现有三个 forward 方法（`forwardBatchPP`、`forwardBatchSeq`、`forwardBatch`）为一个统一的 `forwardBatch` 方法，包含完整的层循环：

```typescript
forwardBatch(tokenIds: number[], seqLen: number, batch?: Batch): ForwardOutput {
  let totalCommTicks = 0;

  for (let layerIdx = 0; layerIdx < this.modelConfig.numLayers; layerIdx++) {
    // 1. Attention 计算
    // 2. TP all-reduce after attention
    totalCommTicks += this.groups.tpSim.allReduceAfterAttn(batchSize);
    // 3. CP KV all-gather（若启用）
    if (this.groups.cpSim) {
      const cpResult = this.groups.cpSim.simulateAttnForward(seqLen);
      this.simMetrics.parallel.cpCommTicks += cpResult.commTicks;
      this.simMetrics.parallel.cpAllGatherCount += 1;
    }
    // 4. MLP / MoE
    if (isMoELayer && this.groups.moeBackend) {
      const moeResult = this.groups.moeBackend.forward(tokenIds, layerIdx);
      totalCommTicks += moeResult.commTicks;
    }
    // 5. TP all-reduce after MLP
    totalCommTicks += this.groups.tpSim.allReduceAfterMlp(batchSize);
    // 6. DP-Attn all-gather after MLP（若启用）
    if (this.groups.dpAttnSim) {
      const dpResult = this.groups.dpAttnSim.simulateMlpForward(localBatchSizes);
      totalCommTicks += dpResult.commTicks;
    }
  }

  // 7. PP 通信仿真
  if (this.groups.ppSim.ppSize > 1 && batch) {
    const ppResult = this.groups.ppSim.simulatePipelineStep(batch);
    this.simMetrics.parallel.ppSendRecvTicks += ppResult.sendRecvTicks;
    this.simMetrics.parallel.ppBubbleTicks += ppResult.bubbleTicks;
  }

  // 8. EPLB maybe_rebalance（tick 末尾）
  if (this.groups.eplbSim && this.groups.moeBackend) {
    this.groups.eplbSim.maybe_rebalance(
      globalStep, this.groups.moeBackend.expertLoadCounts, this.groups.moeBackend
    );
  }

  // 9. 更新 TP 通信指标
  this.simMetrics.parallel.tpCommTicks += this.groups.tpSim.totalCommTicksPerStep();
  this.groups.tpSim.resetStepComm();

  // 10. 采样（仅最后 PP stage）
  ...
}
```

#### 3. SimScheduler 接入 ParallelGroups

**文件**: `server/src/sglang/scheduler/index.ts`

在 `add_request`（请求入队）路径中，若 `dpSize > 1`，调用 `dpController.select_rank_for_request(neededPages)` 分配 DP rank，设置 `req.dpRank = rank`。

#### 4. SimulationMetrics 合并与 HTTP 暴露

**SimulationMetrics.toJSON()**:

```typescript
toJSON(): Record<string, unknown> {
  return {
    parallel: this.parallel.summary(),
    // 后续 Issue 补充其他指标
  };
}
```

**HTTP `/internal/metrics` 端点**:

在 HttpService 的 proxy 路径中增加对 `/api/internal/metrics` 的处理，或在 SimService 中注册路由，返回 `simulationMetrics.toJSON()` 的 JSON 响应。

#### 5. 数据结构改动

| 数据结构 | 改动 | 说明 |
|----------|------|------|
| `ParallelGroups` | 新增接口 | 封装 9 组件引用 |
| `initParallelGroups()` | 新增函数 | 统一初始化入口 |
| `MockEngine` | 构造器参数变更 | 接收 `parallelGroups?`，替代各自创建 |
| `SimulationMetrics` | 新增 `toJSON()` | 暴露并行指标汇总 |

### 修改点清单

1. **新增 `server/src/sglang/parallel/groups.ts`**：定义 `ParallelGroups` 接口和 `initParallelGroups` 工厂函数
2. **修改 `server/src/sglang/parallel/index.ts`**：新增导出 `ParallelGroups`、`initParallelGroups`
3. **修改 `server/src/sglang/engine/index.ts`**：
   - MockEngine 构造器接收 `parallelGroups?` 参数
   - 合并 `forwardBatchPP`/`forwardBatchSeq`/`forwardBatch` 为统一 forward 方法
   - 层循环中注入 TP/CP/MoE/DP-Attn 各组件的 simulate 调用
   - tick 末尾调用 EPLB maybe_rebalance
   - 更新并行指标收集
4. **修改 `server/src/sglang/scheduler/index.ts`**：PrefillManager/PrefillAdder 中集成 `dpController.select_rank_for_request`
5. **修改 `server/src/sglang/metrics/index.ts`**：新增 `toJSON()` 方法
6. **修改 `server/src/http/HttpService.ts`**：新增 `/api/internal/metrics` 路由处理
7. **修改 `server/src/sglang/index.ts`**：新增导出
8. **新增 `server/src/test/sglang-p6.test.ts`**：6 组端到端验收测试

---

## 测试设计

### 验收测试用例清单

**Case 1**: `size=1` 全部退化 noop → 与 Phase 4 纯 TP 串行一致延迟；误差 <1%

- 设置 `tpSize=1, dpSize=1, epSize=1, ppSize=1, cpSize=1`
- 调用 `initParallelGroups` 获取组件
- 各组件的方法调用均返回 0 / noop
- 通过 MockEngine forward 一批请求，验证总通信 ticks = 0
- 与 Phase 4 的纯串行 forward 对比，延迟误差 <1%

**Case 2**: `tp=2,dp=2,ep=2,pp=2,cp=2`（world_size=8）+ MLA + MoE → throughput > 2× decode tokens

- 配置 `tpSize=2, dpSize=2, epSize=2, ppSize=2, cpSize=2`，`isMoe=true`，`useMla=true`
- 验证 `initParallelGroups` 成功创建所有 9 组件（非 null）
- 运行 N tick 仿真，统计 decode token 总数
- 与 `size=1` 基准对比，吞吐量提升 > 2×

**Case 3**: DP Attention 开启 vs 关闭 → tbt 差 < 1.5×

- 同一模型配置，分别设置 `enableDpAttention=true` 和 `false`
- 运行仿真，记录 per-token decode latency（tbt）
- 验证两者 tbt 差值 < 1.5×

**Case 4**: PP 1f1b 的 bubble 为 gpipe 的 1/num_micro_batches 比例

- 配置 `ppSize=4, ppNumMicroBatches=4`
- 分别设置 `ppPipelineSchedule="1f1b"` 和 `"gpipe"`
- 验证 `1f1b_bubble / gpipe_bubble = 1 / numMicroBatches`

**Case 5**: 所有 `validateParallelConfig` 的 7 条 assert 覆盖

- 对 7 条约束分别构造违反配置，验证 `validateParallelConfig` 返回 `ok=false` 且 `errors` 包含对应约束消息
- 正常配置验证 `ok=true`

**Case 6**: TypeScript strict 0 any；`npm run build` strict zero error

- 确认 `tsconfig.json` 中 `strict: true`
- 确认新增代码中无 `any` 类型
- 运行 `npm run build` 无错误

### 边界条件覆盖

| 边界条件 | 测试方式 |
|----------|----------|
| `dpSize=1` 时 `dpController.select_rank` 始终返回 rank 0 | 单元断言 |
| `cpSize=1` 时 `cpSim` 为 null | 构造断言 |
| `enableEplb=false` 时 `eplbSim` 为 null | 构造断言 |
| `isMoe=false` 时 `moeBackend` 为 null | 构造断言 |
| `enableDpAttention=false` 时 `dpAttnSim` 为 null | 构造断言 |
| `validateParallelConfig` 失败时 `initParallelGroups` 抛异常 | try-catch 断言 |
| `numPages=0`（OOM）时 DP 分配均返回 null | DP 分配断言 |
| 极大 world_size（32）正常创建 | 构造断言 |

---

## 风险与注意事项

### 兼容性影响

- MockEngine 构造器签名变更：新增 `parallelGroups?` 可选参数，现有测试传入 `config` 的方式不受影响（向后兼容）
- `forwardBatchPP`/`forwardBatchSeq`/`forwardBatch` 合并为统一方法：需要更新已有测试调用方式，但已有测试可通过 `forwardBatch` 新签名适配
- ParallelMetrics.summary() 已存在，SimulationMetrics.toJSON() 为新增方法，不影响现有代码

### 性能影响

- 层循环中逐层调用各组件 simulate 方法，理论开销极低（纯算术运算），不会成为瓶颈
- EPLB maybe_rebalance 每 100 步检查一次，重平衡仅在方差超阈值时触发，平均开销可忽略

### 回滚方案

- 若 `initParallelGroups` 集成引入问题，MockEngine 保留内部独立创建路径作为 fallback（构造器 `parallelGroups` 参数可选，不传时走旧逻辑）
- 所有变更在 `issue-30` 分支，未合并前不影响 main
