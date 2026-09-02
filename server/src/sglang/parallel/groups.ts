// parallel/groups.ts — P6: ParallelGroups 接口与 initParallelGroups 工厂函数（§10.7）

import type { SimulatorConfig, ModelConfig } from "../types";
import { divEven } from "../core";
import { ParallelTopology } from "./topology";
import { TPCommInfraSimulator } from "./tp_comm_infra";
import { TPSimulator } from "./tp_simulator";
import { DataParallelController } from "./dp_controller";
import { DPAttentionSimulator } from "./dp_attn";
import { PPPipelineSimulator } from "./pp";
import { CPSimulator } from "./cp_simulator";
import { EPLBSimulator } from "./eplb";
import { SimMoeBackend } from "./moe";
import { SimCommGroup } from "./comm_group";
import { validateParallelConfig } from "./validate";
import { calculateMemoryBudgetParallel } from "./budget";
import type { ParallelMetrics } from "./metrics";

// ===== ParallelGroups 接口 =====

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

// ===== InitParallelGroupsOpts =====

/** initParallelGroups 构造选项 */
export interface InitParallelGroupsOpts {
  config: SimulatorConfig;
  modelConfig: ModelConfig;
  numPages: number;
  metrics: ParallelMetrics;
}

// ===== initParallelGroups 工厂函数 =====

/**
 * 统一初始化并行组件（§10.7 init_parallel_groups）
 *
 * 步骤：
 * 1. validateParallelConfig — 失败直接 throw
 * 2. 创建 ParallelTopology
 * 3. 按条件创建 9 组件
 *
 * @param opts - 构造选项
 * @returns ParallelGroups 实例
 * @throws Error — 配置验证失败时抛出
 */
export function initParallelGroups(opts: InitParallelGroupsOpts): ParallelGroups {
  const { config, modelConfig, numPages, metrics } = opts;

  // 步骤 1：验证并行配置
  const validation = validateParallelConfig(config, modelConfig);
  if (!validation.ok) {
    throw new Error(
      `initParallelGroups: parallel config validation failed:\n${validation.errors.join("\n")}`
    );
  }

  // 步骤 2：创建并行拓扑
  const topology = new ParallelTopology({
    tpSize: config.tpSize,
    dpSize: config.dpSize,
    epSize: config.epSize,
    ppSize: config.ppSize,
    cpSize: config.cpSize,
    enableDpAttention: config.enableDpAttention,
  });

  // 步骤 3：按条件创建 9 组件

  // 1. topology（已创建）

  // 2. TPCommInfraSimulator
  const tpComm = new TPCommInfraSimulator(config, modelConfig);

  // 3. TPSimulator
  const tpSim = new TPSimulator(config, modelConfig);

  // 4. DataParallelController — 使用 divEven(numPages, dpSize) 分配
  const dpController = new DataParallelController(
    config.dpSize,
    numPages,
    config.dpLoadBalanceStrategy,
  );

  // 5. DPAttentionSimulator — 仅 enableDpAttention && useMla 时创建
  const dpAttnSim = (config.enableDpAttention && (modelConfig.useMla ?? false))
    ? new DPAttentionSimulator({
        dpSize: config.dpSize,
        hiddenSize: modelConfig.hiddenSize,
        dtypeSize: config.dtypeSize,
        useMla: modelConfig.useMla ?? false,
        enableDpAttention: config.enableDpAttention,
        networkBandwidthGBps: config.networkBandwidthGBps,
        networkLatencyUs: config.networkLatencyUs,
      })
    : null;

  // 6. PPPipelineSimulator
  const ppSim = new PPPipelineSimulator(config, modelConfig);

  // 7. CPSimulator — 仅 cpSize > 1 时创建
  const cpSim = config.cpSize > 1
    ? new CPSimulator(config, modelConfig)
    : null;

  // 8. EPLBSimulator — 仅 enableEplb 时创建
  const eplbSim = config.enableEplb
    ? new EPLBSimulator({
        enabled: true,
        numExperts: modelConfig.numExperts,
        epSize: config.epSize,
        metrics,
      })
    : null;

  // 9. SimMoeBackend — 仅 isMoe 时创建
  const moeBackend = modelConfig.isMoe
    ? new SimMoeBackend({
        modelConfig,
        topology,
        config,
        epCommGroup: new SimCommGroup({
          groupType: "ep",
          size: config.epSize,
          networkBandwidthGBps: config.networkBandwidthGBps,
          latencyUs: config.networkLatencyUs,
          efficiency: config.epEfficiency,
        }),
        metrics,
        seed: config.moeRoutingSeed,
      })
    : null;

  return {
    topology,
    tpComm,
    tpSim,
    dpController,
    dpAttnSim,
    ppSim,
    cpSim,
    eplbSim,
    moeBackend,
  };
}
