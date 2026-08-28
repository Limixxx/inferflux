// parallel/validate.ts — P1b: 并行配置合法性验证 validateParallelConfig (§10.7.3)

import type { ModelConfig, SimulatorConfig } from "../types";
import { ParallelTopology } from "./topology";

/** 验证结果结构 */
export interface ValidationResult {
  ok: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * 验证并行配置合法性
 * 对应 §10.7.3 validate_parallel_config，7 条约束 + 1 条警告
 *
 * @param config - SimulatorConfig（含并行配置）
 * @param modelConfig - ModelConfig（含模型特征参数）
 * @returns ValidationResult（ok/errors/warnings）
 */
export function validateParallelConfig(
  config: SimulatorConfig,
  modelConfig: ModelConfig,
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const tpSize = config.tpSize;
  const dpSize = config.dpSize;
  const epSize = config.epSize;
  const ppSize = config.ppSize;
  const cpSize = config.cpSize;

  // 约束 1：world_size === tp_size × dp_size × pp_size
  const expectedWorldSize = tpSize * dpSize * ppSize;
  const topo = new ParallelTopology({
    tpSize,
    dpSize,
    epSize,
    ppSize,
    cpSize,
    enableDpAttention: config.enableDpAttention,
  });
  const actualWorldSize = topo.worldSize;
  if (actualWorldSize !== expectedWorldSize || expectedWorldSize < 1) {
    errors.push(
      `Constraint 1: world_size (${actualWorldSize}) must equal tp_size (${tpSize}) × dp_size (${dpSize}) × pp_size (${ppSize}) = ${expectedWorldSize} and be >= 1`
    );
  }

  // 约束 2：ep_size >= 1 && (ep_size === 1 || model.isMoe)
  if (epSize < 1) {
    errors.push(`Constraint 2: ep_size (${epSize}) must be >= 1`);
  } else if (epSize > 1 && !modelConfig.isMoe) {
    errors.push(`Constraint 2: ep_size (${epSize}) > 1 but model is not MoE (isMoe=${modelConfig.isMoe})`);
  }

  // 约束 3：tp_size % cp_size === 0
  if (tpSize % cpSize !== 0) {
    errors.push(`Constraint 3: tp_size (${tpSize}) % cp_size (${cpSize}) = ${tpSize % cpSize}, must be 0`);
  }

  // 约束 4：(tp_size / cp_size) % ep_size === 0
  if (cpSize > 0 && tpSize % cpSize === 0) {
    const tpPerCp = tpSize / cpSize;
    if (tpPerCp % epSize !== 0) {
      errors.push(`Constraint 4: (tp_size/cp_size) (${tpPerCp}) % ep_size (${epSize}) = ${tpPerCp % epSize}, must be 0`);
    }
  }

  // 约束 5：pp_size >= 1 && pp_stage_layers 所有阶段层数 >= 1
  if (ppSize < 1) {
    errors.push(`Constraint 5: pp_size (${ppSize}) must be >= 1`);
  } else {
    const stages = topo.ppStageLayers(modelConfig.numLayers);
    const zeroStages = stages.filter(s => s.end - s.start < 1);
    if (zeroStages.length > 0) {
      errors.push(
        `Constraint 5: pp_size (${ppSize}) > numLayers (${modelConfig.numLayers}), ` +
        `${zeroStages.length} stage(s) have 0 layers`
      );
    }
  }

  // 约束 6：dp_size >= 1 && (enable_dp_attention → model.useMla)
  if (dpSize < 1) {
    errors.push(`Constraint 6: dp_size (${dpSize}) must be >= 1`);
  } else if (config.enableDpAttention && !modelConfig.useMla) {
    errors.push(`Constraint 6: enableDpAttention=true but model does not use MLA (useMla=${modelConfig.useMla ?? false})`);
  }

  // 约束 7：mem_fraction > 0 && mem_fraction <= 1
  if (config.memoryRatio <= 0 || config.memoryRatio > 1) {
    errors.push(`Constraint 7: memoryRatio (${config.memoryRatio}) must be > 0 and <= 1`);
  }

  // 警告：(kv_heads * cp_size) % tp_size !== 0
  if ((modelConfig.numKvHeads * cpSize) % tpSize !== 0) {
    warnings.push(
      `Warning: (numKvHeads × cpSize) = ${modelConfig.numKvHeads * cpSize} is not divisible by tpSize (${tpSize}), ` +
      `KV heads may not distribute evenly across TP ranks`
    );
  }

  return {
    ok: errors.length === 0,
    errors,
    warnings,
  };
}
