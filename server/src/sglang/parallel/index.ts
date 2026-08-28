// parallel — P0: SimCommGroup/ParallelTopology/ParallelMetrics + P1b: budget/validate

export {
  CommGroupType,
  SimCommGroupOpts,
  SimCommGroup,
  MockTPGroup,
} from "./comm_group";

export {
  ParallelTopologyOpts,
  ParallelTopology,
} from "./topology";

export {
  ParallelMetrics,
} from "./metrics";

export {
  DPAttentionSimulator,
  DPAttentionSimulatorOpts,
} from "./dp_attn";
// P1b: 并行组合内存预算 + 配置验证
export {
  ParallelMemoryCorrections,
  calculateMemoryBudgetParallel,
} from "./budget";

export {
  ValidationResult,
  validateParallelConfig,
} from "./validate";
