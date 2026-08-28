// parallel — P0: SimCommGroup/ParallelTopology/ParallelMetrics + P5: CPSimulator
// parallel — P0: SimCommGroup/ParallelTopology/ParallelMetrics + P1a: TPSimulator/TPCommInfraSimulator

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
  CPSimulator,
  CPAttnResult,
} from "./cp_simulator";
// P2a: DataParallelController
export {
  DPRankState,
  DataParallelController,
} from "./dp_controller";
export {
  DPAttentionSimulator,
  DPAttentionSimulatorOpts,
} from "./dp_attn";
// P1a: TP 张量并行仿真
export {
  TPSimulator,
} from "./tp_simulator";

export {
  TPCommInfraSimulator,
} from "./tp_comm_infra";
// P1b: 并行组合内存预算 + 配置验证
export {
  ParallelMemoryCorrections,
  calculateMemoryBudgetParallel,
} from "./budget";

export {
  ValidationResult,
  validateParallelConfig,
} from "./validate";
