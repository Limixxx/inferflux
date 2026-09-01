// parallel — P0: SimCommGroup/ParallelTopology/ParallelMetrics + P4: PPPipelineSimulator
// parallel — P0: SimCommGroup/ParallelTopology/ParallelMetrics + P5: CPSimulator
// parallel — P0: SimCommGroup/ParallelTopology/ParallelMetrics
// parallel — P3a: SimMoeBackend
// parallel — P3b: EPLBSimulator
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
  PipelineStepResult,
  PPPipelineSimulator,
} from "./pp";
export {
  CPSimulator,
  CPAttnResult,
} from "./cp_simulator";
export {
  SimMoeBackend,
} from "./moe";

export type {
  SimMoeBackendOpts,
  MoeRouteResult,
  MoeForwardResult,
} from "./moe";
// P3b: EPLBSimulator
export {
  EPLBSimulator,
} from "./eplb";

export type {
  EPLBSimulatorOpts,
  RebalanceResult,
} from "./eplb";
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
// P6: ParallelGroups + initParallelGroups
export type {
  ParallelGroups,
  InitParallelGroupsOpts,
} from "./groups";

export {
  initParallelGroups,
} from "./groups";
