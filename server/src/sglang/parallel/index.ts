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

// P1a: TP 张量并行仿真
export {
  TPSimulator,
} from "./tp_simulator";

export {
  TPCommInfraSimulator,
} from "./tp_comm_infra";
