export {
  SimMode,
  ModelConfig,
  DEFAULT_MODEL_CONFIG,
  SimulatorConfig,
  DEFAULT_SIMULATOR_CONFIG,
  SamplingParams,
  SimRequestMsgTag,
  SimRequestMsg,
  SimRespMsgTag,
  SimRespMsg,
  TableManager,
  BaseCacheHandle,
  MatchResult,
  InsertResult,
  CacheManager,
  SimScheduler,
  SimCommGroup,
  CommGroupType,
} from "./types";

export type { SamplingDtype, SamplingParamsOpts } from "./types";

export { SgSimContext, Simulator } from "./Simulator";

// S1: core 数据结构与工具函数
export {
  alignDown,
  divCeil,
  divEven,
  bytesPerElement,
  Req,
  Batch,
} from "./core";

export type { ReqOpts } from "./core";

// S1: entities
export {
  ChunkedReq,
  PendingReq,
} from "./entities";

export type { PendingReqOpts } from "./entities";

// K5: 内存预算基础公式 (§3.3.9)
export {
  MemoryBudgetResult,
  estimateModelMemory,
  estimateGraphBuffer,
  calculateMemoryBudget,
} from "./cache";

// P0: 并行仿真基础设施
export {
  SimCommGroup as SimCommGroupImpl,
  SimCommGroupOpts,
  MockTPGroup,
  ParallelTopology,
  ParallelTopologyOpts,
  ParallelMetrics,
} from "./parallel";

// P0: 仿真指标集合
export {
  SimulationMetrics,
} from "./metrics";
