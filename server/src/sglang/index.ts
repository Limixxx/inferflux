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

// K1: cache 抽象层 + K5 内存预算 + K2 实现
export {
  // K1 抽象类
  CacheSizeInfo,
  BaseCacheHandle,
  MatchResult,
  InsertResult,
  BaseKVCachePool,
  BasePrefixCache,
  // K5 内存预算公式
  MemoryBudgetResult,
  estimateModelMemory,
  estimateGraphBuffer,
  calculateMemoryBudget,
  // K2 实现
  MockKVCachePool,
  PageAllocation,
  NaivePrefixCache,
  NaiveCacheHandle,
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

// P1b: 并行组合内存预算 + 配置验证
export {
  calculateMemoryBudgetParallel,
  validateParallelConfig,
  ParallelMemoryCorrections,
  ValidationResult,
} from "./parallel";

// P0: 仿真指标集合
export {
  SimulationMetrics,
} from "./metrics";
