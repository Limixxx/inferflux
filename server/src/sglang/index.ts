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
  SimScheduler,
  SimCommGroup,
  CommGroupType,
  TableManager,
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
  ForwardOutput,
} from "./core";

export type { ReqOpts } from "./core";

// S1: entities
export {
  ChunkedReq,
  PendingReq,
} from "./entities";

export type { PendingReqOpts } from "./entities";

// K1: cache 抽象层 + K5 内存预算 + K2 实现 + K3 CacheManager
// K1: cache 抽象层 + K5 内存预算 + K2 实现 + K4 RadixPrefixCache
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
  // K3 CacheManager
  CacheManager,
  // K4 实现
  RadixTreeNode,
  RadixCacheHandle,
  RadixPrefixCache,
} from "./cache";

// P0: 并行仿真基础设施 + P4: PPPipelineSimulator
// P0+P5: 并行仿真基础设施 + CPSimulator
// P0: 并行仿真基础设施 + P2a: DataParallelController
export type { KeyFn } from "./cache";

// P0: 并行仿真基础设施
// P0: 并行仿真基础设施 + P1a: TP 张量并行仿真
export {
  SimCommGroup as SimCommGroupImpl,
  SimCommGroupOpts,
  MockTPGroup,
  ParallelTopology,
  ParallelTopologyOpts,
  ParallelMetrics,
  PPPipelineSimulator,
  CPSimulator,
  CPAttnResult,
  SimMoeBackend,
} from "./parallel";

export type {
  SimMoeBackendOpts,
  MoeRouteResult,
  MoeForwardResult,
  DPRankState,
  DataParallelController,
  TPSimulator,
  TPCommInfraSimulator,
} from "./parallel";

// P2b: DP Attention 仿真器
export {
  DPAttentionSimulator,
  DPAttentionSimulatorOpts,
// P1b: 并行组合内存预算 + 配置验证
export {
  calculateMemoryBudgetParallel,
  validateParallelConfig,
  ParallelMemoryCorrections,
  ValidationResult,
} from "./parallel";

export type { PipelineStepResult } from "./parallel";

// P0: 仿真指标集合
export {
  SimulationMetrics,
} from "./metrics";

// P4: 仿真引擎
export {
  GraphRunner,
  Sampler,
// P3a: MockEngine（含 MoE 集成）
export {
  MockEngine,
} from "./engine";
