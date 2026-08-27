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

// K1: cache 抽象层
export {
  CacheSizeInfo,
  BaseCacheHandle as BaseCacheHandleClass,
  MatchResult as MatchResultClass,
  InsertResult as InsertResultClass,
  BaseKVCachePool,
  BasePrefixCache,
} from "./cache";
