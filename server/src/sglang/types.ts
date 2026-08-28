// ===== SGLang Simulator — 顶层配置与消息类型 (S0) =====

/** 仿真器运行模式 */
export type SimMode = "agg" | "pd-disagg" | "parallel";

/** 模型特征描述（对应 §2.2 + §4.2 ModelConfig） */
export interface ModelConfig {
  numLayers: number;
  hiddenSize: number;
  numKvHeads: number;
  headDim: number;
  vocabSize: number;
  isMoe: boolean;
  numExperts: number;
  moeIntermediateSize: number;
  moeTopK: number;
  intermediateSize: number;
  numAttentionHeads: number;
  rmsNormEps: number;
  ropeTheta: number;
  maxPositionEmbeddings: number;
  useMla?: boolean;
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  numLayers: 32,
  hiddenSize: 4096,
  numKvHeads: 8,
  headDim: 128,
  vocabSize: 128256,
  isMoe: false,
  numExperts: 0,
  moeIntermediateSize: 0,
  moeTopK: 1,
  intermediateSize: 0,
  numAttentionHeads: 0,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  maxPositionEmbeddings: 8192,
};

/** 仿真器统一配置（对应 §4.2 SimulatorConfig） */
export interface SimulatorConfig {
  // ===== 模型配置 =====
  modelConfig: ModelConfig;

  // ===== 调度配置 =====
  maxRunningReq: number;
  maxSeqLen: number;
  maxExtendTokens: number;
  cacheType: "radix" | "naive";

  // ===== KV Cache 配置 =====
  pageSize: number;
  numPages: number | null;

  // ===== 内存配置 =====
  totalGpuMemory: number;
  memoryRatio: number;
  dtypeSize: number;

  // ===== CUDA Graph 配置 =====
  enableCudaGraph: boolean;
  cudaGraphBs: number[] | null;
  cudaGraphMaxBs: number | null;
  graphReplayCostTicks: number;
  eagerForwardCostTicks: number;

  // ===== Overlap Scheduling 配置 =====
  enableOverlap: boolean;
  cpuScheduleCostTicks: number;
  cpuProcessResultCostTicks: number;

  // ===== TP 张量并行配置 =====
  tpSize: number;
  allReduceCostPerByteTicks: number;
  allReduceLatencyTicks: number;
  tpCpuGroupType: string;
  tpGpuGroupType: string;

  // ===== DP 数据并行配置 =====
  dpSize: number;
  dpLoadBalanceStrategy: "round_robin" | "shortest_queue";
  enableDpAttention: boolean;
  dpAttentionAllGatherCostPerByteTicks: number;

  // ===== EP 专家并行配置 =====
  epSize: number;
  allToAllCostPerByteTicks: number;
  allToAllLatencyTicks: number;
  moeRoutingMode: "mock" | "hash" | "simulated";
  moeRoutingSeed?: number;
  enableEplb: boolean;

  // ===== CP Context Parallel 配置 =====
  cpSize: number;
  cpAllGatherCostPerByteTicks: number;

  // ===== PP 流水并行配置 =====
  ppSize: number;
  ppNumMicroBatches: number;
  ppSendRecvCostPerByteTicks: number;
  ppPipelineSchedule: "1f1b" | "gpipe" | "interleaved";

  // ===== 通信成本通用配置 =====
  commBandwidthBytesPerTick: number;
  commOverlapWithCompute: boolean;

  // ===== 并行通信统一参数（P0 新增）=====
  networkBandwidthGBps: number;
  networkLatencyUs: number;
  tpEfficiency: number;
  epEfficiency: number;
  cpEfficiency: number;

  // ===== 离线模式 =====
  offlineMode: boolean;

  // ===== Tokenizer =====
  eosTokenId: number;

  // ===== 采样配置 =====
  mockSampleMode: "random" | "greedy" | "fixed";
  fixedOutputToken: number;

  // ===== 仿真控制 =====
  maxTicks: number | null;
  logLevel: string;
  enableMetrics: boolean;
}

export const DEFAULT_SIMULATOR_CONFIG: SimulatorConfig = {
  modelConfig: DEFAULT_MODEL_CONFIG,
  maxRunningReq: 128,
  maxSeqLen: 8192,
  maxExtendTokens: 8192,
  cacheType: "radix",
  pageSize: 1,
  numPages: null,
  totalGpuMemory: 80 * 1024 ** 3,
  memoryRatio: 0.88,
  dtypeSize: 2,
  enableCudaGraph: true,
  cudaGraphBs: null,
  cudaGraphMaxBs: null,
  graphReplayCostTicks: 1,
  eagerForwardCostTicks: 10,
  enableOverlap: true,
  cpuScheduleCostTicks: 1,
  cpuProcessResultCostTicks: 1,
  tpSize: 1,
  allReduceCostPerByteTicks: 0.001,
  allReduceLatencyTicks: 2,
  tpCpuGroupType: "gloo",
  tpGpuGroupType: "nccl",
  dpSize: 1,
  dpLoadBalanceStrategy: "round_robin",
  enableDpAttention: false,
  dpAttentionAllGatherCostPerByteTicks: 0.0015,
  epSize: 1,
  allToAllCostPerByteTicks: 0.002,
  allToAllLatencyTicks: 3,
  moeRoutingMode: "mock",
  enableEplb: false,
  cpSize: 1,
  cpAllGatherCostPerByteTicks: 0.001,
  ppSize: 1,
  ppNumMicroBatches: 1,
  ppSendRecvCostPerByteTicks: 0.0005,
  ppPipelineSchedule: "1f1b",
  commBandwidthBytesPerTick: 1_000_000,
  commOverlapWithCompute: true,
  networkBandwidthGBps: 100,
  networkLatencyUs: 5,
  tpEfficiency: 0.95,
  epEfficiency: 0.90,
  cpEfficiency: 0.90,
  offlineMode: false,
  eosTokenId: 0,
  mockSampleMode: "random",
  fixedOutputToken: 0,
  maxTicks: null,
  logLevel: "INFO",
  enableMetrics: true,
};

// 采样参数（S1 升级为 class，S0 的 interface 替换为类型别名）
import { SamplingParams as SamplingParamsClass, SamplingDtype, SamplingParamsOpts } from "./core";
export { SamplingParamsClass as SamplingParams };
export type { SamplingDtype, SamplingParamsOpts };

/** 请求消息标签 */
export type SimRequestMsgTag = "req_in" | "req_resume";

/** 请求进入/续接消息（对应 §4.3 接口） */
export interface SimRequestMsg {
  tag: SimRequestMsgTag;
  uid: number;
  inputIds: number[];
  samplingParams: SamplingParamsClass | null;
  outputLen: number;
}

/** 响应消息标签 */
export type SimRespMsgTag = "resp_token" | "resp_done" | "resp_reject";

/** 响应消息（对应 §4.3 接口） */
export interface SimRespMsg {
  tag: SimRespMsgTag;
  uid: number;
  nextToken: number | null;
  finished: boolean;
  reason?: string;
}

// ===== K1 类型导入 =====

import { TableManager } from "./scheduler";
export { TableManager };

// K3: CacheManager class 引用
export { CacheManager } from "./cache";

// ===== 占位接口（后续 Issue 实现） =====

/** 调度器桩（S1 实现） */
export interface SimScheduler {
  runTick(incoming: SimRequestMsg[]): SimRespMsg[];
}

/** 通信组类型标识（P0 新增） */
export type CommGroupType = "tp" | "ep" | "pp" | "cp" | "dp_attn";

/** 通信组（P0 完整实现，size=1 时为 noop） */
export interface SimCommGroup {
  readonly groupType: CommGroupType;
  readonly size: number;
  allReduce(tensorBytes: number): number;
  allGather(sizes: number[]): number;
  allToAll(sendSizes: number[], recvSizes: number[]): number;
  sendRecv(bytes: number, peer: number): number;
  barrier(): void;
}
