// workload — S6: WorkloadGenerator/WorkloadConfig（§4.3 / §9.10）

import type { SimRequestMsg } from "../types";
import { SamplingParams } from "../core";

/** 工作负载生成器配置（对应 §4.4 WorkloadConfig / §9.10） */
export interface WorkloadConfig {
  /** 生成请求数量 */
  numRequests: number;
  /** 输入长度分布类型 */
  inputLenDistribution: "uniform" | "normal";
  /** 输入长度最小值 */
  inputLenMin: number;
  /** 输入长度最大值 */
  inputLenMax: number;
  /** 输入长度均值（normal 分布用） */
  inputLenMean?: number;
  /** 输入长度标准差（normal 分布用） */
  inputLenStd?: number;
  /** 输出长度分布类型 */
  outputLenDistribution: "uniform" | "normal";
  /** 输出长度最小值 */
  outputLenMin: number;
  /** 输出长度最大值 */
  outputLenMax: number;
  /** 输出长度均值（normal 分布用） */
  outputLenMean?: number;
  /** 输出长度标准差（normal 分布用） */
  outputLenStd?: number;
  /** 请求到达速率（每 tick 请求数） */
  arrivalRate: number;
  /** 到达分布类型 */
  arrivalDistribution: "poisson" | "uniform" | "trace";
  /** 共享前缀比例 */
  sharedPrefixRatio: number;
  /** 共享前缀长度 */
  sharedPrefixLen: number;
  /** Trace 数据（trace 模式用） */
  trace?: SimRequestMsg[];
}

export const DEFAULT_WORKLOAD_CONFIG: WorkloadConfig = {
  numRequests: 100,
  inputLenDistribution: "uniform",
  inputLenMin: 128,
  inputLenMax: 1024,
  outputLenDistribution: "uniform",
  outputLenMin: 100,
  outputLenMax: 1024,
  arrivalRate: 10.0,
  arrivalDistribution: "poisson",
  sharedPrefixRatio: 0.3,
  sharedPrefixLen: 100,
};

/** 带 arrivalTick 元数据的请求消息 */
export interface SimRequestWithArrival extends SimRequestMsg {
  arrivalTick: number;
}

/**
 * WorkloadGenerator — 工作负载生成器（§4.3 / §9.10）
 *
 * 支持 Poisson/固定/trace 三种到达分布和 uniform/normal 两种长度分布。
 * 生成 SimRequestMsg[] 并附带 arrivalTick 元数据，供 Simulator 按 tick 注入。
 */
export class WorkloadGenerator {
  private _rng: () => number;

  constructor(rng?: () => number) {
    this._rng = rng ?? Math.random;
  }

  /** 生成模拟请求序列 */
  generate(config: WorkloadConfig): SimRequestWithArrival[] {
    if (config.arrivalDistribution === "trace" && config.trace) {
      return config.trace.map((r, i) => ({ ...r, arrivalTick: i }));
    }

    const requests: SimRequestWithArrival[] = [];
    for (let i = 0; i < config.numRequests; i++) {
      const inputLen = this._sampleLen(
        config.inputLenDistribution,
        config.inputLenMin, config.inputLenMax,
        config.inputLenMean ?? 0, config.inputLenStd ?? 0,
      );
      const outputLen = this._sampleLen(
        config.outputLenDistribution,
        config.outputLenMin, config.outputLenMax,
        config.outputLenMean ?? 0, config.outputLenStd ?? 0,
      );
      const arrivalTick = this._sampleArrival(config, i);
      const inputIds = this._generateTokens(
        inputLen, config.sharedPrefixRatio, config.sharedPrefixLen, i,
      );

      requests.push({
        tag: "req_in",
        uid: i,
        inputIds,
        samplingParams: new SamplingParams({ maxNewTokens: outputLen }),
        outputLen,
        arrivalTick,
      });
    }

    // 按 arrivalTick 排序
    requests.sort((a, b) => a.arrivalTick - b.arrivalTick);
    return requests;
  }

  /** 采样长度值 */
  private _sampleLen(
    distribution: string,
    minVal: number,
    maxVal: number,
    mean: number,
    std: number,
  ): number {
    if (distribution === "normal" && std > 0) {
      const effectiveMean = mean > 0 ? mean : (minVal + maxVal) / 2;
      let val = this._boxMuller(effectiveMean, std);
      val = Math.max(minVal, Math.min(maxVal, Math.round(val)));
      return val;
    }
    // uniform
    return Math.floor(minVal + this._rng() * (maxVal - minVal + 1));
  }

  /** Box-Muller 变换生成正态分布随机数 */
  private _boxMuller(mean: number, std: number): number {
    const u1 = this._rng();
    const u2 = this._rng();
    const z = Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
    return mean + z * std;
  }

  /** 采样到达 tick（对齐 §9.10 伪代码） */
  private _sampleArrival(config: WorkloadConfig, index: number): number {
    if (config.arrivalRate <= 0) {
      return 0;
    }
    if (config.arrivalDistribution === "poisson") {
      // §9.10: arrivalTick = index / arrivalRate
      return Math.floor(index / config.arrivalRate);
    }
    // uniform: 均匀分布在 ticks 中
    if (config.arrivalRate > 0) {
      return Math.floor(index / config.arrivalRate);
    }
    return 0;
  }

  /** 生成 token ID 序列（含共享前缀策略，对齐 §9.10） */
  private _generateTokens(
    length: number,
    ratio: number,
    prefixLen: number,
    uid: number,
  ): number[] {
    const tokens: number[] = [];
    const useSharedPrefix = ratio > 0 && uid % 3 === 0;

    if (useSharedPrefix && prefixLen > 0 && prefixLen <= length) {
      // 共享前缀部分
      for (let i = 0; i < prefixLen; i++) {
        tokens.push(i % 256);
      }
      // 非共享部分
      for (let i = prefixLen; i < length; i++) {
        tokens.push((uid * 7 + i) % 256);
      }
    } else {
      // 无共享前缀
      for (let i = 0; i < length; i++) {
        tokens.push((uid * 13 + i) % 256);
      }
    }

    return tokens;
  }
}
