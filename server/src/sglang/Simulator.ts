import type {
  SimulatorConfig,
  SimRequestMsg,
  SimRespMsg,
  TableManager,
  CacheManager,
  SimScheduler,
  SimCommGroup,
} from "./types";

import { SimScheduler as SimSchedulerImpl, SimulationClock } from "./scheduler";
import { SimulationMetrics } from "./metrics";
import { WorkloadGenerator, type WorkloadConfig, type SimRequestWithArrival } from "./workload";
import { SGHttpApi } from "./api";
import { MockEngine } from "./engine";
import { CacheManager as CacheManagerImpl } from "./cache";
import { TableManager as TableManagerImpl } from "./scheduler";

/**
 * SGLang 仿真全局上下文（对应 §4.2 Context + 全局上下文工具函数）
 *
 * 设计原则：
 * - 属性初始为 null，后续 Issue 逐步注入实际实例
 * - newId() 严格单调递增，用于生成全局唯一请求 ID
 * - clock 为离散 tick 计数器，由外部 run_tick 驱动
 */
export class SgSimContext {
  readonly config: SimulatorConfig;

  // ===== 占位引用（后续 Issue 赋值） =====
  tableMgr: TableManager | null = null;
  cacheMgr: CacheManager | null = null;
  scheduler: SimScheduler | null = null;
  tpGroup: SimCommGroup | null = null;

  // ===== 基础设施 =====
  private _nextId: number = 0;
  private _clock: number = 0;

  constructor(config: SimulatorConfig) {
    this.config = config;
  }

  /** 生成全局唯一 ID（严格单调递增） */
  newId(): number {
    return ++this._nextId;
  }

  /** 当前 tick */
  get clock(): number {
    return this._clock;
  }

  /** 推进时钟（由外部 run_tick 调用） */
  advanceClock(ticks: number = 1): void {
    this._clock += ticks;
  }

  /** 重置上下文状态（用于 reset 场景） */
  reset(): void {
    this._nextId = 0;
    this._clock = 0;
    this.tableMgr = null;
    this.cacheMgr = null;
    this.scheduler = null;
    this.tpGroup = null;
  }
}

/**
 * Simulator 入口桩（S0 仅定义接口，S1 实现完整调度循环）
 */
export class Simulator {
  readonly ctx: SgSimContext;

  constructor(config: SimulatorConfig) {
    this.ctx = new SgSimContext(config);
  }

  /**
   * 执行一个调度 tick
   * @param incoming 进入的请求消息列表
   * @returns 响应消息列表
   */
  runTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    // S1 实现：消息分发 → 调度 → forward → 结果处理
    return [];
  }

  /** 重置仿真器状态 */
  reset(): void {
    this.ctx.reset();
  }
}

// ===== S6: SgSimInstance + createSimulator（§4.3 导出） =====

/**
 * SgSimInstance — 仿真器实例接口（§4.3 导出）
 */
export interface SgSimInstance {
  readonly ctx: SgSimContext;
  readonly scheduler: SimSchedulerImpl;
  readonly metrics: SimulationMetrics;
  readonly workload: WorkloadGenerator;
  readonly httpApi: SGHttpApi;

  /** 启动仿真循环（在线模式启动 interval；离线模式一次性运行） */
  start(): void;

  /** 入队一个请求（在线模式下推入 _incomingQueue，下一 tick 被 runTick 消费） */
  enqueue(msg: SimRequestMsg): void;

  /** 获取当前指标快照 */
  getMetrics(): Record<string, unknown>;

  /** 加载 workload 请求（在 start() 前调用） */
  loadWorkload(config: WorkloadConfig): void;

  /** 关闭仿真器（清除 interval，停止循环） */
  shutdown(): void;
}

/**
 * SgSimInstanceImpl — SgSimInstance 实现
 */
class SgSimInstanceImpl implements SgSimInstance {
  readonly ctx: SgSimContext;
  readonly scheduler: SimSchedulerImpl;
  readonly metrics: SimulationMetrics;
  readonly workload: WorkloadGenerator;
  readonly httpApi: SGHttpApi;

  private _intervalTimer: ReturnType<typeof setInterval> | null = null;
  private _incomingQueue: SimRequestMsg[] = [];
  private _workloadRequests: SimRequestWithArrival[] = [];
  private _currentTick: number = 0;
  private _workloadIdx: number = 0;
  private _stopped: boolean = false;
  // Track request arrival/first-token/finish for latency metrics
  private _requestArrivalTick: Map<number, number> = new Map();
  private _requestFirstTokenTick: Map<number, number> = new Map();

  constructor(config: SimulatorConfig) {
    this.ctx = new SgSimContext(config);
    this.metrics = new SimulationMetrics();
    this.workload = new WorkloadGenerator();

    const engine = new MockEngine(config);
    const clock = (config.enableOverlap && config.enableMetrics)
      ? new SimulationClock()
      : undefined;

    this.scheduler = new SimSchedulerImpl(config, {
      engine,
      simMetrics: this.metrics,
      clock,
    });

    this.httpApi = new SGHttpApi();
    this.httpApi.bind(this.scheduler, this.metrics);

    // 注入上下文引用
    this.ctx.scheduler = this.scheduler as any;
    this.ctx.cacheMgr = this.scheduler.cacheManager as any;
    this.ctx.tableMgr = this.scheduler.tableManager as any;

    // 注册 SimulationClock tick 回调
    if (clock && config.enableMetrics) {
      clock.onTick((tick) => {
        this.metrics.tick(tick);
      });
    }
  }

  start(): void {
    const config = this.ctx.config;
    this._stopped = false;

    if (config.offlineMode) {
      // 离线模式：预加载 workload 后循环运行所有 tick
      if (this._workloadRequests.length === 0) {
        // 无 workload，立即完成
        return;
      }
      const maxTicks = config.maxTicks ?? 10000;
      for (let t = 0; t < maxTicks && !this._stopped; t++) {
        this._runOneTick(t);
      }
    } else {
      // 在线模式：启动 setInterval
      const intervalMs = config.tickIntervalMs || 10;
      this._currentTick = 0;
      this._intervalTimer = setInterval(() => {
        if (this._stopped) return;
        this._runOneTick(this._currentTick);
        this._currentTick += 1;
        if (config.maxTicks !== null && this._currentTick >= config.maxTicks) {
          this.shutdown();
        }
      }, intervalMs);
    }
  }

  enqueue(msg: SimRequestMsg): void {
    if (this._stopped) return;
    this._incomingQueue.push(msg);
    this._requestArrivalTick.set(msg.uid, this._currentTick);
  }

  getMetrics(): Record<string, unknown> {
    return this.metrics.toJSON();
  }

  shutdown(): void {
    this._stopped = true;
    if (this._intervalTimer !== null) {
      clearInterval(this._intervalTimer);
      this._intervalTimer = null;
    }
  }

  /** 加载 workload 请求（在 start() 前调用） */
  loadWorkload(config: WorkloadConfig): void {
    this._workloadRequests = this.workload.generate(config);
    this._workloadIdx = 0;
    this.metrics.totalRequests = this._workloadRequests.length;
  }

  /** 执行一个 tick */
  private _runOneTick(tick: number): void {
    // 收集当前 tick 应到达的 workload 请求
    const incoming: SimRequestMsg[] = [...this._incomingQueue];
    this._incomingQueue.length = 0;

    while (
      this._workloadIdx < this._workloadRequests.length &&
      this._workloadRequests[this._workloadIdx].arrivalTick <= tick
    ) {
      const req = this._workloadRequests[this._workloadIdx];
      incoming.push(req);
      if (!this._requestArrivalTick.has(req.uid)) {
        this._requestArrivalTick.set(req.uid, req.arrivalTick);
      }
      this._workloadIdx++;
    }

    // 重置 lastForwardOutput，确保仅反映当前 tick 的 forward
    this.scheduler.lastForwardOutput = null;

    // 执行调度器 tick
    const replies = this.scheduler.runTick(incoming);

    // 记录指标
    if (replies.length > 0) {
      this.metrics.recordReply(replies, tick);
      this._trackLatencies(replies, tick);
    }

    // 计算 GPU busy：使用 ForwardOutput 中的精确时间模型判断本 tick GPU 是否忙碌
    // 对齐 §9.11：基于 prefillBatchTime + decodeBatchTime 判断（而非粗略的恒为 1）
    const gpuBusy = this._extractGpuBusy();
    this.metrics.recordTick(tick, gpuBusy);
  }

  /** 从 scheduler 最近一次 forward 中判断本 tick GPU 是否忙碌（0 或 1） */
  private _extractGpuBusy(): number {
    // 通过 scheduler 的内部状态获取当前 tick 的 forward 时间信息
    const lastForwardOutput = this.scheduler.lastForwardOutput;
    if (lastForwardOutput === null) return 0;
    const prefillTime = lastForwardOutput.prefillBatchTime ?? 0;
    const decodeTime = lastForwardOutput.decodeBatchTime ?? 0;
    // 精确时间模型：有非零 forward cost 表示 GPU 本 tick 处于忙碌状态
    return (prefillTime + decodeTime) > 0 ? 1 : 0;
  }

  /** 追踪请求延迟 */
  private _trackLatencies(replies: SimRespMsg[], tick: number): void {
    for (const reply of replies) {
      if (reply.nextToken !== null && !this._requestFirstTokenTick.has(reply.uid)) {
        this._requestFirstTokenTick.set(reply.uid, tick);
      }
      if (reply.finished) {
        const arrivalTick = this._requestArrivalTick.get(reply.uid) ?? 0;
        const firstTokenTick = this._requestFirstTokenTick.get(reply.uid) ?? tick;
        // decodeSteps = tick - firstTokenTick (至少 1)
        const decodeSteps = Math.max(1, tick - firstTokenTick);
        this.metrics.recordRequestLatency(
          reply.uid, arrivalTick, firstTokenTick, tick, decodeSteps
        );
        this._requestArrivalTick.delete(reply.uid);
        this._requestFirstTokenTick.delete(reply.uid);
      }
    }
  }
}

/**
 * 创建仿真器实例（§4.3 导出函数）
 */
export function createSimulator(config: SimulatorConfig): SgSimInstance {
  return new SgSimInstanceImpl(config);
}
