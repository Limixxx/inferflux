import type {
  SimulatorConfig,
  SimRequestMsg,
  SimRespMsg,
  TableManager,
  CacheManager,
  SimScheduler,
  SimCommGroup,
} from "./types";

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
