// api — S6: SGHttpApi 仿真 HTTP API 消息处理器（§4.3）

import type { SimScheduler } from "../scheduler";
import type { SimulationMetrics } from "../metrics";
import type { SimRequestMsg } from "../types";
import { SamplingParams } from "../core";

/** OpenAI ChatCompletion 请求格式 */
export interface ChatCompletionRequest {
  model?: string;
  messages: Array<{ role: string; content: string }>;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
}

/** OpenAI ChatCompletion 响应格式 */
export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: string; content: string };
    finish_reason: string | null;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * SGHttpApi — SGLang 仿真 HTTP API 消息处理器（§4.3）
 *
 * 不独立监听端口，而是作为 HttpService 和 SimService 的路由处理器。
 * 在 HttpService 中注册 /v1/* 路由时，委托此类处理请求逻辑。
 */
export class SGHttpApi {
  private _scheduler: SimScheduler | null = null;
  private _metrics: SimulationMetrics | null = null;
  private _nextReqId: number = 0;

  /** 注入调度器和指标实例 */
  bind(scheduler: SimScheduler, metrics: SimulationMetrics): void {
    this._scheduler = scheduler;
    this._metrics = metrics;
  }

  /** 处理 POST /v1/chat/completions 请求体，返回 OpenAI 格式占位响应 */
  handleChatCompletions(body: ChatCompletionRequest): ChatCompletionResponse {
    if (!this._scheduler) {
      throw new Error("SGHttpApi not bound to scheduler");
    }

    const reqId = this._nextReqId++;

    // 提取 messages → 拼接为文本 → 生成 inputIds（简单 token 计数）
    const text = body.messages.map(m => m.content).join(" ");
    const inputIds = Array.from({ length: text.length }, (_, i) => i % 256);
    const maxTokens = body.max_tokens ?? 128;

    // 构造请求消息并注入调度器
    const msg: SimRequestMsg = {
      tag: "req_in",
      uid: reqId,
      inputIds,
      samplingParams: new SamplingParams({ maxNewTokens: maxTokens }),
      outputLen: maxTokens,
    };

    this._scheduler.addRequest(msg);

    // 立即返回占位响应
    return {
      id: `chatcmpl-${reqId}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: body.model ?? "sglang-sim",
      choices: [{
        index: 0,
        message: { role: "assistant", content: "" },
        finish_reason: null,
      }],
      usage: {
        prompt_tokens: inputIds.length,
        completion_tokens: 0,
        total_tokens: inputIds.length,
      },
    };
  }

  /** 处理 GET /v1/internal/metrics，返回 metrics.toJSON() + 调度器快照 */
  handleInternalMetrics(): Record<string, unknown> {
    const result: Record<string, unknown> = {};

    if (this._metrics) {
      Object.assign(result, this._metrics.toJSON());
    }

    if (this._scheduler) {
      result.scheduler = {
        pendingReqs: this._scheduler.prefillManager.pendingList.length,
        runningReqs: this._scheduler.decodeManager.runningReqs.size,
        availableTableIndices: this._scheduler.tableManager.availableSize,
        tickCounter: this._scheduler.tickCounter,
        globalStep: this._scheduler.globalStep,
      };
    }

    return result;
  }

  /** 处理 GET /v1/internal/state，返回调度器状态快照 */
  handleInternalState(): Record<string, unknown> {
    if (!this._scheduler) {
      return { error: "Scheduler not available" };
    }

    const scheduler = this._scheduler;
    return {
      pendingReqs: scheduler.prefillManager.pendingList.length,
      runningReqs: scheduler.decodeManager.runningReqs.size,
      availableTableIndices: scheduler.tableManager.availableSize,
      tickCounter: scheduler.tickCounter,
      globalStep: scheduler.globalStep,
      cacheSizeInfo: scheduler.cacheManager.availableSize,
      overlapEnabled: scheduler.overlapEnabled,
    };
  }
}
