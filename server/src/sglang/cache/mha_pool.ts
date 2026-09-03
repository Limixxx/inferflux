// mha_pool — K2: MockKVCachePool + PageAllocation (§3.4.2 / §9.3b)

import type { ModelConfig, SimulatorConfig } from "../types";
import { BaseKVCachePool } from "./base";
import { divEven } from "../core";

/** 页分配结果，封装分配的页索引、槽位和总数 */
export class PageAllocation {
  /** 分配的页索引列表（每页起始 token 位置） */
  readonly pages: number[];
  /** 分配的槽位列表（展开后的 token 级别位置） */
  readonly slots: number[];
  /** 分配的槽位总数 */
  readonly slotCount: number;

  constructor(pages: number[], slots: number[], slotCount: number) {
    this.pages = pages;
    this.slots = slots;
    this.slotCount = slotCount;
  }
}

/** Mock KV cache 存储池（§3.4.2 / §9.3b） — 仅做内存记账，不存真实 tensor */
export class MockKVCachePool extends BaseKVCachePool {
  private readonly _numPages: number;
  private readonly _pageSize: number;
  private readonly _numLayers: number;
  private readonly _headDim: number;
  private readonly _numKvHeads: number;
  private readonly _cachePerPage: number;
  private _usedPages: number = 0;

  /** 空闲页池：存储每页的起始 token 位置 */
  freePagesPool: number[];

  constructor(
    modelConfig: ModelConfig,
    numPages: number,
    pageSize: number,
    config?: SimulatorConfig,
  ) {
    super();
    this._numPages = numPages;
    this._pageSize = pageSize;
    this._numLayers = modelConfig.numLayers;
    this._headDim = modelConfig.headDim;
    this._numKvHeads = this._calcKvHeadsPerGpu(modelConfig, config);
    this._cachePerPage = this._calcCachePerPage(modelConfig, config);

    // 初始化 free_pages_pool：每页起始位置 = page_index * pageSize
    this.freePagesPool = Array.from(
      { length: numPages }, (_, i) => i * pageSize
    );
  }

  // ===== BaseKVCachePool 抽象属性实现 =====
  get numPages(): number { return this._numPages; }
  get pageSize(): number { return this._pageSize; }
  get totalCapacity(): number { return this._numPages * this._pageSize; }
  get usedCapacity(): number { return this._usedPages * this._pageSize; }

  // ===== 额外属性 =====
  get cachePerPage(): number { return this._cachePerPage; }
  get usedPages(): number { return this._usedPages; }
  get freePages(): number { return this.freePagesPool.length; }

  // ===== BaseKVCachePool 抽象方法实现 =====
  storeKV(_k: number[], _v: number[], _outLoc: number[], _layerId: number): void {
    // 仿真中为 noop，不存储真实数据
  }

  // ===== 页分配/回收 =====

  /** 计算 GPU 上的 KV head 数（含 TP 分布） */
  private _calcKvHeadsPerGpu(
    modelConfig: ModelConfig,
    config?: SimulatorConfig,
  ): number {
    const tpSize = config?.tpSize ?? 1;
    return divEven(modelConfig.numKvHeads, tpSize, true)
      .reduce((sum, v) => sum + v, 0);
  }

  /** 计算 cache_per_page 常量（§3.3.9） */
  private _calcCachePerPage(
    modelConfig: ModelConfig,
    config?: SimulatorConfig,
  ): number {
    const dtypeSize = config?.dtypeSize ?? 2;
    return 2 *                    // key + value
           modelConfig.headDim *
           this._numKvHeads *
           this._pageSize *
           dtypeSize *
           modelConfig.numLayers;
  }

  /** 分配指定数量的页，返回 PageAllocation */
  allocatePaged(neededPages: number): PageAllocation {
    if (neededPages > this.freePagesPool.length) {
      throw new Error(
        `MockKVCachePool: allocatePaged failed, needed=${neededPages}, ` +
        `available=${this.freePagesPool.length}`
      );
    }

    const pages: number[] = [];
    const slots: number[] = [];
    for (let i = 0; i < neededPages; i++) {
      const pageStart = this.freePagesPool.pop()!;
      pages.push(pageStart);
      for (let j = 0; j < this._pageSize; j++) {
        slots.push(pageStart + j);
      }
    }
    this._usedPages += neededPages;

    return new PageAllocation(pages, slots, neededPages * this._pageSize);
  }

  /** 回收已分配的 PageAllocation */
  deallocatePageAllocation(pageAlloc: PageAllocation): void {
    for (const pageStart of pageAlloc.pages) {
      this.freePagesPool.push(pageStart);
    }
    this._usedPages -= pageAlloc.pages.length;
  }

  // ===== 延迟模型 =====

  /** 计算 decode 步骤的延迟（ticks） */
  decodeStepLatency(
    numDecodeTokens: number,
    tokenDecodeCost: number = 1,
    cudaGraphOverhead: number = 0,
  ): number {
    return numDecodeTokens * tokenDecodeCost + cudaGraphOverhead;
  }
}
