import { SimParams, SimRequest } from "../../shared/types";
import { TransferItem } from "../../shared/types";
import { cellSizeOf } from "../../shared/utils";
import { TransferLink } from "./TransferLink";

/** Slot for an in-progress chunked prefill. */
export interface PrefillSlot {
  req: SimRequest;
  busyUntil: number;
}

/** A single Prefill server instance with its own queues and KV pool. */
export class PrefillInstance {
  id: number;
  bootstrapQ: SimRequest[] = [];
  waitingQ: SimRequest[] = [];
  slots: PrefillSlot[] = [];
  inflight: SimRequest[] = [];
  kvUsed = 0;
  draining = false;
  link: TransferLink;

  constructor(id: number) {
    this.id = id;
    this.link = new TransferLink();
  }

  maxTokens(P: SimParams): number {
    return Math.max(1, Math.floor(P.kvGbP * 2**30 / cellSizeOf(P)));
  }

  pendingLoad(): number {
    let s = 0;
    for (const r of this.bootstrapQ) s += r.uncachedLen;
    for (const r of this.waitingQ) s += r.uncachedLen;
    for (const slot of this.slots) s += slot.req.uncachedLen;
    return s;
  }

  isEmpty(): boolean {
    return this.slots.length === 0 && !this.link.current && this.bootstrapQ.length === 0 &&
           this.waitingQ.length === 0 && this.inflight.length === 0 && this.link.queue.length === 0;
  }
}
