import { TransferItem } from "../../shared/types";

/** One egress link per Prefill instance. */
export class TransferLink {
  queue: TransferItem[] = [];
  current: TransferItem | null = null;
  startAt = 0;
  doneAt = 0;

  get depth(): number {
    return this.queue.length + (this.current ? 1 : 0);
  }
}
