import { SimParams, SimRequest, ISimEngine, KVPOLL } from "../../shared/types";
import { sampleLen, RNG } from "../../shared/rng";
import { clamp } from "../../shared/utils";

let REQ_SEQ = 0;

export function resetReqSeq(): void { REQ_SEQ = 0; }

/** Create a new simulation request (mirrors SGLang disaggregation lifecycle). */
export function makeRequest(engine: ISimEngine, arrivalT: number): SimRequest {
  const P = engine.P;
  const rng = engine.rng;
  const inputLen = sampleLen(rng, P.inputLenMean, P.inputDist);
  const outputLen = Math.max(2, sampleLen(rng, P.outputLenMean, P.outputDist));
  const hit = clamp(P.cacheHitRate + (rng() - 0.5) * 0.2, 0, 1);
  const cachedLen = Math.min(inputLen - 1, Math.round(inputLen * hit));
  return {
    id: REQ_SEQ++,
    room: Math.floor(rng() * 1e9),
    inputLen, outputLen, cachedLen,
    uncachedLen: inputLen - cachedLen,
    stage: "tokenize", kvPoll: null,
    p: null, d: null,
    readyAt: Infinity,
    dReadyAt: Infinity,
    dPrealloc: false,
    chunksTotal: 0, chunksComputed: 0, chunksQueued: 0, chunksTransferred: 0,
    bytesTotal: 0, bytesDone: 0,
    tokensOut: 0,
    retracted: false,
    stamps: { recv: arrivalT, tokenized: NaN, bootstrapDone: NaN, pQueueExit: NaN,
              prefillDone: NaN, transferDone: NaN, preallocDone: NaN,
              dQueueExit: NaN, firstToken: NaN, lastToken: NaN, detokDone: NaN },
    lastTokenT: 0,
  };
}
