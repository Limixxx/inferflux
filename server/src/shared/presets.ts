import { SimParams, ModelPreset, GpuPreset } from "./types";
import { fmtTokens } from "./utils";

export const MODEL_PRESETS: Record<string, ModelPreset> = {
  mimo_v25:    {layers:48, fullLayers:9,  kvHeads:4, headDim:192, dtypeBytes:2, hybrid:true, mla:false, activeB:15.4, qHeads:32, swaWindow:128},
  mimo_v25_pro:{layers:70, fullLayers:10, kvHeads:8, headDim:192, dtypeBytes:2, hybrid:true, mla:false, activeB:44.5, qHeads:48, swaWindow:128},
  llama8b:   {layers:32, kvHeads:8,  headDim:128, dtypeBytes:2, mla:false, activeB:8,  qHeads:32},
  llama70b:  {layers:80, kvHeads:8,  headDim:128, dtypeBytes:2, mla:false, activeB:70, qHeads:64},
  qwen32b:   {layers:64, kvHeads:8,  headDim:128, dtypeBytes:2, mla:false, activeB:32, qHeads:40},
  deepseekv3:{layers:61, kvHeads:1,  headDim:576, dtypeBytes:2, mla:true, kvLoraRank:512, qkRope:64, activeB:37, qHeads:128},
};

export const GPU_PRESETS: Record<string, GpuPreset> = {
  a100_80g:{hbm:80,  tflops:312,  bwTBs:2.0},
  h100:    {hbm:80,  tflops:990,  bwTBs:3.35},
  h200:    {hbm:141, tflops:990,  bwTBs:4.8},
  b200:    {hbm:192, tflops:2250, bwTBs:8.0},
  b300:    {hbm:288, tflops:2250, bwTBs:8.0},
  mi300x:  {hbm:192, tflops:1300, bwTBs:5.3},
};

export const DEFAULTS: SimParams = {
  qps:4, arrivalDist:"poisson", inputLenMean:2048, inputDist:"lognormal",
  outputLenMean:256, outputDist:"lognormal", cacheHitRate:0.9,
  numP:2, numD:2, lbPolicyP:"least", lbPolicyD:"least",
  tokenizeUsPerTok:1, transferOverheadMs:5, detokenizeMs:2,
  activeB:15.4, prefillTokPerSec:20000, chunkSize:8192,
  decodeMsBase:20, decodeMsPerReq:0.15, maxRunning:64, newTokenRatio:0.2,
  modelPreset:"mimo_v25", gpu:"h200",
  layers:48, fullLayers:9, kvHeads:4, headDim:192, dtypeBytes:2, hybrid:true,
  mla:false, kvLoraRank:512, qkRope:64, qHeads:32, swaWindow:128,
  kvGbP:99, kvGbD:99, bandwidthGBs:50,
};

export const PRESETS: Record<string, Partial<SimParams>> = {
  balanced: {...DEFAULTS},
  transferBound: {...DEFAULTS, qps:6, inputLenMean:8192, cacheHitRate:0, bandwidthGBs:1.5,
                  transferOverheadMs:15, modelPreset:"llama70b",
                  layers:80, fullLayers:80, kvHeads:8, headDim:128, dtypeBytes:2,
                  hybrid:false, mla:false},
  prefillBound: {...DEFAULTS, qps:12, inputLenMean:6144, cacheHitRate:0.05,
                 prefillTokPerSec:8000, numP:1, numD:2, bandwidthGBs:40},
  decodeSaturated: {...DEFAULTS, qps:10, inputLenMean:512, outputLenMean:1024,
                    maxRunning:24, numP:2, numD:1, decodeMsPerReq:0.4},
};

/** Parameter group display order */
export const GROUPS = ["workload","topology","latency","compute","kv"] as const;

/** Sidebar slider/select definitions */
export interface ParamDef {
  group: string;
  key: string;
  min?: number;
  max?: number;
  step?: number;
  log?: boolean;
  steps?: number[];
  type?: "select" | "toggle";
  options?: string[];
  i18nPrefix?: string;
  customOnly?: boolean;
  fmt?: (v: number) => string;
}

export const PARAM_DEFS: ParamDef[] = [
  {group:"workload", key:"qps", min:0.2, max:64, step:0.2, fmt:v=>v.toFixed(1)},
  {group:"workload", key:"arrivalDist", type:"select", options:["poisson","uniform"], i18nPrefix:"arrival."},
  {group:"workload", key:"inputLenMean", min:16, max:1048576, log:true, fmt:(v:number)=>fmtTokens(v)},
  {group:"workload", key:"inputDist", type:"select", options:["fixed","uniform","lognormal"], i18nPrefix:"dist."},
  {group:"workload", key:"outputLenMean", min:4, max:32768, log:true, fmt:(v:number)=>fmtTokens(v)},
  {group:"workload", key:"outputDist", type:"select", options:["fixed","uniform","lognormal"], i18nPrefix:"dist."},
  {group:"workload", key:"cacheHitRate", min:0, max:1, step:0.01, fmt:v=>(v*100).toFixed(0)+"%"},
  {group:"topology", key:"numP", min:1, max:8, step:1, fmt:v=>v.toFixed(0)},
  {group:"topology", key:"numD", min:1, max:8, step:1, fmt:v=>v.toFixed(0)},
  {group:"topology", key:"lbPolicyP", type:"select", options:["least","round_robin","power_of_two","random"], i18nPrefix:"lb."},
  {group:"topology", key:"lbPolicyD", type:"select", options:["least","round_robin","power_of_two","random"], i18nPrefix:"lb."},
  {group:"latency", key:"tokenizeUsPerTok", min:0, max:50, step:0.5, fmt:v=>v.toFixed(1)},
  {group:"latency", key:"transferOverheadMs", min:0, max:200, step:1, fmt:v=>v.toFixed(0)},
  {group:"latency", key:"detokenizeMs", min:0, max:50, step:0.5, fmt:v=>v.toFixed(1)},
  {group:"compute", key:"prefillTokPerSec", min:1000, max:100000, step:500, fmt:v=>(v/1000).toFixed(1)+"k"},
  {group:"compute", key:"chunkSize", steps:[1024,2048,4096,8192,16384,32768,65536], fmt:(v:number)=>fmtTokens(v)},
  {group:"compute", key:"decodeMsBase", min:5, max:200, step:1, fmt:v=>v.toFixed(0)},
  {group:"compute", key:"decodeMsPerReq", min:0, max:2, step:0.05, fmt:v=>v.toFixed(2)},
  {group:"compute", key:"maxRunning", min:1, max:512, step:1, fmt:v=>v.toFixed(0)},
  {group:"compute", key:"newTokenRatio", min:0.05, max:1, step:0.05, fmt:v=>v.toFixed(2)},
  {group:"kv", key:"modelPreset", type:"select", options:["mimo_v25","mimo_v25_pro","llama8b","llama70b","qwen32b","deepseekv3","custom"], i18nPrefix:"model."},
  {group:"kv", key:"layers", min:4, max:128, step:1, fmt:v=>v.toFixed(0), customOnly:true},
  {group:"kv", key:"kvHeads", min:1, max:64, step:1, fmt:v=>v.toFixed(0), customOnly:true},
  {group:"kv", key:"headDim", min:32, max:1024, step:16, fmt:v=>v.toFixed(0), customOnly:true},
  {group:"kv", key:"dtypeBytes", min:1, max:4, step:1, fmt:v=>v.toFixed(0), customOnly:true},
  {group:"kv", key:"swaWindow", steps:[0,128,256,512,1024,2048,4096], fmt:(v:number)=>v===0?"off":fmtTokens(v)},
  {group:"kv", key:"gpu", type:"select", options:["a100_80g","h100","h200","b200","b300","mi300x","custom"], i18nPrefix:"gpu."},
  {group:"kv", key:"kvGbP", min:1, max:288, step:1, fmt:v=>v.toFixed(0)},
  {group:"kv", key:"kvGbD", min:1, max:288, step:1, fmt:v=>v.toFixed(0)},
  {group:"kv", key:"bandwidthGBs", min:0.25, max:200, step:0.25, fmt:v=>v.toFixed(2)},
];
