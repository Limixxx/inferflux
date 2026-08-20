# PD Disaggregation Simulator — TS Server

PD 分离推理模拟器的 TypeScript 后端，将原始单文件 `pd-disagg.html` 中的模拟引擎与前端渲染逻辑分离，采用面向对象设计，HTTP 服务与模拟服务独立运行。

支持两种部署模式：
- **PD-Disagg（PD 分离）**：Prefill 与 Decode 运行在物理分离的实例上，通过 KV 传输链路连接。
- **Agg（聚合）**：统一 Worker 实例就地完成 prefill→decode，无传输链路，模拟 SGLang make_batch 混合批调度。

## 目录结构

```
server/
├── package.json                # 项目配置
├── tsconfig.json               # TypeScript 编译配置
├── README.md                   # 本文件
├── AGENT.md                    # TS ↔ HTML 对应关系文档
├── scripts/                    # 启动脚本
│   ├── start.ps1               # Windows 启动脚本
│   └── start.sh                # Linux/macOS 启动脚本
├── https_server.py             # 原始 Python HTTPS 静态服务器（保留）
└── src/
    ├── index.ts                # 主入口：启动 SimService + HttpService
    │
    ├── shared/                 # 共享层（类型、常量、工具、国际化）
    │   ├── types.ts            # 所有接口与类型定义
    │   ├── constants.ts        # 引擎常量（TICK, RING_MAX, KVPOLL, BD_KEYS_* …）
    │   ├── presets.ts          # 模型/GPU 预设、默认参数、侧边栏定义
    │   ├── utils.ts            # 工具函数（clamp, cellSizeOf, chunkPrefillMs, fullPrefillMs …）
    │   ├── rng.ts              # 伪随机数生成器（mulberry32, 采样分布）
    │   └── i18n.ts             # 中英文词典 + t() 翻译函数
    │
    ├── sim/                    # 模拟服务（独立 HTTP API）
    │   ├── entities/
    │   │   ├── Request.ts      # 请求工厂 makeRequest
    │   │   ├── TransferLink.ts # Prefill→Decode 传输链路（pd-disagg 模式）
    │   │   ├── PrefillInstance.ts  # Prefill 实例（bootstrapQ, waitingQ, slots, link）
    │   │   ├── DecodeInstance.ts   # Decode 实例（preallocQ, transferQ, running, retract）
    │   │   └── WorkerInstance.ts   # Worker 实例（agg 模式：waitingQ, running, stepLatencyMs）
    │   ├── LoadBalancer.ts     # 负载均衡策略（least/round_robin/P2C/random）
    │   ├── MetricsCollector.ts # 指标采集、百分位、时序 sparkline 数据（模式感知）
    │   ├── SimEngine.ts        # 核心引擎（step 驱动 PD/Agg 两种生命周期）
    │   └── SimService.ts       # 模拟服务 HTTP 服务器（端口 3001）
    │
    ├── http/
    │   └── HttpService.ts      # HTTP 静态服务 + API 代理（端口 8888）
    │
    └── test/
        └── agg.test.ts         # agg 模式验收测试（16 个用例）
```

## 架构设计

### 两服务分离

| 服务 | 端口 | 职责 |
|------|------|------|
| **SimService** | 3001 | 封装 SimEngine，提供 REST API，驱动模拟循环 |
| **HttpService** | 8888 | 提供前端静态文件，代理 `/api/*` 到 SimService |

### 双部署模式

| 模式 | 参数 `mode` | 实体 | 传输链路 | 请求生命周期 |
|------|------------|------|---------|------------|
| **PD-Disagg** | `"pd-disagg"` | PrefillInstance + DecodeInstance | TransferLink | tokenize → bootstrap → P queue → chunked prefill → KV transfer → D queue → decode loop → detok |
| **Agg** | `"agg"` | WorkerInstance | 无 | tokenize → waiting → prefill (单次/分块) → decode → detok |

Agg 模式核心特征：
- **统一 running 批次**：w_prefill / w_chunked_prefill / w_decode 共处同一 `running` 数组
- **步骤延迟**：`step = max(prefill_compute, decode_step)` — 模拟 GPU 上 compute-bound prefill 与 memory-bound decode 重叠
- **首 Token 时机**：非分块 prefill 在单次 GPU 迭代后即采样首 token；分块 prefill 需全部分片完成后才采样
- **KV 预分配**：make_batch 入场时一次性预分配全部 inputLen 的 KV 物理块
- **RadixCache 复用**：沿用 cacheHitRate → cachedLen/uncachedLen 机制，prefill 计算量仅取决于 uncachedLen

### SimService API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/state` | 获取当前模拟状态（指标、快照、时序数据） |
| GET | `/render` | 获取前端渲染所需的实体级状态（模式感知序列化） |
| POST | `/command` | 控制模拟（start / pause / step / reset / speed） |
| POST | `/params` | 更新模拟参数（mode/chunkedPrefill 变更触发 reset） |
| POST | `/preset` | 应用预设场景（含 4 个 agg 预设） |
| GET | `/health` | 健康检查 |

### OOP 类关系

```
SimService  ──contains──▶  SimEngine
    │                         │
    │                    ┌────┴──── mode? ────┐
    │                    │                    │
    │              pd-disagg                 agg
    │                    │                    │
    │         ┌──────────┼──────────┐    ┌───┴───┐
    │         ▼          ▼          ▼    ▼       ▼
    │   PrefillInst  DecodeInst  Metrics  WorkerInst  Metrics
    │     ├──TransferLink  ├──running[]  Collector  ├──waitingQ
    │     ├──slots[]       └──retract()            ├──running[] (混合批次)
    │     └──waitingQ                              └──retract()
    │         │          │
    │         └──uses──▶ selectByPolicy()  (LoadBalancer)
    │
    └── HTTP API  ←──proxy──  HttpService  ──serves──▶  前端 HTML/JS
```

### TTFT Breakdown 列数

| 模式 | BD_KEYS 常量 | 列 |
|------|-------------|-----|
| pd-disagg | `BD_KEYS_DISAGG` | tokenize / bootstrap / pQueue / prefill / transfer / dQueue / detok (7列) |
| agg | `BD_KEYS_AGG` | tokenize / queue / prefill / detok (4列) |

## 快速开始

```bash
# 安装依赖
cd server
npm install

# 编译 + 启动
npm run build
npm start

# 或开发模式（ts-node 热编译）
npm run dev
```

启动后：
- 前端页面：`http://localhost:8888/pd-disagg.html`
- 模拟 API：`http://localhost:3001/state`

也可使用启动脚本：

```bash
# Windows
.\scripts\start.ps1

# Linux/macOS
./scripts/start.sh
```

## 编译检查

所有 TypeScript 文件均通过 strict 模式编译，零诊断错误：

```bash
npm run build   # 编译到 dist/
```

## 验收测试

agg 模式验收测试（16 个用例），使用 Node 内置 assert 模块：

```bash
npx ts-node src/test/agg.test.ts
```

覆盖范围：模式切换、非分块/分块 prefill、KV 约束、4 列 breakdown、gauges、RadixCache、BlockManager 预分配、混合批处理、预设加载、边界条件。

## 与原始 HTML 的关系

原始 `pd-disagg.html` 是一个 101KB 的单文件 Web 应用，包含模拟引擎 + Canvas 渲染器 + UI 逻辑。本 TS 项目提取了**模拟引擎核心**并重构为面向对象组件，同时新增了 HTTP API 层和 agg 部署模式。

详细的对应关系见 [AGENT.md](./AGENT.md)。

前端渲染器（Renderer）和 UI 逻辑（侧边栏、指标卡片、Sparkline）仍保留在 HTML 文件中，通过 HttpService 提供的静态文件服务加载，通过 `/api/*` 代理与 SimService 通信。agg 模式的前端适配（drawAgg/drawWorker、4 个 agg 预设按钮、参数联动）也在 HTML 中实现。
