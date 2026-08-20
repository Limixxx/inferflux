# PD Disaggregation Simulator — TS Server

PD 分离推理模拟器的 TypeScript 后端，将原始单文件 `pd-disagg.html` 中的模拟引擎与前端渲染逻辑分离，采用面向对象设计，HTTP 服务与模拟服务独立运行。

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
    │   ├── constants.ts        # 引擎常量（TICK, RING_MAX, KVPOLL …）
    │   ├── presets.ts          # 模型/GPU 预设、默认参数、侧边栏定义
    │   ├── utils.ts            # 工具函数（clamp, cellSizeOf, chunkPrefillMs …）
    │   ├── rng.ts              # 伪随机数生成器（mulberry32, 采样分布）
    │   └── i18n.ts             # 中英文词典 + t() 翻译函数
    │
    ├── sim/                    # 模拟服务（独立 HTTP API）
    │   ├── entities/
    │   │   ├── Request.ts      # 请求工厂 makeRequest
    │   │   ├── TransferLink.ts # Prefill→Decode 传输链路
    │   │   ├── PrefillInstance.ts  # Prefill 实例（bootstrapQ, waitingQ, slots, link）
    │   │   └── DecodeInstance.ts   # Decode 实例（preallocQ, transferQ, running, retract）
    │   ├── LoadBalancer.ts     # 负载均衡策略（least/round_robin/P2C/random）
    │   ├── MetricsCollector.ts # 指标采集、百分位、时序 sparkline 数据
    │   ├── SimEngine.ts        # 核心引擎（step 驱动整个 PD 生命周期）
    │   └── SimService.ts       # 模拟服务 HTTP 服务器（端口 3001）
    │
    └── http/
        └── HttpService.ts      # HTTP 静态服务 + API 代理（端口 8888）
```

## 架构设计

### 两服务分离

| 服务 | 端口 | 职责 |
|------|------|------|
| **SimService** | 3001 | 封装 SimEngine，提供 REST API，驱动模拟循环 |
| **HttpService** | 8888 | 提供前端静态文件，代理 `/api/*` 到 SimService |

### SimService API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/state` | 获取当前模拟状态（指标、快照、时序数据） |
| POST | `/command` | 控制模拟（start / pause / step / reset） |
| POST | `/params` | 更新模拟参数 |
| POST | `/preset` | 应用预设场景 |
| GET | `/health` | 健康检查 |

### OOP 类关系

```
SimService  ──contains──▶  SimEngine  ──contains──▶  PrefillInstance[]
    │                         │                          ├── TransferLink
    │                         │                          └── slots[] (PrefillSlot)
    │                         ├──contains──▶  DecodeInstance[]
    │                         │                   ├── running[] (SimRequest)
    │                         │                   └── retractDecode()
    │                         ├──contains──▶  MetricsCollector
    │                         └──uses──▶  selectByPolicy()  (LoadBalancer)
    │
    └── HTTP API  ←──proxy──  HttpService  ──serves──▶  前端 HTML/JS
```

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

## 与原始 HTML 的关系

原始 `pd-disagg.html` 是一个 101KB 的单文件 Web 应用，包含模拟引擎 + Canvas 渲染器 + UI 逻辑。本 TS 项目提取了**模拟引擎核心**并重构为面向对象组件，同时新增了 HTTP API 层。

详细的对应关系见 [AGENT.md](./AGENT.md)。

前端渲染器（Renderer）和 UI 逻辑（侧边栏、指标卡片、Sparkline）仍保留在 HTML 文件中，通过 HttpService 提供的静态文件服务加载，通过 `/api/*` 代理与 SimService 通信。
