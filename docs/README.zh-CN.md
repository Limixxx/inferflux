# InferFlux — 推理系统可视化

[English](../README.md) · **简体中文** · [Español](README.es-ES.md)

一组面向 **LLM 推理系统** 的交互式单文件网页动画。
每个页面独立成页、零依赖，拖动参数即可实时观察系统行为的变化。

在线访问 · Live: **https://Abatom.github.io/inferflux/**

## 页面

- **`index.html`** — 首页，各页入口
- **`pd-disagg.html`** — PD 分离模拟器（Prefill/Decode disaggregation）
  忠实还原 SGLang 的 PD 分离请求生命周期，实时联动 TTFT / TPOT / 吞吐 / 排队 / KV 利用率。
- **`calc-input.html`** — 输入吞吐计算器（prefill，MFU / roofline）
- **`calc-output.html`** — 输出吞吐计算器（decode，MBU / TPOT）

## 用法

直接用浏览器打开 `index.html`，无需安装、构建或服务端。

每个页面都是自包含的静态 HTML 文件（CSS + DOM + JS），可单独分发。

### 本地启动

若需在本地通过 HTTPS 访问（部分页面需要安全上下文），先生成自签名证书（仓库未包含），再运行内置的 HTTPS server：

```bash
# 生成自签名证书
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout server/key.pem -out server/cert.pem -subj "/CN=localhost"

# 启动
python3 server/https_server.py
```

然后浏览器打开 **https://localhost:8888/**（自签名证书，首次访问需手动信任）。

## 协议

本项目采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)（署名-非商业）协议。
欢迎自由使用、修改、分享，但**须署名且不得用于商业用途**；商业使用请事先联系作者授权。详见 [`LICENSE`](../LICENSE)。
