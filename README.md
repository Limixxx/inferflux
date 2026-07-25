# InferFlux — 推理系统可视化

A collection of interactive, single-file web animations for **LLM inference systems**.
每个页面独立成页、零依赖，拖动参数即可实时观察系统行为的变化。

在线访问 · Live: **https://Abatom.github.io/inferflux/**

## 页面 · Pages

- **`index.html`** — 首页，各页入口 · homepage
- **`pd-disagg.html`** — PD 分离模拟器 · Prefill/Decode disaggregation simulator
  忠实还原 SGLang 的 PD 分离请求生命周期，实时联动 TTFT / TPOT / 吞吐 / 排队 / KV 利用率。
- **`calc-input.html`** — 输入吞吐计算器 · prefill/input throughput (MFU / roofline)
- **`calc-output.html`** — 输出吞吐计算器 · decode/output throughput (MBU / TPOT)

## 用法 · Usage

直接用浏览器打开 `index.html`，无需安装、构建或服务端。
Open `index.html` in a browser. No install, no build, no server.

每个页面都是自包含的静态 HTML 文件（CSS + DOM + JS），可单独分发。
Every page is a self-contained static HTML file — copy one file and it just works.

### 本地启动 · Run locally

若需在本地通过 HTTPS 访问（部分页面需要安全上下文），先生成自签名证书（仓库未包含），再运行内置的 HTTPS server：
For local access over HTTPS (some pages require a secure context), first generate a self-signed cert (not included in the repo), then start the bundled HTTPS server:

```bash
# 生成自签名证书 · generate a self-signed cert
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout server/key.pem -out server/cert.pem -subj "/CN=localhost"

# 启动 · start
python3 server/https_server.py
```

然后浏览器打开 · then open **https://localhost:8888/**（自签名证书，首次访问需手动信任 · self-signed cert, accept the warning on first visit）。

## 协议 · License

本项目采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)（署名-非商业）协议。
欢迎自由使用、修改、分享，但**须署名且不得用于商业用途**；商业使用请事先联系作者授权。

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Free to use, modify, and share **with attribution and for non-commercial purposes only**.
For commercial use, please obtain prior permission from the author. See [`LICENSE`](LICENSE).
