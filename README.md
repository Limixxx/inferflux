# InferFlux — LLM Inference System Visualizations

**English** · [简体中文](docs/README.zh-CN.md) · [Español](docs/README.es-ES.md)

A collection of interactive, single-file web animations for **LLM inference systems**.
Each page is standalone and dependency-free — drag the parameters to watch system behavior change in real time.

Online · Live: **https://Abatom.github.io/inferflux/**

## Pages

- **`index.html`** — homepage, entry point to all sections
- **`pd-disagg.html`** — Prefill/Decode disaggregation simulator
  Faithfully reproduces SGLang's PD-disaggregated request lifecycle, with live TTFT / TPOT / throughput / queueing / KV utilization.
- **`calc-input.html`** — input throughput calculator (prefill, MFU / roofline)
- **`calc-output.html`** — output throughput calculator (decode, MBU / TPOT)

## Usage

Open `index.html` in a browser. No install, no build, no server.

Every page is a self-contained static HTML file (CSS + DOM + JS) — copy one file and it just works.

### Run locally

For local access over HTTPS (some pages require a secure context), first generate a self-signed cert (not included in the repo), then start the bundled HTTPS server:

```bash
# generate a self-signed cert
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout server/key.pem -out server/cert.pem -subj "/CN=localhost"

# start
python3 server/https_server.py
```

Then open **https://localhost:8888/** in your browser (self-signed cert, accept the warning on first visit).

## License

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Attribution-NonCommercial).
Free to use, modify, and share **with attribution and for non-commercial purposes only**.
For commercial use, please obtain prior permission from the author. See [`LICENSE`](LICENSE).
