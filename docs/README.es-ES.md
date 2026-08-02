# InferFlux — Visualización de sistemas de inferencia

[English](../README.md) · [简体中文](README.zh-CN.md) · **Español**

Una colección de animaciones web interactivas de archivo único para **sistemas de inferencia de LLM**.
Cada página es independiente y sin dependencias; arrastra los parámetros para observar en tiempo real los cambios en el comportamiento del sistema.

Acceso en línea · Live: **https://Abatom.github.io/inferflux/**

## Páginas

- **`index.html`** — Página de inicio, acceso a las demás secciones
- **`pd-disagg.html`** — Simulador de separación PD (Prefill/Decode)
  Replica fielmente el ciclo de vida de las solicitudes de separación PD de SGLang, mostrando en tiempo real TTFT / TPOT / rendimiento / colas / uso de KV.
- **`calc-input.html`** — Calculadora de rendimiento de entrada (prefill, MFU / roofline)
- **`calc-output.html`** — Calculadora de rendimiento de salida (decode, MBU / TPOT)

## Uso

Abre `index.html` en un navegador. Sin instalación, sin compilación, sin servidor.

Cada página es un archivo HTML estático autocontenido (CSS + DOM + JS): copia un archivo y funcionará directamente.

### Ejecución local

Para acceder localmente a través de HTTPS (algunas páginas requieren un contexto seguro), primero genera un certificado autofirmado (no incluido en el repositorio) y luego inicia el servidor HTTPS incluido:

```bash
# Generar certificado autofirmado
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout server/key.pem -out server/cert.pem -subj "/CN=localhost"

# Iniciar
python3 server/https_server.py
```

Luego abre **https://localhost:8888/** en tu navegador (certificado autofirmado; acepta la advertencia en la primera visita).

## Licencia

Licenciado bajo [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Atribución-No comercial).
Libre para usar, modificar y compartir **con atribución y exclusivamente para fines no comerciales**.
Para uso comercial, solicita permiso previo al autor. Consulta [`LICENSE`](../LICENSE).
