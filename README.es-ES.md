

# InferFlux — Visualización de sistemas de inferencia

Una colección de animaciones web interactivas de archivo único para **sistemas de inferencia de LLM**.
Cada página es independiente y sin dependencias; arrastra los parámetros para observar en tiempo real los cambios en el comportamiento del sistema.

Acceso en línea · En vivo: **https://Abatom.github.io/inferflux/**

## Páginas

- **`index.html`** — Página de inicio, acceso a las demás secciones
- **`pd-disagg.html`** — Simulador de separación PD (Prefill/Decode)
  Replica fielmente el ciclo de vida de las solicitudes de separación PD de SGLang, mostrando en tiempo real TTFT / TPOT / rendimiento / colas / uso de KV.
- **`calc-input.html`** — Calculadora de rendimiento de entrada (MFU / roofline)
- **`calc-output.html`** — Calculadora de rendimiento de salida (MBU / TPOT)

## Uso

Abre `index.html` directamente en el navegador. No requiere instalación, compilación ni servidor.
Abre `index.html` en un navegador. Sin instalación, sin compilación, sin servidor.

Cada página es un archivo HTML estático autocontenido (CSS + DOM + JS), listo para distribuir de forma individual.
Cada página es un archivo HTML estático autocontenido: copia un archivo y funcionará directamente.

### Ejecución local

Para acceder localmente mediante HTTPS (algunas páginas requieren un contexto seguro), primero genera un certificado autofirmado (no incluido en el repositorio) y luego ejecuta el servidor HTTPS integrado:
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

Este proyecto se distribuye bajo la licencia [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Atribución-No comercial).
Se permite su uso, modificación y compartición libres, pero **debe incluir atribución y solo para fines no comerciales**; para uso comercial, contacta al autor para obtener permiso previo.

Licenciado bajo [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Libre para usar, modificar y compartir **con atribución y exclusivamente para fines no comerciales**.
Para uso comercial, solicita permiso previo al autor. Consulta [`LICENSE`](LICENSE).
