#!/usr/bin/env bash
# ============================================================
#  PD-Disaggregation Simulator 服务启动脚本 (Linux/macOS)
#
#  用法:
#    ./scripts/start.sh                    # 默认端口 (HTTP 8888, SIM 3001)
#    ./scripts/start.sh --http-port 9000   # 自定义 HTTP 端口
#    ./scripts/start.sh --sim-port 3002    # 自定义 SIM 端口
#    ./scripts/start.sh --dev              # 开发模式 (ts-node)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"

# 默认参数
DEV_MODE=false
EXTRA_ARGS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      DEV_MODE=true
      shift
      ;;
    --http-port|--sim-port)
      EXTRA_ARGS="$EXTRA_ARGS $1=$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift
      ;;
  esac
done

echo ""
echo "=========================================="
echo "  PD-Disaggregation Simulator Server"
echo "=========================================="
echo ""

# 1. 检查 Node.js
if ! command -v node &>/dev/null; then
  echo "[ERROR] Node.js not found. Please install Node.js >= 16."
  exit 1
fi
NODE_VER=$(node --version)
echo "[OK] Node.js $NODE_VER"

# 2. 检查/安装依赖
if [ ! -d "$SERVER_DIR/node_modules" ]; then
  echo "[INFO] Installing dependencies..."
  cd "$SERVER_DIR"
  npm install
  echo "[OK] Dependencies installed."
else
  echo "[OK] node_modules exists."
fi

# 3. 启动
if [ "$DEV_MODE" = true ]; then
  echo "[INFO] Starting in DEV mode (ts-node)..."
  cd "$SERVER_DIR"
  npx ts-node src/index.ts $EXTRA_ARGS
else
  if [ ! -d "$SERVER_DIR/dist" ]; then
    echo "[INFO] Compiling TypeScript..."
    cd "$SERVER_DIR"
    npm run build
    echo "[OK] Compilation done."
  else
    echo "[OK] dist/ exists."
  fi
  echo "[INFO] Starting server..."
  cd "$SERVER_DIR"
  node dist/index.js $EXTRA_ARGS
fi
