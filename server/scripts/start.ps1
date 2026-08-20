<#
.SYNOPSIS
  PD-Disaggregation Simulator 服务启动脚本 (Windows PowerShell)
.DESCRIPTION
  自动检查依赖、编译 TypeScript、启动 HTTP 服务 (8888) 和模拟服务 (3001)
.PARAMETER httpPort
  HTTP 静态服务端口，默认 8888
.PARAMETER simPort
  模拟引擎 API 端口，默认 3001
.PARAMETER Dev
  使用 ts-node 开发模式（跳过编译，热加载）
.EXAMPLE
  .\scripts\start.ps1
  .\scripts\start.ps1 -httpPort 9000 -simPort 3002
  .\scripts\start.ps1 -Dev
#>

param(
  [int]$httpPort = 8888,
  [int]$simPort  = 3001,
  [switch]$Dev
)

$ErrorActionPreference = "Stop"
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$serverDir  = Split-Path -Parent $scriptDir

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  PD-Disaggregation Simulator Server" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查 Node.js
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  Write-Host "[ERROR] Node.js not found. Please install Node.js >= 16." -ForegroundColor Red
  exit 1
}
$nodeVer = (node --version)
Write-Host "[OK] Node.js $nodeVer" -ForegroundColor Green

# 2. 检查/安装依赖
if (-not (Test-Path "$serverDir\node_modules")) {
  Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
  Push-Location $serverDir
  npm install
  Pop-Location
  Write-Host "[OK] Dependencies installed." -ForegroundColor Green
} else {
  Write-Host "[OK] node_modules exists." -ForegroundColor Green
}

if ($Dev) {
  # 开发模式：ts-node 直接运行
  Write-Host "[INFO] Starting in DEV mode (ts-node)..." -ForegroundColor Yellow
  Push-Location $serverDir
  npx ts-node src/index.ts --http-port=$httpPort --sim-port=$simPort
  Pop-Location
} else {
  # 生产模式：先编译再运行
  if (-not (Test-Path "$serverDir\dist")) {
    Write-Host "[INFO] Compiling TypeScript..." -ForegroundColor Yellow
    Push-Location $serverDir
    npm run build
    Pop-Location
    Write-Host "[OK] Compilation done." -ForegroundColor Green
  } else {
    Write-Host "[OK] dist/ exists." -ForegroundColor Green
  }

  Write-Host "[INFO] Starting server..." -ForegroundColor Yellow
  Push-Location $serverDir
  node dist/index.js --http-port=$httpPort --sim-port=$simPort
  Pop-Location
}
