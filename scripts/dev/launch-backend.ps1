# 🚀 AGI Platform Quick Launch Script for Windows
# This launches the backend server with your existing security-first architecture

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 LAUNCHING AGI PLATFORM BACKEND!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  Please update .env with your configuration!" -ForegroundColor Red
}

Write-Host ""
Write-Host "✓ Environment ready" -ForegroundColor Green
Write-Host ""
Write-Host "Starting FastAPI Backend Server..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host "API will be available at:" -ForegroundColor White
Write-Host "  - http://localhost:8000" -ForegroundColor Cyan
Write-Host "  - http://localhost:8000/docs (Swagger UI)" -ForegroundColor Cyan
Write-Host "  - http://localhost:8000/health (Health Check)" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host ""

# Start the server
Set-Location backend
E:\Projects\aegis-llm-chain\venv\Scripts\python.exe -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
