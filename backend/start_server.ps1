# AGI Platform - Start Server Script
# Quick start for development server

Write-Host "🚀 Starting AGI Platform Development Server..." -ForegroundColor Cyan
Write-Host ""

# Change to src directory
Set-Location "E:\Projects\aegis-llm-chain\backend\src"

# Start server
Write-Host "📍 Location: " -NoNewline
Write-Host (Get-Location) -ForegroundColor Yellow
Write-Host "🐍 Python: " -NoNewline
Write-Host "venv_fresh (Python 3.11)" -ForegroundColor Yellow
Write-Host "🌐 Server: " -NoNewline
Write-Host "http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "📖 API Docs: " -NoNewline
Write-Host "http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Run server
E:\Projects\aegis-llm-chain\backend\venv_fresh\Scripts\python.exe -m uvicorn main:app --reload --port 8000
