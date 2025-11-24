# Start AEGIS Backend Server with HF_TOKEN
# This script will prompt for your token if not set

Write-Host "================================" -ForegroundColor Cyan
Write-Host "🚀 Starting AEGIS Backend Server" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if HF_TOKEN is set
if (-not $env:HF_TOKEN) {
    Write-Host "`n⚠️  HF_TOKEN not set in environment!" -ForegroundColor Yellow
    Write-Host "`nPlease enter your HuggingFace token:" -ForegroundColor Cyan
    Write-Host "(Get it from: https://huggingface.co/settings/tokens)" -ForegroundColor Gray
    $token = Read-Host "HF_TOKEN"
    
    if ($token) {
        $env:HF_TOKEN = $token
        Write-Host "`n✅ Token set for this session!" -ForegroundColor Green
    } else {
        Write-Host "`n❌ No token provided. Exiting..." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n✅ HF_TOKEN detected: $($env:HF_TOKEN.Substring(0, 10))..." -ForegroundColor Green
}

Write-Host "`n🔄 Starting server..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:8000" -ForegroundColor White
Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "   API: http://localhost:8000/api/v1/models/" -ForegroundColor White
Write-Host "`n   Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "================================`n" -ForegroundColor Cyan

# Start server with environment variable
Set-Location backend\src
& "..\..\venv\Scripts\python.exe" main.py
