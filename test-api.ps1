# Quick API test script
Write-Host "Testing AGI Platform API..." -ForegroundColor Cyan
Write-Host ""

try {
    # Test health endpoint
    Write-Host "Testing /health endpoint..." -ForegroundColor Yellow
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
    Write-Host "✓ Status: $($health.status)" -ForegroundColor Green
    Write-Host ""
    
    # Test root endpoint
    Write-Host "Testing / endpoint..." -ForegroundColor Yellow
    $root = Invoke-RestMethod -Uri "http://127.0.0.1:8000/" -Method Get
    Write-Host "✓ Message: $($root.message)" -ForegroundColor Green
    Write-Host "✓ Version: $($root.version)" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "✓ All tests passed! Server is running correctly." -ForegroundColor Green
    Write-Host ""
    Write-Host "Available endpoints:" -ForegroundColor Cyan
    Write-Host "  - http://127.0.0.1:8000 (Root)" -ForegroundColor White
    Write-Host "  - http://127.0.0.1:8000/health (Health check)" -ForegroundColor White
    Write-Host "  - http://127.0.0.1:8000/docs (Swagger UI)" -ForegroundColor White
    Write-Host "  - http://127.0.0.1:8000/api/v1/cognitive/process (Cognitive engine)" -ForegroundColor White
    Write-Host "  - http://127.0.0.1:8000/api/v1/models/list (Models)" -ForegroundColor White
    
} catch {
    Write-Host "✗ Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure the server is running:" -ForegroundColor Yellow
    Write-Host "  cd backend\src" -ForegroundColor White
    Write-Host "  ..\..\venv\Scripts\python.exe -m uvicorn main:app --reload" -ForegroundColor White
}
