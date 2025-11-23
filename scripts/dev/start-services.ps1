# Start development services
# This script starts all required services for development

Write-Host "Starting AGI Platform Development Services..." -ForegroundColor Cyan
Write-Host ""

# Start Docker services
Write-Host "Starting Docker services (Database, Redis)..." -ForegroundColor Yellow
docker-compose -f infrastructure\docker\docker-compose.yml up -d db redis

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker services started" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to start Docker services" -ForegroundColor Red
    exit 1
}

# Wait for services to be ready
Write-Host ""
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if database is ready
Write-Host "Checking database connection..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$dbReady = $false

while (-not $dbReady -and $attempt -lt $maxAttempts) {
    try {
        $result = docker exec $(docker ps -qf "name=db") pg_isready -U postgres 2>&1
        if ($result -match "accepting connections") {
            $dbReady = $true
            Write-Host "✓ Database is ready" -ForegroundColor Green
        } else {
            $attempt++
            Start-Sleep -Seconds 1
        }
    } catch {
        $attempt++
        Start-Sleep -Seconds 1
    }
}

if (-not $dbReady) {
    Write-Host "✗ Database failed to start" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Services Started Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available services:" -ForegroundColor Yellow
Write-Host "- PostgreSQL: localhost:5432" -ForegroundColor White
Write-Host "- Redis: localhost:6379" -ForegroundColor White
Write-Host ""
Write-Host "To start the backend:" -ForegroundColor Yellow
Write-Host "  cd backend" -ForegroundColor White
Write-Host "  uvicorn src.main:app --reload" -ForegroundColor White
Write-Host ""
Write-Host "To start the frontend:" -ForegroundColor Yellow
Write-Host "  cd frontend" -ForegroundColor White
Write-Host "  npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "To stop services:" -ForegroundColor Yellow
Write-Host "  docker-compose -f infrastructure\docker\docker-compose.yml down" -ForegroundColor White
