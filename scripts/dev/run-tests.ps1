# Run tests for the AGI Platform
# This script runs all tests with optional coverage

param(
    [switch]$Coverage,
    [switch]$Verbose,
    [string]$TestPath = "backend\tests"
)

Write-Host "Running AGI Platform Tests..." -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Build pytest command
$pytestArgs = @($TestPath)

if ($Verbose) {
    $pytestArgs += "-v"
}

if ($Coverage) {
    $pytestArgs += "--cov=backend\src"
    $pytestArgs += "--cov-report=html"
    $pytestArgs += "--cov-report=term"
}

# Run pytest
Write-Host "Running tests..." -ForegroundColor Yellow
Write-Host "Command: pytest $($pytestArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

pytest @pytestArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ All tests passed!" -ForegroundColor Green
    
    if ($Coverage) {
        Write-Host ""
        Write-Host "Coverage report generated at: htmlcov\index.html" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "✗ Tests failed" -ForegroundColor Red
    exit 1
}

# Run frontend tests if requested
$runFrontendTests = Read-Host "Run frontend tests? (y/n)"
if ($runFrontendTests -eq "y") {
    Write-Host ""
    Write-Host "Running frontend tests..." -ForegroundColor Yellow
    Push-Location frontend
    npm run test
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Frontend tests passed!" -ForegroundColor Green
    } else {
        Write-Host "✗ Frontend tests failed" -ForegroundColor Red
    }
    Pop-Location
}

Write-Host ""
Write-Host "Testing complete!" -ForegroundColor Cyan
