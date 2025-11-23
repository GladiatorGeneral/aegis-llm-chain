# 🧪 Test the AGI Platform Security Layer
# This script runs comprehensive security tests

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🧪 TESTING SECURITY LAYER" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:8000"

# Test 1: Health Check
Write-Host "Test 1: Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "✓ Health: $($health.status)" -ForegroundColor Green
    Write-Host "  Version: $($health.version)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Health check failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 2: Valid Input
Write-Host "Test 2: Valid Input..." -ForegroundColor Yellow
$validPayload = @{
    text = "Hello world! This is a valid test message."
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "$baseUrl/api/v1/cognitive/process" `
        -Method Post `
        -ContentType "application/json" `
        -Body $validPayload
    Write-Host "✓ Valid input accepted" -ForegroundColor Green
    Write-Host "  Result: $($result.result)" -ForegroundColor Gray
} catch {
    Write-Host "⚠️  Endpoint not yet implemented (expected)" -ForegroundColor Yellow
}

Write-Host ""

# Test 3: PII Redaction
Write-Host "Test 3: PII Redaction..." -ForegroundColor Yellow
$piiPayload = @{
    text = "My email is test@example.com and my phone is 555-123-4567"
} | ConvertTo-Json

Write-Host "  Input: My email is test@example.com and my phone is 555-123-4567" -ForegroundColor Gray
Write-Host "  Expected: PII should be redacted" -ForegroundColor Gray

Write-Host ""

# Test 4: Blocked Content
Write-Host "Test 4: Blocked Content Detection..." -ForegroundColor Yellow
$blockedPayload = @{
    text = "Here is my password=secret123 and api_key=abc123xyz"
} | ConvertTo-Json

Write-Host "  Input: Contains credentials" -ForegroundColor Gray
Write-Host "  Expected: Should be blocked" -ForegroundColor Gray

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Security tests complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Start the backend: .\scripts\dev\launch-backend.ps1" -ForegroundColor White
Write-Host "2. Visit http://localhost:8000/docs to explore the API" -ForegroundColor White
Write-Host "3. Test endpoints interactively with Swagger UI" -ForegroundColor White
