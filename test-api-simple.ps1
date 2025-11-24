#!/usr/bin/env pwsh
# Simple API Test Script

Write-Host "================================" -ForegroundColor Cyan
Write-Host "🧪 Testing AEGIS API" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:8000"

# Test 1: Root endpoint
Write-Host "1. Testing Root Endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "   ✅ Response:" -ForegroundColor Green
    Write-Host "   $($response | ConvertTo-Json)" -ForegroundColor White
} catch {
    Write-Host "   ❌ Error: $_" -ForegroundColor Red
}

Write-Host ""

# Test 2: List all models
Write-Host "2. Listing Available Models..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/models/" -Method Get
    Write-Host "   ✅ Found $($response.count) models:" -ForegroundColor Green
    foreach ($model in $response.models | Select-Object -First 5) {
        Write-Host "      • $($model.key): $($model.name)" -ForegroundColor White
    }
    if ($response.count -gt 5) {
        Write-Host "      ... and $($response.count - 5) more" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ❌ Error: $_" -ForegroundColor Red
}

Write-Host ""

# Test 3: Get specific model info
Write-Host "3. Getting Model Details (cogito-671b)..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/models/cogito-671b" -Method Get
    Write-Host "   ✅ Model Info:" -ForegroundColor Green
    Write-Host "      Name: $($response.name)" -ForegroundColor White
    Write-Host "      Type: $($response.type)" -ForegroundColor White
    Write-Host "      Context: $($response.context_length) tokens" -ForegroundColor White
    Write-Host "      Status: $($response.status)" -ForegroundColor White
} catch {
    Write-Host "   ❌ Error: $_" -ForegroundColor Red
}

Write-Host ""

# Test 4: Try chat completion
Write-Host "4. Testing Chat Completion..." -ForegroundColor Yellow
Write-Host "   Prompt: 'Say hello in 5 words'" -ForegroundColor Gray

$chatBody = @{
    messages = @(
        @{
            role = "user"
            content = "Say hello in exactly 5 words"
        }
    )
    temperature = 0.7
    max_tokens = 20
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/models/cogito-671b/chat" -Method Post -Body $chatBody -ContentType "application/json"
    Write-Host "   ✅ Response:" -ForegroundColor Green
    Write-Host "      $($response.choices[0].message.content)" -ForegroundColor White
    Write-Host "      Tokens used: $($response.usage.total_tokens)" -ForegroundColor Gray
} catch {
    Write-Host "   ⚠️  Chat completion requires HF_TOKEN and may have rate limits" -ForegroundColor Yellow
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "✅ API Test Complete!" -ForegroundColor Green
Write-Host "" -ForegroundColor Cyan
Write-Host "📖 View full API docs at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
