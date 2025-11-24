# Test Model API Endpoints
# Make sure to start the backend first: python backend/src/main.py

Write-Host "================================" -ForegroundColor Cyan
Write-Host "🧪 Testing Model API Endpoints" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$baseUrl = "http://localhost:8000/api/v1/models"

# Test 1: List all models
Write-Host "`n✅ Test 1: List All Models" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "   Models found: $($response.count)" -ForegroundColor White
    Write-Host "   Sample: $($response.models[0].key) - $($response.models[0].name)" -ForegroundColor White
} catch {
    Write-Host "   ❌ Failed: $_" -ForegroundColor Red
}

# Test 2: Get specific model info
Write-Host "`n✅ Test 2: Get Cogito Model Info" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/cogito-671b" -Method Get
    Write-Host "   Name: $($response.name)" -ForegroundColor White
    Write-Host "   Type: $($response.type)" -ForegroundColor White
    Write-Host "   Context: $($response.context_length)" -ForegroundColor White
    Write-Host "   Tasks: $($response.supported_tasks -join ', ')" -ForegroundColor White
} catch {
    Write-Host "   ❌ Failed: $_" -ForegroundColor Red
}

# Test 3: Filter by type
Write-Host "`n✅ Test 3: Filter Chat Models" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/?model_type=chat" -Method Get
    Write-Host "   Chat models: $($response.count)" -ForegroundColor White
} catch {
    Write-Host "   ❌ Failed: $_" -ForegroundColor Red
}

# Test 4: Search models
Write-Host "`n✅ Test 4: Search for 'coding' Models" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/search/query?q=coding" -Method Get
    Write-Host "   Results: $($response.count)" -ForegroundColor White
    foreach ($model in $response.results) {
        Write-Host "      - $($model.key): $($model.name)" -ForegroundColor White
    }
} catch {
    Write-Host "   ❌ Failed: $_" -ForegroundColor Red
}

# Test 5: Chat completion (requires HF_TOKEN)
Write-Host "`n✅ Test 5: Chat Completion" -ForegroundColor Green
Write-Host "   (This test requires HF_TOKEN to be set)" -ForegroundColor Yellow

$body = @{
    messages = @(
        @{
            role = "user"
            content = "Say 'Hello from AEGIS!' in one sentence."
        }
    )
    temperature = 0.7
    max_tokens = 50
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/cogito-671b/chat" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   ✅ Response: $($response.choices[0].message.content)" -ForegroundColor White
    Write-Host "   📊 Tokens used: $($response.usage.total_tokens)" -ForegroundColor White
} catch {
    Write-Host "   ⚠️  Skipped (HF_TOKEN required): $_" -ForegroundColor Yellow
}

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "✅ API Tests Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
