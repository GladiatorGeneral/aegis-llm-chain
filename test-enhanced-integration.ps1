#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test Optima + LLM-FE Enhanced Integration

.DESCRIPTION
    Tests all new enhanced API endpoints and cognitive engine integration
    
.EXAMPLE
    .\test-enhanced-integration.ps1
#>

$ErrorActionPreference = "Continue"
$BASE_URL = "http://localhost:8000"
$API_TOKEN = $env:HUGGINGFACE_TOKEN

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "🧪 ENHANCED INTEGRATION TEST SUITE" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Test 1: Enhanced Health Check
Write-Host "`n[Test 1] Enhanced Health Check..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/health/enhanced" -Method Get
    if ($response.enhanced_engines) {
        Write-Host "✅ Enhanced health check successful" -ForegroundColor Green
        Write-Host "   Optima Status: $($response.enhanced_engines.optima.status)" -ForegroundColor Gray
        Write-Host "   LLM-FE Status: $($response.enhanced_engines.llm_fe.status)" -ForegroundColor Gray
    } else {
        Write-Host "❌ Enhanced engines not initialized" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Optima Capabilities
Write-Host "`n[Test 2] Optima Capabilities..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/optima/capabilities" -Method Get
    if ($response.success) {
        Write-Host "✅ Optima capabilities retrieved" -ForegroundColor Green
        Write-Host "   Reasoning Depths: $($response.data.reasoning_depths -join ', ')" -ForegroundColor Gray
        Write-Host "   Reasoning Modes: $($response.data.reasoning_modes -join ', ')" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Optima not available: $($response.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Capabilities check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: LLM-FE Capabilities
Write-Host "`n[Test 3] LLM-FE Capabilities..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/llm-fe/capabilities" -Method Get
    if ($response.success) {
        Write-Host "✅ LLM-FE capabilities retrieved" -ForegroundColor Green
        Write-Host "   Routing Strategies: $($response.data.routing_strategies -join ', ')" -ForegroundColor Gray
        Write-Host "   Available Models: $($response.data.available_models.Count)" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  LLM-FE not available: $($response.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Capabilities check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Optima Reasoning (Answer-First Mode)
Write-Host "`n[Test 4] Optima Chain-of-Thought Reasoning..." -ForegroundColor Yellow
try {
    $body = @{
        prompt = "What is 2+2 and why?"
        reasoning_mode = "answer_first"
        depth = "quick"
        parameters = @{
            max_length = 200
        }
    } | ConvertTo-Json

    $headers = @{
        "Content-Type" = "application/json"
        "Authorization" = "Bearer test-token"
    }

    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/optima/reason" `
        -Method Post `
        -Body $body `
        -Headers $headers

    if ($response.success) {
        Write-Host "✅ Optima reasoning completed" -ForegroundColor Green
        Write-Host "   Answer: $($response.data.answer)" -ForegroundColor Gray
        Write-Host "   Confidence: $($response.data.confidence)" -ForegroundColor Gray
        Write-Host "   Steps: $($response.data.reasoning_steps.Count)" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Reasoning failed: $($response.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Reasoning test failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 5: LLM-FE Intelligent Routing
Write-Host "`n[Test 5] LLM-FE Intelligent Routing..." -ForegroundColor Yellow
try {
    $body = @{
        content = "Generate a business report about Q4 earnings"
        task_type = "business_report"
        strategy = "intelligent"
        parameters = @{}
    } | ConvertTo-Json

    $headers = @{
        "Content-Type" = "application/json"
        "Authorization" = "Bearer test-token"
    }

    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/llm-fe/route" `
        -Method Post `
        -Body $body `
        -Headers $headers

    if ($response.success) {
        Write-Host "✅ LLM-FE routing completed" -ForegroundColor Green
        Write-Host "   Selected Engine: $($response.data.engine)" -ForegroundColor Gray
        Write-Host "   Selected Model: $($response.data.model)" -ForegroundColor Gray
        Write-Host "   Confidence: $($response.data.confidence)" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Routing failed: $($response.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Routing test failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 6: Optima Metrics
Write-Host "`n[Test 6] Optima Performance Metrics..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/optima/metrics" -Method Get
    if ($response.success) {
        Write-Host "✅ Optima metrics retrieved" -ForegroundColor Green
        Write-Host "   Requests Processed: $($response.data.total_requests)" -ForegroundColor Gray
        Write-Host "   Avg Reasoning Steps: $($response.data.avg_reasoning_steps)" -ForegroundColor Gray
        Write-Host "   Avg Confidence: $($response.data.avg_confidence)" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Metrics not available: $($response.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Metrics check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 7: LLM-FE Metrics
Write-Host "`n[Test 7] LLM-FE Routing Metrics..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/llm-fe/metrics" -Method Get
    if ($response.success) {
        Write-Host "✅ LLM-FE metrics retrieved" -ForegroundColor Green
        Write-Host "   Total Routes: $($response.data.total_routes)" -ForegroundColor Gray
        Write-Host "   Cache Hit Rate: $($response.data.cache_hit_rate)" -ForegroundColor Gray
        Write-Host "   Avg Route Time: $($response.data.avg_route_time)" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Metrics not available: $($response.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Metrics check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 8: Clear LLM-FE Cache
Write-Host "`n[Test 8] Clear LLM-FE Routing Cache..." -ForegroundColor Yellow
try {
    $headers = @{
        "Authorization" = "Bearer test-token"
    }

    $response = Invoke-RestMethod -Uri "$BASE_URL/api/v1/llm-fe/cache" `
        -Method Delete `
        -Headers $headers

    if ($response.success) {
        Write-Host "✅ Cache cleared successfully" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Cache clear failed: $($response.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Cache clear failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "✅ INTEGRATION TEST COMPLETE" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "`nNEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Start the server: .\start-server.ps1" -ForegroundColor Gray
Write-Host "2. Run this test: .\test-enhanced-integration.ps1" -ForegroundColor Gray
Write-Host "3. Test cognitive engine with use_optima=true and use_llm_fe=true" -ForegroundColor Gray
Write-Host ""
