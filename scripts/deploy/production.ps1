# AEGIS LLM Chain - Production Deployment Script (PowerShell)
# Run with: .\production.ps1

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { param($message) Write-Host "[OK] $message" -ForegroundColor Green }
function Write-Error-Custom { param($message) Write-Host "[ERROR] $message" -ForegroundColor Red }
function Write-Warning-Custom { param($message) Write-Host "[WARNING] $message" -ForegroundColor Yellow }
function Write-Info { param($message) Write-Host "[INFO] $message" -ForegroundColor Cyan }

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$DockerDir = Join-Path $ProjectRoot "infrastructure\docker"

Write-Host ""
Write-Host "========================================" -ForegroundColor Blue
Write-Host "  AEGIS LLM Chain Deployment Script" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Check if Docker is installed
function Test-Docker {
    Write-Info "Checking Docker installation..."
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        # Try docker compose (v2)
        try {
            docker compose version | Out-Null
        } catch {
            Write-Error-Custom "Docker Compose is not available."
            exit 1
        }
    }
    
    Write-Success "Docker and Docker Compose are installed"
}

# Check if .env file exists
function Test-EnvFile {
    Write-Info "Checking environment configuration..."
    
    $envFile = Join-Path $ProjectRoot ".env"
    $envExample = Join-Path $ProjectRoot ".env.example"
    
    if (-not (Test-Path $envFile)) {
        Write-Warning-Custom ".env file not found"
        Write-Info "Creating .env from .env.example..."
        Copy-Item $envExample $envFile
        Write-Warning-Custom "Please edit .env file with your actual configuration"
        Write-Warning-Custom "Especially set your HF_TOKEN!"
        exit 1
    }
    
    Write-Success "Environment file exists"
}

# Check if HF_TOKEN is set
function Test-HFToken {
    Write-Info "Checking HuggingFace token..."
    
    $envFile = Join-Path $ProjectRoot ".env"
    
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile
        $tokenLine = $envContent | Where-Object { $_ -match "^HF_TOKEN=" }
        
        if (-not $tokenLine -or $tokenLine -match "your_huggingface_token_here") {
            Write-Error-Custom "HF_TOKEN not set or using default value"
            Write-Warning-Custom "Please set your HuggingFace token in .env file"
            Write-Info "Get your token from: https://huggingface.co/settings/tokens"
            exit 1
        }
    }
    
    Write-Success "HF_TOKEN is configured"
}

# Pull latest code
function Update-Code {
    Write-Info "Checking for updates..."
    
    if (Test-Path (Join-Path $ProjectRoot ".git")) {
        Push-Location $ProjectRoot
        
        try {
            git fetch
            $local = git rev-parse "@"
            $remote = git rev-parse "@{u}"
            
            if ($local -ne $remote) {
                Write-Warning-Custom "Updates available"
                $response = Read-Host "Pull latest changes? (y/N)"
                
                if ($response -eq "y" -or $response -eq "Y") {
                    git pull
                    Write-Success "Code updated"
                }
            } else {
                Write-Success "Code is up to date"
            }
        } finally {
            Pop-Location
        }
    }
}

# Build Docker images
function Build-Images {
    Write-Info "Building Docker images..."
    
    Push-Location $DockerDir
    
    try {
        docker compose build --parallel
        Write-Success "Docker images built successfully"
    } catch {
        Write-Error-Custom "Failed to build Docker images"
        Write-Error-Custom $_.Exception.Message
        exit 1
    } finally {
        Pop-Location
    }
}

# Start services
function Start-Services {
    Write-Info "Starting services..."
    
    Push-Location $DockerDir
    
    try {
        docker compose up -d
        Write-Success "Services started successfully"
    } catch {
        Write-Error-Custom "Failed to start services"
        Write-Error-Custom $_.Exception.Message
        exit 1
    } finally {
        Pop-Location
    }
}

# Wait for services to be healthy
function Wait-ForHealth {
    Write-Info "Waiting for services to be healthy..."
    
    $maxAttempts = 30
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host ""
                Write-Success "Backend is healthy"
                return $true
            }
        } catch {
            # Continue waiting
        }
        
        $attempt++
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
    
    Write-Host ""
    Write-Error-Custom "Backend health check failed after $maxAttempts attempts"
    Write-Info "Check logs with: docker compose logs backend"
    return $false
}

# Show service status
function Show-Status {
    Write-Info "Service Status:"
    
    Push-Location $DockerDir
    docker compose ps
    Pop-Location
    
    Write-Host ""
}

# Show URLs
function Show-URLs {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "     Deployment Successful!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available Services:" -ForegroundColor Cyan
    Write-Host "  Backend API:       http://localhost:8000" -ForegroundColor White
    Write-Host "  API Documentation: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  Frontend:          http://localhost:3000" -ForegroundColor White
    Write-Host "  PostgreSQL:        localhost:5432" -ForegroundColor White
    Write-Host "  Redis:             localhost:6379" -ForegroundColor White
    Write-Host "  pgAdmin:           http://localhost:5050" -ForegroundColor White
    Write-Host ""
    Write-Host "Useful Commands:" -ForegroundColor Cyan
    Write-Host "  View logs:         docker compose logs -f" -ForegroundColor White
    Write-Host "  Stop services:     docker compose down" -ForegroundColor White
    Write-Host "  Restart backend:   docker compose restart backend" -ForegroundColor White
    Write-Host ""
}

# Main deployment flow
function Main {
    Write-Host "Starting deployment..." -ForegroundColor White
    Write-Host ""
    
    Test-Docker
    Test-EnvFile
    Test-HFToken
    Update-Code
    
    Write-Host ""
    $response = Read-Host "Continue with deployment? (Y/n)"
    
    if ($response -eq "n" -or $response -eq "N") {
        Write-Warning-Custom "Deployment cancelled"
        exit 0
    }
    
    Build-Images
    Start-Services
    
    if (Wait-ForHealth) {
        Show-Status
        Show-URLs
    } else {
        Write-Error-Custom "Deployment completed but health check failed"
        Write-Info "Check logs for more information"
        exit 1
    }
}

# Run main function
Main
