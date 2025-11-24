#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup Python environment for AGI Platform with proper version checking

.DESCRIPTION
    This script checks Python version compatibility, creates/recreates virtual environment,
    and installs all dependencies for the AGI Platform project.

.EXAMPLE
    .\setup-python-env.ps1
    .\setup-python-env.ps1 -Force    # Force recreate venv even if it exists
#>

param(
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🐍 AGI Platform - Python Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Function to get Python version
function Get-PythonVersion {
    param($pythonCmd)
    
    try {
        # Handle py launcher with version argument
        if ($pythonCmd -like "py *") {
            $args = $pythonCmd.Split(" ")
            $versionOutput = & $args[0] $args[1] --version 2>&1
        }
        else {
            $versionOutput = & $pythonCmd --version 2>&1
        }
        
        if ($versionOutput -match "Python (\d+)\.(\d+)\.(\d+)") {
            return @{
                Major = [int]$Matches[1]
                Minor = [int]$Matches[2]
                Patch = [int]$Matches[3]
                Full = "$($Matches[1]).$($Matches[2]).$($Matches[3])"
                Command = $pythonCmd
            }
        }
    }
    catch {
        return $null
    }
    return $null
}

# Check for Python installations
Write-Host "🔍 Checking Python installations..." -ForegroundColor Yellow
Write-Host ""

$pythonCommands = @("py -3.11", "python3.11", "python", "py")
$foundPython = $null
$pythonCmd = $null

foreach ($cmd in $pythonCommands) {
    Write-Host "  Trying: $cmd" -ForegroundColor Gray
    $version = Get-PythonVersion $cmd
    
    if ($version) {
        Write-Host "    Found: Python $($version.Full)" -ForegroundColor Gray
        
        # Check if it's Python 3.11
        if ($version.Major -eq 3 -and $version.Minor -eq 11) {
            Write-Host "    ✅ Python 3.11 found!" -ForegroundColor Green
            $foundPython = $version
            $pythonCmd = $cmd
            break
        }
        elseif ($version.Major -eq 3 -and $version.Minor -in @(9, 10)) {
            Write-Host "    ⚠️  Python $($version.Major).$($version.Minor) works but 3.11 recommended" -ForegroundColor Yellow
            if (-not $foundPython) {
                $foundPython = $version
                $pythonCmd = $cmd
            }
        }
        elseif ($version.Major -eq 3 -and $version.Minor -ge 12) {
            Write-Host "    ❌ Python $($version.Major).$($version.Minor) is too new (packages incompatible)" -ForegroundColor Red
        }
    }
}

Write-Host ""

if (-not $foundPython) {
    Write-Host "❌ ERROR: No compatible Python installation found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Required: Python 3.11.x (Recommended)" -ForegroundColor Yellow
    Write-Host "Supported: Python 3.9.x, 3.10.x, 3.11.x" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📥 Download Python 3.11.9 from:" -ForegroundColor Cyan
    Write-Host "   https://www.python.org/downloads/release/python-3119/" -ForegroundColor White
    Write-Host ""
    Write-Host "Installation tips:" -ForegroundColor Yellow
    Write-Host "  1. Download 'Windows installer (64-bit)'" -ForegroundColor White
    Write-Host "  2. Check 'Add Python to PATH' during installation" -ForegroundColor White
    Write-Host "  3. Choose 'Install for all users' (optional)" -ForegroundColor White
    Write-Host "  4. Restart your terminal after installation" -ForegroundColor White
    Write-Host ""
    exit 1
}

if ($foundPython.Minor -ne 11) {
    Write-Host "⚠️  WARNING: Using Python $($foundPython.Full)" -ForegroundColor Yellow
    Write-Host "   Recommended: Python 3.11.x for best compatibility" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        Write-Host "Setup cancelled." -ForegroundColor Yellow
        exit 0
    }
}
else {
    Write-Host "✅ Using Python $($foundPython.Full)" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "📦 Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
$venvPath = ".\venv"
$venvExists = Test-Path $venvPath

if ($venvExists -and -not $Force) {
    Write-Host "⚠️  Virtual environment already exists at: $venvPath" -ForegroundColor Yellow
    Write-Host ""
    $recreate = Read-Host "Recreate virtual environment? This will delete the existing one. (y/N)"
    
    if ($recreate -ne 'y' -and $recreate -ne 'Y') {
        Write-Host ""
        Write-Host "ℹ️  Using existing virtual environment" -ForegroundColor Cyan
        Write-Host "   To force recreate, run: .\setup-python-env.ps1 -Force" -ForegroundColor Gray
    }
    else {
        $Force = $true
    }
}

if ($Force -and $venvExists) {
    Write-Host "🗑️  Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
    Write-Host "   ✅ Removed" -ForegroundColor Green
}

if (-not (Test-Path $venvPath)) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    
    try {
        # Handle py launcher with version argument
        if ($pythonCmd -like "py *") {
            $args = $pythonCmd.Split(" ")
            & $args[0] $args[1] -m venv venv
        }
        else {
            & $pythonCmd -m venv venv
        }
        Write-Host "   ✅ Virtual environment created" -ForegroundColor Green
    }
    catch {
        Write-Host "   ❌ Failed to create virtual environment" -ForegroundColor Red
        Write-Host "   Error: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "📥 Installing Dependencies" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv and install packages
$activateScript = ".\venv\Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "❌ ERROR: Virtual environment activation script not found" -ForegroundColor Red
    exit 1
}

Write-Host "🔄 Upgrading pip..." -ForegroundColor Yellow
& .\venv\Scripts\python.exe -m pip install --upgrade pip --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ pip upgraded" -ForegroundColor Green
}
else {
    Write-Host "   ⚠️  pip upgrade failed (continuing anyway)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📦 Installing base requirements..." -ForegroundColor Yellow
Write-Host "   This may take several minutes..." -ForegroundColor Gray
Write-Host ""

$requirementsFiles = @(
    "backend\requirements\base.txt"
)

foreach ($reqFile in $requirementsFiles) {
    if (Test-Path $reqFile) {
        Write-Host "   Installing from $reqFile..." -ForegroundColor Cyan
        & .\venv\Scripts\pip.exe install -r $reqFile
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $reqFile installed successfully" -ForegroundColor Green
            Write-Host ""
        }
        else {
            Write-Host "   ❌ Failed to install $reqFile" -ForegroundColor Red
            Write-Host ""
            Write-Host "Common issues:" -ForegroundColor Yellow
            Write-Host "  1. Python version incompatibility (need 3.11.x)" -ForegroundColor White
            Write-Host "  2. Missing Visual C++ Build Tools (for Windows)" -ForegroundColor White
            Write-Host "  3. Network/connectivity issues" -ForegroundColor White
            Write-Host ""
            exit 1
        }
    }
    else {
        Write-Host "   ⚠️  $reqFile not found, skipping..." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Get installed packages count
$packageCount = (& .\venv\Scripts\pip.exe list --format=freeze | Measure-Object).Count

Write-Host "📊 Environment Summary:" -ForegroundColor Cyan
Write-Host "   Python Version: $($foundPython.Full)" -ForegroundColor White
Write-Host "   Virtual Environment: $venvPath" -ForegroundColor White
Write-Host "   Packages Installed: $packageCount" -ForegroundColor White
Write-Host ""

Write-Host "🚀 Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   1. Activate the virtual environment:" -ForegroundColor White
Write-Host "      .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "   2. Run the security scanner test:" -ForegroundColor White
Write-Host "      python test-security-scanner.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "   3. Start the backend server:" -ForegroundColor White
Write-Host "      cd backend\src" -ForegroundColor Yellow
Write-Host "      python -m uvicorn main:app --reload" -ForegroundColor Yellow
Write-Host ""
Write-Host "   4. Or use the launch script:" -ForegroundColor White
Write-Host "      .\launch-server.ps1" -ForegroundColor Yellow
Write-Host ""

Write-Host "💡 Tip: If you see import errors in VS Code:" -ForegroundColor Cyan
Write-Host "   1. Open Command Palette (Ctrl+Shift+P)" -ForegroundColor White
Write-Host "   2. Type: Python: Select Interpreter" -ForegroundColor White
Write-Host "   3. Choose: .\venv\Scripts\python.exe" -ForegroundColor White
Write-Host ""
