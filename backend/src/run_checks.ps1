# AGI Platform Dependency Check Script
Write-Host "🔍 Running AGI Platform Dependency Checks..." -ForegroundColor Cyan

# Change to project directory
Set-Location "E:\Projects\aegis-llm-chain\backend\src"

# Run comprehensive check
Write-Host "`n📋 Running comprehensive check..." -ForegroundColor Yellow
python check_dependencies.py

# Run quick check
Write-Host "`n🚀 Running quick check..." -ForegroundColor Yellow
python quick_check.py

Write-Host "`n📝 Check complete! Review the output above." -ForegroundColor Green
Read-Host "Press Enter to continue"
