@echo off
chcp 65001 >nul
echo 🔍 Running AGI Platform Dependency Checks...
echo.

cd /d E:\Projects\aegis-llm-chain\backend\src

echo 📋 Running comprehensive check...
python check_dependencies.py

echo.
echo 🚀 Running quick check...
python quick_check.py

echo.
echo 📝 Check complete! Review the output above.
pause
