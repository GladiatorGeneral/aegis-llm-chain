#!/usr/bin/env python3
"""
Quick Dependency Check for AGI Platform
Fast check for critical dependencies only
"""

import importlib
import sys


def quick_check():
    """Quick check for critical dependencies"""
    critical_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "transformers",
        "torch",
        "docx",
        "magic",
        "PyPDF2",
        "pdfplumber",
        "PIL",
    ]

    print("🚀 Quick Dependency Check")
    print("=" * 40)

    missing = []
    available = []

    for package in critical_packages:
        try:
            importlib.import_module(package)
            available.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")

    print("=" * 40)

    if missing:
        print(f"❌ Missing {len(missing)} critical packages:")
        print("pip install " + " ".join(missing))
        return False
    else:
        print("✅ All critical packages available!")
        return True


if __name__ == "__main__":
    success = quick_check()
    sys.exit(0 if success else 1)
