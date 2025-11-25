#!/usr/bin/env python3
"""
Dependency Check Script for AGI Platform
Checks for all required packages and system components
"""

import sys
import importlib
import subprocess
import platform
import os
from pathlib import Path


class DependencyChecker:
    def __init__(self):
        self.system_info = {}
        self.missing_packages = []
        self.available_packages = []
        self.warnings = []

    def get_system_info(self):
        """Collect system information"""
        self.system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }
        return self.system_info

    def check_package(self, package_name, import_name=None, min_version=None):
        """Check if a package is available and optionally check version"""
        import_name = import_name or package_name
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")

            if min_version and version != "unknown":
                # Simple version comparison (for demonstration)
                from packaging import version as pkg_version

                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    self.warnings.append(
                        f"{package_name} version {version} is below recommended {min_version}"
                    )
                    self.available_packages.append(
                        f"{package_name} ({version}) - WARNING: below {min_version}"
                    )
                else:
                    self.available_packages.append(f"{package_name} ({version})")
            else:
                self.available_packages.append(f"{package_name} ({version})")

            return True
        except ImportError:
            self.missing_packages.append(package_name)
            return False

    def check_package_with_pip(self, package_name):
        """Alternative check using pip show"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Parse version from pip show output
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
                        self.available_packages.append(f"{package_name} ({version})")
                        return True
            return False
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def check_file_handling_deps(self):
        """Check document processing dependencies"""
        print("📄 Checking document processing dependencies...")

        doc_deps = [
            ("python-docx", "docx"),
            ("PyPDF2", "PyPDF2"),
            ("pdfplumber", "pdfplumber"),
            ("Pillow", "PIL", "10.0.0"),  # PIL is the import name for Pillow
            ("python-magic-bin", "magic"),
        ]

        for pip_name, import_name, *version in doc_deps:
            min_version = version[0] if version else None
            self.check_package(pip_name, import_name, min_version)

    def check_web_framework_deps(self):
        """Check web framework dependencies"""
        print("🌐 Checking web framework dependencies...")

        web_deps = [
            ("fastapi", "fastapi", "0.104.0"),
            ("uvicorn", "uvicorn", "0.24.0"),
            ("pydantic", "pydantic", "2.0.0"),
            ("python-multipart", "multipart"),
            ("aiofiles", "aiofiles", "23.0.0"),
        ]

        for pip_name, import_name, *version in web_deps:
            min_version = version[0] if version else None
            self.check_package(pip_name, import_name, min_version)

    def check_ai_ml_deps(self):
        """Check AI/ML dependencies"""
        print("🤖 Checking AI/ML dependencies...")

        ai_deps = [
            ("transformers", "transformers", "4.35.0"),
            ("torch", "torch", "2.1.0"),
            ("torchvision", "torchvision", "0.16.0"),
            ("torchaudio", "torchaudio", "2.1.0"),
            ("numpy", "numpy", "1.24.0"),
            ("pandas", "pandas", "2.0.0"),
        ]

        for pip_name, import_name, *version in ai_deps:
            min_version = version[0] if version else None
            self.check_package(pip_name, import_name, min_version)

    def check_security_deps(self):
        """Check security and authentication dependencies"""
        print("🔐 Checking security dependencies...")

        security_deps = [
            ("python-jose", "jose"),
            ("passlib", "passlib"),
            ("bcrypt", "bcrypt"),
            ("python-dotenv", "dotenv"),
            ("cryptography", "cryptography"),
        ]

        for pip_name, import_name, *version in security_deps:
            min_version = version[0] if version else None
            self.check_package(pip_name, import_name, min_version)

    def check_utility_deps(self):
        """Check utility dependencies"""
        print("🔧 Checking utility dependencies...")

        utility_deps = [
            ("requests", "requests", "2.31.0"),
            ("httpx", "httpx", "0.25.0"),
            ("jinja2", "jinja2", "3.1.0"),
            ("packaging", "packaging"),  # For version comparisons
        ]

        for pip_name, import_name, *version in utility_deps:
            min_version = version[0] if version else None
            self.check_package(pip_name, import_name, min_version)

    def check_environment(self):
        """Check environment variables and paths"""
        print("🏠 Checking environment...")

        # Check virtual environment
        in_venv = sys.prefix != sys.base_prefix
        if in_venv:
            print("✅ Running in virtual environment")
        else:
            self.warnings.append("Not running in virtual environment")

        # Check current directory
        current_dir = Path.cwd()
        print(f"📁 Current directory: {current_dir}")

        # Check if we're in the right location
        expected_dirs = ["backend", "src", "api", "engines"]
        for dir_name in expected_dirs:
            dir_path = current_dir / dir_name
            if dir_path.exists():
                print(f"✅ Found {dir_name}/ directory")
            else:
                self.warnings.append(f"Missing {dir_name}/ directory")

    def check_huggingface_token(self):
        """Check if HuggingFace token is available"""
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("✅ HF_TOKEN environment variable found")
        else:
            self.warnings.append("HF_TOKEN environment variable not set")

    def generate_requirements_file(self):
        """Generate a requirements.txt file with missing packages"""
        if self.missing_packages:
            requirements_content = "\n".join(self.missing_packages)
            with open("missing_requirements.txt", "w", encoding="utf-8") as f:
                f.write(requirements_content)
            print(
                "📝 Generated 'missing_requirements.txt' with "
                f"{len(self.missing_packages)} missing packages"
            )

    def generate_install_script(self):
        """Generate an installation script for missing packages"""
        if self.missing_packages:
            joined = " ".join(self.missing_packages)
            script_content = f"""#!/bin/bash
# Auto-generated installation script
# Run this to install missing dependencies

cd E:\\Projects\\aegis-llm-chain\\backend

# Activate virtual environment (if exists)
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Install missing packages
pip install {joined}

echo "Installation complete!"
"""

            with open("install_missing_deps.sh", "w", encoding="utf-8") as f:
                f.write(script_content)

            ps_script = f"""# Auto-generated PowerShell installation script
cd E:\\Projects\\aegis-llm-chain\\backend

# Activate virtual environment (if exists)
if (Test-Path "venv\\Scripts\\Activate.ps1") {{
    .\\venv\\Scripts\\Activate.ps1
}}

# Install missing packages
pip install {joined}

Write-Host "Installation complete!" -ForegroundColor Green
"""

            with open("install_missing_deps.ps1", "w", encoding="utf-8") as f:
                f.write(ps_script)

            print(
                "📜 Generated installation scripts: "
                "install_missing_deps.sh and install_missing_deps.ps1"
            )

    def run_all_checks(self):
        """Run all dependency checks"""
        print("🔍 Starting comprehensive dependency check...")
        print("=" * 60)

        self.get_system_info()
        self.check_environment()
        self.check_huggingface_token()

        print("\n" + "=" * 60)
        self.check_file_handling_deps()
        self.check_web_framework_deps()
        self.check_ai_ml_deps()
        self.check_security_deps()
        self.check_utility_deps()

        print("\n" + "=" * 60)
        success = self.print_summary()

        if self.missing_packages:
            self.generate_requirements_file()
            self.generate_install_script()

        return success

    def print_summary(self):
        """Print summary of dependency check"""
        print("\n📊 DEPENDENCY CHECK SUMMARY")
        print("=" * 60)

        print(f"✅ Available packages ({len(self.available_packages)}):")
        for pkg in sorted(self.available_packages):
            print(f"   - {pkg}")

        if self.missing_packages:
            print(f"\n❌ Missing packages ({len(self.missing_packages)}):")
            for pkg in sorted(self.missing_packages):
                print(f"   - {pkg}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")

        print("\n" + "=" * 60)

        if not self.missing_packages:
            print("🎉 All dependencies are satisfied! Your environment is ready.")
            return True
        else:
            print(
                "💡 Run the generated installation scripts to install "
                f"{len(self.missing_packages)} missing packages"
            )
            return False


def main():
    """Main function"""
    checker = DependencyChecker()

    try:
        success = checker.run_all_checks()
        return 0 if success else 1
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"❌ Error during dependency check: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
