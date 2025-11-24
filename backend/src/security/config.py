"""
Security Scanner Configuration
"""
import os
from typing import Dict, Any

class SecurityConfig:
    """Security scanner configuration"""
    
    # Scan intervals in seconds
    SCAN_INTERVALS = {
        "quick": 900,      # 15 minutes
        "standard": 3600,  # 1 hour
        "full": 86400      # 24 hours
    }
    
    # Alert thresholds
    ALERT_THRESHOLDS = {
        "critical": 0,     # Any critical vulnerabilities
        "high": 1,         # 1 or more high severity
        "medium": 5,       # 5 or more medium severity  
        "low": 10          # 10 or more low severity
    }
    
    # Security sources configuration
    SECURITY_SOURCES = {
        "nvd": {
            "enabled": True,
            "api_key": os.getenv("NVD_API_KEY", ""),
            "rate_limit": 5
        },
        "github_advisory": {
            "enabled": True,
            "rate_limit": 1
        },
        "snyk": {
            "enabled": True,
            "api_key": os.getenv("SNYK_API_KEY", ""),
            "rate_limit": 10
        },
        "osv": {
            "enabled": True,
            "rate_limit": 10
        },
        "cisa_kev": {
            "enabled": True,
            "rate_limit": 2
        }
    }
    
    # File patterns to scan for secrets
    SECRET_PATTERNS = {
        "api_key": r'[aA][pP][iI][_-]?[kK][eE][yY].*?[\'\"]([^\'\"]{10,50})[\'\"]',
        "password": r'[pP][aA][sS][sS][wW][oO][rR][dD].*?[\'\"]([^\'\"]{5,30})[\'\"]',
        "secret": r'[sS][eE][cC][rR][eE][tT].*?[\'\"]([^\'\"]{10,50})[\'\"]',
        "token": r'[tT][oO][kK][eE][nN].*?[\'\"]([^\'\"]{10,100})[\'\"]',
        "private_key": r'-----BEGIN PRIVATE KEY-----',
        "rsa_key": r'-----BEGIN RSA PRIVATE KEY-----'
    }
    
    # File extensions to scan
    SCAN_EXTENSIONS = ['.py', '.js', '.json', '.yml', '.yaml', '.env', '.toml', '.ini', '.conf']
    
    # Files/directories to exclude from scanning
    SCAN_EXCLUSIONS = [
        'node_modules',
        'venv',
        'env',
        '__pycache__',
        '.git',
        'dist',
        'build',
        '.next',
        'model_cache'
    ]

# Global configuration instance
security_config = SecurityConfig()
