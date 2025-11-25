"""Security module initialization."""

from .security_scanner import security_scanner, SecurityScanner, Vulnerability, SecurityScanResult
from .monitoring_service import security_monitor, SecurityMonitoringService
from .config import security_config, SecurityConfig

__all__ = [
    'security_scanner',
    'SecurityScanner',
    'Vulnerability',
    'SecurityScanResult',
    'security_monitor',
    'SecurityMonitoringService',
    'security_config',
    'SecurityConfig'
]
"""Security subsystem package.

This provides lightweight mock implementations for development when
full security scanner/monitoring services are not available.
"""
