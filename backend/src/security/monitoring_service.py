import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class MockVulnerability:
    def __init__(self, vulnerability_id: str = "mock-vuln") -> None:
        self.id = vulnerability_id
        self.package = "mock-package"
        self.version = "0.0.0"
        self.severity = "low"
        self.description = "Mock vulnerability. No real risk."


class MockScanResult:
    def __init__(self) -> None:
        self.scan_id = "mock-scan"
        self.timestamp = "1970-01-01T00:00:00Z"
        self.summary = "No real scan performed."
        self.vulnerabilities: List[MockVulnerability] = []
        self.recommendations: List[str] = []
        self.scan_duration = 0.0


class SecurityMonitor:
    """Lightweight mock security monitoring service for development.

    Provides the attributes and methods expected by the API layer
    without performing any real security monitoring.
    """

    def __init__(self) -> None:
        self.is_monitoring: bool = False
        self.last_scan_result: Optional[MockScanResult] = None
        self.alert_history: List[Dict[str, Any]] = []

    def _serialize_vulnerability(self, v: MockVulnerability) -> Dict[str, Any]:
        return {
            "id": v.id,
            "package": v.package,
            "version": v.version,
            "severity": v.severity,
            "description": v.description,
        }

    async def get_vulnerability_details(self, vulnerability_id: str) -> Optional[Dict[str, Any]]:
        if not self.last_scan_result:
            return None
        for v in self.last_scan_result.vulnerabilities:
            if v.id == vulnerability_id:
                return self._serialize_vulnerability(v)
        return None

    def get_monitoring_status(self) -> Dict[str, Any]:
        return {"is_monitoring": self.is_monitoring}

    async def start_continuous_monitoring(self, scan_type: str = "standard") -> None:
        logger.info("Starting mock continuous security monitoring (type=%s)", scan_type)
        self.is_monitoring = True
        if self.last_scan_result is None:
            self.last_scan_result = MockScanResult()

    async def stop_monitoring(self) -> None:
        logger.info("Stopping mock continuous security monitoring")
        self.is_monitoring = False

    def get_security_dashboard(self) -> Dict[str, Any]:
        return {
            "summary": "Mock security dashboard. No real data.",
            "is_monitoring": self.is_monitoring,
            "last_scan_timestamp": getattr(self.last_scan_result, "timestamp", None),
        }


security_monitor = SecurityMonitor()
"""
Security Monitoring Service
Continuous vulnerability monitoring with alerting
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .security_scanner import security_scanner, SecurityScanResult, Vulnerability
from .config import SecurityConfig

logger = logging.getLogger(__name__)

class SecurityMonitoringService:
    """
    Continuous Security Monitoring Service
    Monitors vulnerabilities and generates alerts
    """
    
    def __init__(self):
        self.config = SecurityConfig()
        self.is_monitoring = False
        self.monitoring_task = None
        self.alert_history = []
        self.last_scan_result: Optional[SecurityScanResult] = None
        
        logger.info("🛡️ Security Monitoring Service Initialized")
    
    async def start_continuous_monitoring(self, scan_type: str = "standard"):
        """Start continuous security monitoring"""
        if self.is_monitoring:
            logger.warning("⚠️  Security monitoring already running")
            return
        
        self.is_monitoring = True
        scan_interval = self.config.SCAN_INTERVALS.get(scan_type, 3600)
        
        logger.info(f"▶️  Starting continuous security monitoring (interval: {scan_interval}s)")
        
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(scan_interval)
        )
    
    async def stop_monitoring(self):
        """Stop continuous security monitoring"""
        if not self.is_monitoring:
            logger.warning("⚠️  Security monitoring not running")
            return
        
        logger.info("⏹  Stopping security monitoring...")
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Security monitoring stopped")
    
    async def _monitoring_loop(self, scan_interval: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                logger.info(f"🔍 Running scheduled security scan...")
                
                # Perform scan
                scan_result = await security_scanner.perform_comprehensive_scan()
                self.last_scan_result = scan_result
                
                # Check for alerts
                await self._check_and_alert(scan_result)
                
                logger.info(f"✅ Scan complete. Next scan in {scan_interval}s")
                
                # Wait for next scan
                await asyncio.sleep(scan_interval)
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {str(e)}")
                # Continue monitoring despite errors
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _check_and_alert(self, scan_result: SecurityScanResult):
        """Check scan results and generate alerts"""
        severity_counts = scan_result.summary["severity_breakdown"]
        
        # Check thresholds for each severity level
        alerts = []
        
        for severity, count in severity_counts.items():
            threshold = self.config.ALERT_THRESHOLDS.get(severity, 0)
            if count >= threshold:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "severity": severity,
                    "count": count,
                    "threshold": threshold,
                    "scan_id": scan_result.scan_id,
                    "message": f"🚨 {severity.upper()}: {count} vulnerabilities found (threshold: {threshold})"
                }
                alerts.append(alert)
                self.alert_history.append(alert)
        
        # Send alerts
        if alerts:
            await self._send_alerts(alerts, scan_result)
    
    async def _send_alerts(self, alerts: List[Dict], scan_result: SecurityScanResult):
        """Send security alerts"""
        for alert in alerts:
            logger.warning(f"🚨 SECURITY ALERT: {alert['message']}")
            
            # Log critical vulnerabilities
            if alert["severity"] == "critical":
                critical_vulns = [v for v in scan_result.vulnerabilities if v.severity == "critical"]
                logger.critical(f"⚠️  CRITICAL VULNERABILITIES FOUND: {len(critical_vulns)}")
                for vuln in critical_vulns[:5]:  # Log top 5
                    logger.critical(f"  • {vuln.package} {vuln.version}: {vuln.description}")
        
        # In production, you would integrate with:
        # - Email notifications (SMTP)
        # - Slack/Teams webhooks
        # - PagerDuty / Incident management systems
        # - SIEM systems
        # - Custom alert handlers
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "is_monitoring": self.is_monitoring,
            "last_scan": self.last_scan_result.timestamp if self.last_scan_result else None,
            "total_vulnerabilities": len(self.last_scan_result.vulnerabilities) if self.last_scan_result else 0,
            "critical_count": sum(1 for v in (self.last_scan_result.vulnerabilities if self.last_scan_result else []) if v.severity == "critical"),
            "high_count": sum(1 for v in (self.last_scan_result.vulnerabilities if self.last_scan_result else []) if v.severity == "high"),
            "recent_alerts": self.alert_history[-10:] if self.alert_history else []
        }
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data"""
        if not self.last_scan_result:
            return {
                "status": "no_data",
                "message": "No security scans performed yet"
            }
        
        # Calculate trends
        vulnerability_trends = self._calculate_vulnerability_trends()
        
        # Get top vulnerabilities
        top_critical = [v for v in self.last_scan_result.vulnerabilities if v.severity == "critical"][:10]
        top_high = [v for v in self.last_scan_result.vulnerabilities if v.severity == "high"][:10]
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(self.last_scan_result)
        
        return {
            "status": "active",
            "last_scan": {
                "timestamp": self.last_scan_result.timestamp,
                "scan_id": self.last_scan_result.scan_id,
                "duration": self.last_scan_result.scan_duration
            },
            "summary": self.last_scan_result.summary,
            "risk_score": risk_score,
            "top_vulnerabilities": {
                "critical": [self._serialize_vulnerability(v) for v in top_critical],
                "high": [self._serialize_vulnerability(v) for v in top_high]
            },
            "trends": vulnerability_trends,
            "recommendations": self.last_scan_result.recommendations,
            "alert_history": self.alert_history[-20:],
            "monitoring_enabled": self.is_monitoring
        }
    
    def _calculate_vulnerability_trends(self) -> Dict[str, Any]:
        """Calculate vulnerability trends over time"""
        # In production, store historical scan data
        # For now, return placeholder
        return {
            "trend": "stable",
            "change_percentage": 0,
            "historical_scans": 1
        }
    
    def _calculate_risk_score(self, scan_result: SecurityScanResult) -> Dict[str, Any]:
        """Calculate overall security risk score"""
        severity_weights = {
            "critical": 10,
            "high": 5,
            "medium": 2,
            "low": 1
        }
        
        weighted_score = 0
        for vuln in scan_result.vulnerabilities:
            weighted_score += severity_weights.get(vuln.severity, 0)
        
        # Normalize to 0-100 scale
        max_score = 100
        normalized_score = min(weighted_score, max_score)
        
        # Determine risk level
        if normalized_score >= 80:
            risk_level = "critical"
        elif normalized_score >= 50:
            risk_level = "high"
        elif normalized_score >= 20:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "score": normalized_score,
            "level": risk_level,
            "max_score": max_score
        }
    
    def _serialize_vulnerability(self, vuln: Vulnerability) -> Dict[str, Any]:
        """Serialize vulnerability for JSON response"""
        return {
            "id": vuln.id,
            "severity": vuln.severity,
            "package": vuln.package,
            "version": vuln.version,
            "fixed_version": vuln.fixed_version,
            "description": vuln.description,
            "cve_id": vuln.cve_id,
            "cvss_score": vuln.cvss_score,
            "published_date": vuln.published_date,
            "references": vuln.references,
            "affected_components": vuln.affected_components
        }
    
    async def get_vulnerability_details(self, vulnerability_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific vulnerability"""
        if not self.last_scan_result:
            return None
        
        for vuln in self.last_scan_result.vulnerabilities:
            if vuln.id == vulnerability_id:
                return self._serialize_vulnerability(vuln)
        
        return None
    
    async def get_vulnerabilities_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get all vulnerabilities of a specific severity"""
        if not self.last_scan_result:
            return []
        
        filtered_vulns = [
            v for v in self.last_scan_result.vulnerabilities 
            if v.severity == severity.lower()
        ]
        
        return [self._serialize_vulnerability(v) for v in filtered_vulns]
    
    async def get_vulnerabilities_by_package(self, package: str) -> List[Dict[str, Any]]:
        """Get all vulnerabilities for a specific package"""
        if not self.last_scan_result:
            return []
        
        filtered_vulns = [
            v for v in self.last_scan_result.vulnerabilities 
            if v.package == package
        ]
        
        return [self._serialize_vulnerability(v) for v in filtered_vulns]

# Global monitoring service instance
security_monitor = SecurityMonitoringService()
