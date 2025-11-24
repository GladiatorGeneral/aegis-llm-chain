"""
Security API Endpoints
REST API for security scanner and monitoring
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Optional, List
from pydantic import BaseModel
import logging

from backend.src.security.security_scanner import security_scanner
from backend.src.security.monitoring_service import security_monitor

logger = logging.getLogger(__name__)

security_router = APIRouter(prefix="/security", tags=["security"])

# Request/Response Models
class ScanRequest(BaseModel):
    scan_type: str = "standard"  # quick, standard, full

class ScanStatusResponse(BaseModel):
    status: str
    scan_id: Optional[str] = None
    message: str

class MonitoringRequest(BaseModel):
    scan_type: str = "standard"
    interval: Optional[int] = None

# API Endpoints

@security_router.get("/health")
async def security_health():
    """Security scanner health check"""
    return {
        "status": "healthy",
        "scanner": "operational",
        "monitoring": security_monitor.is_monitoring
    }

@security_router.post("/scan/start", response_model=ScanStatusResponse)
async def start_security_scan(
    background_tasks: BackgroundTasks,
    scan_type: str = Query(default="standard", regex="^(quick|standard|full)$")
):
    """
    Start a new security scan
    
    - **quick**: Fast dependency scan (15 minutes)
    - **standard**: Comprehensive scan (1 hour)
    - **full**: Deep scan with all checks (24 hours)
    """
    try:
        logger.info(f"🔍 Starting {scan_type} security scan via API")
        
        # Run scan in background
        background_tasks.add_task(security_scanner.perform_comprehensive_scan)
        
        return ScanStatusResponse(
            status="started",
            scan_id=None,
            message=f"{scan_type.capitalize()} security scan started successfully"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start security scan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start scan: {str(e)}")

@security_router.get("/scan/latest")
async def get_latest_scan():
    """Get the latest security scan results"""
    try:
        if not security_monitor.last_scan_result:
            return {
                "status": "no_data",
                "message": "No security scans performed yet"
            }
        
        scan_result = security_monitor.last_scan_result
        
        return {
            "status": "success",
            "scan_id": scan_result.scan_id,
            "timestamp": scan_result.timestamp,
            "summary": scan_result.summary,
            "total_vulnerabilities": len(scan_result.vulnerabilities),
            "vulnerabilities": [
                security_monitor._serialize_vulnerability(v) 
                for v in scan_result.vulnerabilities
            ],
            "recommendations": scan_result.recommendations,
            "scan_duration": scan_result.scan_duration
        }
    except Exception as e:
        logger.error(f"❌ Failed to get latest scan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scan results: {str(e)}")

@security_router.get("/scan/results/{scan_id}")
async def get_scan_results(scan_id: str):
    """Get specific scan results by scan ID"""
    try:
        if scan_id in security_scanner.scan_results:
            scan_result = security_scanner.scan_results[scan_id]
            return {
                "status": "success",
                "scan_id": scan_result.scan_id,
                "timestamp": scan_result.timestamp,
                "summary": scan_result.summary,
                "vulnerabilities": [
                    security_monitor._serialize_vulnerability(v) 
                    for v in scan_result.vulnerabilities
                ],
                "recommendations": scan_result.recommendations
            }
        else:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get scan results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scan results: {str(e)}")

@security_router.get("/vulnerabilities")
async def get_vulnerabilities(
    severity: Optional[str] = Query(None, regex="^(critical|high|medium|low)$"),
    package: Optional[str] = None,
    limit: int = Query(default=100, le=1000)
):
    """
    Query vulnerabilities with filters
    
    - **severity**: Filter by severity level (critical, high, medium, low)
    - **package**: Filter by package name
    - **limit**: Maximum number of results
    """
    try:
        if not security_monitor.last_scan_result:
            return {
                "status": "no_data",
                "vulnerabilities": [],
                "count": 0
            }
        
        vulnerabilities = security_monitor.last_scan_result.vulnerabilities
        
        # Apply filters
        if severity:
            vulnerabilities = [v for v in vulnerabilities if v.severity == severity.lower()]
        
        if package:
            vulnerabilities = [v for v in vulnerabilities if v.package == package]
        
        # Apply limit
        vulnerabilities = vulnerabilities[:limit]
        
        return {
            "status": "success",
            "count": len(vulnerabilities),
            "vulnerabilities": [
                security_monitor._serialize_vulnerability(v) 
                for v in vulnerabilities
            ]
        }
    except Exception as e:
        logger.error(f"❌ Failed to query vulnerabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query vulnerabilities: {str(e)}")

@security_router.get("/vulnerabilities/{vulnerability_id}")
async def get_vulnerability_details(vulnerability_id: str):
    """Get detailed information about a specific vulnerability"""
    try:
        details = await security_monitor.get_vulnerability_details(vulnerability_id)
        
        if details:
            return {
                "status": "success",
                "vulnerability": details
            }
        else:
            raise HTTPException(status_code=404, detail=f"Vulnerability {vulnerability_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get vulnerability details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve vulnerability details: {str(e)}")

@security_router.get("/monitoring/status")
async def get_monitoring_status():
    """Get security monitoring status"""
    try:
        status = security_monitor.get_monitoring_status()
        return {
            "status": "success",
            "monitoring": status
        }
    except Exception as e:
        logger.error(f"❌ Failed to get monitoring status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve monitoring status: {str(e)}")

@security_router.post("/monitoring/start")
async def start_monitoring(request: MonitoringRequest):
    """
    Start continuous security monitoring
    
    - **scan_type**: quick (15min), standard (1hr), full (24hr)
    """
    try:
        await security_monitor.start_continuous_monitoring(scan_type=request.scan_type)
        
        return {
            "status": "success",
            "message": f"Security monitoring started with {request.scan_type} scan type",
            "is_monitoring": True
        }
    except Exception as e:
        logger.error(f"❌ Failed to start monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@security_router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop continuous security monitoring"""
    try:
        await security_monitor.stop_monitoring()
        
        return {
            "status": "success",
            "message": "Security monitoring stopped",
            "is_monitoring": False
        }
    except Exception as e:
        logger.error(f"❌ Failed to stop monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@security_router.get("/dashboard")
async def get_security_dashboard():
    """Get comprehensive security dashboard data"""
    try:
        dashboard = security_monitor.get_security_dashboard()
        return {
            "status": "success",
            "dashboard": dashboard
        }
    except Exception as e:
        logger.error(f"❌ Failed to get dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")

@security_router.get("/recommendations")
async def get_security_recommendations():
    """Get security recommendations based on latest scan"""
    try:
        if not security_monitor.last_scan_result:
            return {
                "status": "no_data",
                "recommendations": []
            }
        
        return {
            "status": "success",
            "recommendations": security_monitor.last_scan_result.recommendations,
            "scan_timestamp": security_monitor.last_scan_result.timestamp
        }
    except Exception as e:
        logger.error(f"❌ Failed to get recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recommendations: {str(e)}")

@security_router.get("/dependencies")
async def get_dependencies():
    """Get list of all scanned dependencies"""
    try:
        if not security_monitor.last_scan_result:
            return {
                "status": "no_data",
                "dependencies": []
            }
        
        # Extract unique packages
        packages = {}
        for vuln in security_monitor.last_scan_result.vulnerabilities:
            if vuln.package not in packages:
                packages[vuln.package] = {
                    "name": vuln.package,
                    "version": vuln.version,
                    "vulnerability_count": 0,
                    "highest_severity": "low"
                }
            
            packages[vuln.package]["vulnerability_count"] += 1
            
            # Update highest severity
            severity_order = ["low", "medium", "high", "critical"]
            current_severity = packages[vuln.package]["highest_severity"]
            if severity_order.index(vuln.severity) > severity_order.index(current_severity):
                packages[vuln.package]["highest_severity"] = vuln.severity
        
        return {
            "status": "success",
            "dependencies": list(packages.values()),
            "total_packages": len(packages)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get dependencies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dependencies: {str(e)}")

@security_router.get("/alerts/history")
async def get_alert_history(limit: int = Query(default=50, le=500)):
    """Get security alert history"""
    try:
        alerts = security_monitor.alert_history[-limit:]
        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get alert history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alert history: {str(e)}")
