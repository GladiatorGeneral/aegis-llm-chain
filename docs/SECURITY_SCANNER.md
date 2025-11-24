# 🛡️ Security Scanner Documentation

## Overview

The comprehensive security scanner provides real-time vulnerability monitoring and security assessment for the AGI Platform. It integrates with multiple security intelligence sources to detect vulnerabilities across dependencies, Docker configurations, and exposed secrets.

## Architecture

### Components

```
backend/src/security/
├── __init__.py                    # Module initialization
├── config.py                      # Security configuration
├── security_scanner.py            # Core scanning engine
├── monitoring_service.py          # Continuous monitoring
└── README.md                      # This documentation

backend/src/api/
└── security.py                    # REST API endpoints
```

### Security Intelligence Sources

The scanner integrates with 6 major security databases:

1. **OSV (Open Source Vulnerabilities)** - https://osv.dev
   - Comprehensive vulnerability database for open source
   - Real-time updates from multiple sources
   - Package-specific vulnerability tracking

2. **GitHub Security Advisories** - https://github.com/advisories
   - GitHub's curated vulnerability database
   - Integration with Dependabot alerts
   - Detailed remediation guidance

3. **CISA KEV (Known Exploited Vulnerabilities)** - https://www.cisa.gov/known-exploited-vulnerabilities
   - Actively exploited vulnerabilities
   - US Government cybersecurity intelligence
   - Critical infrastructure focus

4. **Snyk Database** (configurable)
   - Commercial vulnerability intelligence
   - Advanced threat analysis
   - Developer-friendly remediation advice

5. **Debian Security Tracker** (configurable)
   - Linux package vulnerabilities
   - Distribution-specific security updates
   - System-level vulnerability tracking

6. **NVD (National Vulnerability Database)** (configurable)
   - NIST vulnerability database
   - CVSS scoring and metrics
   - Comprehensive CVE information

## Features

### 1. Dependency Scanning

**Python Dependencies:**
- Scans `requirements/base.txt`, `requirements/prod.txt`, `requirements/dev.txt`
- Checks each package against OSV vulnerability database
- Identifies outdated packages with known vulnerabilities
- Provides upgrade recommendations with fixed versions

**Node.js Dependencies:**
- Scans `frontend/package.json`
- Checks npm packages for vulnerabilities
- Tracks both dependencies and devDependencies
- Identifies security patches and updates

**Docker Dependencies:**
- Analyzes Docker base images
- Checks for outdated container images
- Identifies image vulnerabilities
- Recommends secure base images

### 2. Configuration Security

**System Configuration:**
- File permission checks for sensitive files
- Environment variable security validation
- Secret key strength validation
- Database credential security

**Docker Configuration:**
- Docker compose security analysis
- Privileged container detection
- Host network mode warnings
- Default password detection
- Volume mount security checks

### 3. Secret Detection

Scans codebase for exposed secrets using regex patterns:
- API keys
- Passwords
- Secret keys
- Tokens
- Private keys (RSA, EC, DSA)

**Scanned Extensions:**
`.py`, `.js`, `.json`, `.yml`, `.yaml`, `.env`, `.toml`, `.ini`, `.conf`

**Excluded Directories:**
`node_modules`, `venv`, `env`, `__pycache__`, `.git`, `dist`, `build`

### 4. Continuous Monitoring

Three scan intervals available:

- **Quick Scan** (15 minutes): Fast dependency check
- **Standard Scan** (1 hour): Comprehensive vulnerability scan
- **Full Scan** (24 hours): Deep scan with all security checks

### 5. Alert System

Configurable severity thresholds:
- **Critical**: Alert on any critical vulnerability (threshold: 0)
- **High**: Alert when 1+ high severity vulnerabilities found
- **Medium**: Alert when 5+ medium severity vulnerabilities found
- **Low**: Alert when 10+ low severity vulnerabilities found

## API Reference

### Base URL
```
http://localhost:8000/api/v1/security
```

### Endpoints

#### 1. Health Check
```bash
GET /security/health
```

**Response:**
```json
{
  "status": "healthy",
  "scanner": "operational",
  "monitoring": false
}
```

#### 2. Start Security Scan
```bash
POST /security/scan/start?scan_type=standard
```

**Parameters:**
- `scan_type`: `quick` | `standard` | `full` (default: `standard`)

**Response:**
```json
{
  "status": "started",
  "scan_id": null,
  "message": "Standard security scan started successfully"
}
```

#### 3. Get Latest Scan Results
```bash
GET /security/scan/latest
```

**Response:**
```json
{
  "status": "success",
  "scan_id": "scan_20251123_231226",
  "timestamp": "2025-11-23T23:12:26",
  "summary": {
    "total_vulnerabilities": 46,
    "severity_breakdown": {
      "critical": 31,
      "high": 12,
      "medium": 0,
      "low": 3
    },
    "critical_high_count": 43,
    "risk_level": "critical"
  },
  "vulnerabilities": [...],
  "recommendations": [...],
  "scan_duration": 77.73
}
```

#### 4. Get Specific Scan Results
```bash
GET /security/scan/results/{scan_id}
```

#### 5. Query Vulnerabilities
```bash
GET /security/vulnerabilities?severity=critical&limit=100
```

**Query Parameters:**
- `severity`: `critical` | `high` | `medium` | `low` (optional)
- `package`: Filter by package name (optional)
- `limit`: Max results (default: 100, max: 1000)

**Response:**
```json
{
  "status": "success",
  "count": 31,
  "vulnerabilities": [
    {
      "id": "GHSA-53q9-r3pm-6pq6",
      "severity": "critical",
      "package": "torch",
      "version": "2.0.0",
      "fixed_version": "2.6.0",
      "description": "PyTorch: `torch.load` with `weights_only=True` leads to remote code execution",
      "cve_id": "CVE-2024-12345",
      "cvss_score": 9.8,
      "published_date": "2024-10-15",
      "references": ["https://github.com/advisories/GHSA-53q9-r3pm-6pq6"],
      "affected_components": ["PyPI:torch"]
    }
  ]
}
```

#### 6. Get Vulnerability Details
```bash
GET /security/vulnerabilities/{vulnerability_id}
```

#### 7. Start Continuous Monitoring
```bash
POST /security/monitoring/start
```

**Request Body:**
```json
{
  "scan_type": "standard",
  "interval": null
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Security monitoring started with standard scan type",
  "is_monitoring": true
}
```

#### 8. Stop Monitoring
```bash
POST /security/monitoring/stop
```

#### 9. Get Monitoring Status
```bash
GET /security/monitoring/status
```

**Response:**
```json
{
  "status": "success",
  "monitoring": {
    "is_monitoring": false,
    "last_scan": "2025-11-23T23:12:26",
    "total_vulnerabilities": 46,
    "critical_count": 31,
    "high_count": 12,
    "recent_alerts": []
  }
}
```

#### 10. Get Security Dashboard
```bash
GET /security/dashboard
```

**Response:**
```json
{
  "status": "active",
  "last_scan": {
    "timestamp": "2025-11-23T23:12:26",
    "scan_id": "scan_20251123_231226",
    "duration": 77.73
  },
  "summary": {...},
  "risk_score": {
    "score": 95,
    "level": "critical",
    "max_score": 100
  },
  "top_vulnerabilities": {
    "critical": [...],
    "high": [...]
  },
  "trends": {...},
  "recommendations": [...],
  "alert_history": [...],
  "monitoring_enabled": false
}
```

#### 11. Get Security Recommendations
```bash
GET /security/recommendations
```

#### 12. Get Dependencies
```bash
GET /security/dependencies
```

**Response:**
```json
{
  "status": "success",
  "dependencies": [
    {
      "name": "torch",
      "version": "2.0.0",
      "vulnerability_count": 3,
      "highest_severity": "critical"
    }
  ],
  "total_packages": 45
}
```

#### 13. Get Alert History
```bash
GET /security/alerts/history?limit=50
```

## Usage Examples

### Quick Start

1. **Run a single scan:**
```bash
curl -X POST http://localhost:8000/api/v1/security/scan/start?scan_type=standard
```

2. **Get results:**
```bash
curl http://localhost:8000/api/v1/security/scan/latest
```

3. **View critical vulnerabilities:**
```bash
curl "http://localhost:8000/api/v1/security/vulnerabilities?severity=critical"
```

### Enable Continuous Monitoring

```bash
# Start monitoring with hourly scans
curl -X POST http://localhost:8000/api/v1/security/monitoring/start \
  -H "Content-Type: application/json" \
  -d '{"scan_type": "standard"}'

# Check status
curl http://localhost:8000/api/v1/security/monitoring/status

# Stop monitoring
curl -X POST http://localhost:8000/api/v1/security/monitoring/stop
```

### Python Client Example

```python
import asyncio
from security.security_scanner import security_scanner
from security.monitoring_service import security_monitor

async def scan_example():
    # Perform scan
    result = await security_scanner.perform_comprehensive_scan()
    
    # Print summary
    print(f"Found {len(result.vulnerabilities)} vulnerabilities")
    print(f"Critical: {result.summary['severity_breakdown']['critical']}")
    
    # Get recommendations
    for rec in result.recommendations[:5]:
        print(f"- {rec}")
    
    # Start monitoring
    await security_monitor.start_continuous_monitoring("standard")
    
    # Get dashboard
    dashboard = security_monitor.get_security_dashboard()
    print(f"Risk Score: {dashboard['risk_score']['score']}/100")

# Run
asyncio.run(scan_example())
```

## Configuration

Edit `backend/src/security/config.py` to customize:

### Scan Intervals
```python
SCAN_INTERVALS = {
    "quick": 900,      # 15 minutes
    "standard": 3600,  # 1 hour
    "full": 86400      # 24 hours
}
```

### Alert Thresholds
```python
ALERT_THRESHOLDS = {
    "critical": 0,   # Alert on any critical
    "high": 1,       # Alert on 1+ high
    "medium": 5,     # Alert on 5+ medium
    "low": 10        # Alert on 10+ low
}
```

### Security Sources
```python
SECURITY_SOURCES = {
    "osv": {
        "enabled": True,
        "api_key_env": "OSV_API_KEY",
        "rate_limit": 10
    },
    # Add custom sources...
}
```

## Test Results (2025-11-23)

**Scan Duration:** 77.73 seconds  
**Total Vulnerabilities:** 46

### Severity Breakdown
- 🔴 **CRITICAL**: 31
- 🟠 **HIGH**: 12
- 🟡 **MEDIUM**: 0
- 🔵 **LOW**: 3

### Top Critical Vulnerabilities

1. **GHSA-53q9-r3pm-6pq6** - PyTorch RCE
   - Package: `torch 2.0.0`
   - Fix: Upgrade to `2.6.0`

2. **GHSA-7m75-x27w-r52r** - Qdrant Input Validation
   - Package: `qdrant-client 1.7.0`
   - Fix: Upgrade to `1.9.0`

3. **GHSA-3f63-hfp8-52jq** - Pillow Code Execution
   - Package: `Pillow 10.1.0`
   - Fix: Upgrade to `10.2.0`

4. **docker-weak-password** - Weak Database Password
   - Component: Docker Compose `db` service
   - Fix: Use strong passwords from environment variables

5. **docker-weak-password** - Weak Admin Password
   - Component: Docker Compose `pgadmin` service
   - Fix: Use strong passwords from environment variables

## Security Best Practices

### 1. Regular Scanning
- Run scans daily or on each deployment
- Enable continuous monitoring in production
- Review critical vulnerabilities immediately

### 2. Dependency Management
- Keep dependencies up to date
- Use version pinning in production
- Test updates in staging environment

### 3. Secret Management
- Never commit secrets to version control
- Use environment variables for sensitive data
- Rotate secrets regularly
- Use secret management services (AWS Secrets Manager, HashiCorp Vault)

### 4. Docker Security
- Use official base images
- Avoid privileged containers
- Don't use default passwords
- Implement least privilege access
- Scan images before deployment

### 5. Alert Response
- Define incident response procedures
- Set up alert notifications (email, Slack, PagerDuty)
- Assign security responsibilities
- Document remediation steps

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r backend/requirements/base.txt
      
      - name: Run security scan
        run: python test-security-scanner.py
      
      - name: Upload scan results
        uses: actions/upload-artifact@v2
        with:
          name: security-scan-results
          path: scan-results.json
```

## Troubleshooting

### Issue: Scan times out
**Solution:** Increase timeout in `aiohttp.ClientSession` or reduce concurrent checks

### Issue: Too many false positives
**Solution:** Adjust secret patterns in `config.py` to exclude test data

### Issue: API rate limits exceeded
**Solution:** Reduce scan frequency or use API keys for higher limits

### Issue: Missing vulnerabilities
**Solution:** Ensure all security sources are enabled and API keys are configured

## Future Enhancements

- [ ] Integration with SIEM systems
- [ ] Machine learning for vulnerability prioritization
- [ ] Automated patching recommendations
- [ ] Integration with ticketing systems (Jira, ServiceNow)
- [ ] Historical trend analysis
- [ ] Compliance reporting (SOC 2, ISO 27001)
- [ ] Container image scanning (Trivy integration)
- [ ] License compliance checking
- [ ] Supply chain attack detection

## Support

For issues or questions:
1. Check this documentation
2. Review test script: `test-security-scanner.py`
3. Check logs in console output
4. Open GitHub issue with scan results and error messages

## License

Part of AGI Platform - See main LICENSE file
