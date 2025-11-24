# 🛡️ Security Scanner Quick Reference

## Quick Commands

### Run Security Scan
```bash
# Python test script
python test-security-scanner.py

# API endpoint
curl -X POST http://localhost:8000/api/v1/security/scan/start
```

### Get Latest Results
```bash
curl http://localhost:8000/api/v1/security/scan/latest
```

### View Critical Vulnerabilities
```bash
curl "http://localhost:8000/api/v1/security/vulnerabilities?severity=critical"
```

### Start Monitoring
```bash
curl -X POST http://localhost:8000/api/v1/security/monitoring/start \
  -H "Content-Type: application/json" \
  -d '{"scan_type": "standard"}'
```

## Current Vulnerabilities (2025-11-23)

### 🔴 CRITICAL (31)
1. **torch 2.0.0** → Upgrade to 2.6.0
2. **qdrant-client 1.7.0** → Upgrade to 1.9.0
3. **Pillow 10.1.0** → Upgrade to 10.2.0
4. **python-multipart 0.0.6** → Upgrade to 0.0.7
5. Docker weak passwords (2 services)

### 🟠 HIGH (12)
- Various dependency vulnerabilities
- Configuration security issues

## Immediate Action Items

1. **Update Critical Packages:**
```bash
pip install torch==2.6.0 qdrant-client==1.9.0 Pillow==10.2.0 python-multipart==0.0.7
```

2. **Fix Docker Passwords:**
- Update `docker-compose.yml`
- Use environment variables for passwords
- Remove default credentials

3. **Enable Monitoring:**
```bash
curl -X POST http://localhost:8000/api/v1/security/monitoring/start
```

## Key Metrics

- **Total Vulnerabilities:** 46
- **Risk Level:** CRITICAL
- **Scan Duration:** ~77 seconds
- **Last Scan:** 2025-11-23 23:12:26

## Documentation

Full documentation: `docs/SECURITY_SCANNER.md`

## Support

Issues: Create GitHub issue with scan results
