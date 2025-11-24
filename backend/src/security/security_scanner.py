"""
Comprehensive Security Scanner Module
Real-time vulnerability monitoring and security assessment
"""
import asyncio
import aiohttp
import json
import logging
import os
import subprocess
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

@dataclass
class Vulnerability:
    """Vulnerability data structure"""
    id: str
    severity: str  # "critical", "high", "medium", "low"
    package: str
    version: str
    fixed_version: str
    description: str
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    published_date: Optional[str] = None
    references: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)

@dataclass
class SecurityScanResult:
    """Comprehensive security scan results"""
    timestamp: str
    scan_id: str
    vulnerabilities: List[Vulnerability]
    summary: Dict[str, Any]
    recommendations: List[str]
    scan_duration: float

class SecurityScanner:
    """
    Comprehensive Security Scanner
    Monitors vulnerabilities from top security sources and software dependencies
    """
    
    def __init__(self):
        self.scan_results = {}
        self.vulnerability_db = {}
        self.security_sources = self._initialize_security_sources()
        self.scan_interval = 3600  # 1 hour default
        self.last_scan = None
        
        logger.info("🛡️ Security Scanner Initialized")
    
    def _initialize_security_sources(self) -> Dict[str, Any]:
        """Initialize top security vulnerability sources"""
        return {
            "osv": {
                "name": "Open Source Vulnerabilities",
                "url": "https://api.osv.dev/v1/query",
                "enabled": True,
                "rate_limit": 10
            },
            "github_advisory": {
                "name": "GitHub Security Advisories", 
                "url": "https://api.github.com/advisories",
                "enabled": True,
                "rate_limit": 1
            },
            "cisa_kev": {
                "name": "CISA Known Exploited Vulnerabilities",
                "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
                "enabled": True,
                "rate_limit": 2
            }
        }
    
    async def perform_comprehensive_scan(self) -> SecurityScanResult:
        """Perform comprehensive security scan across all components"""
        scan_start = datetime.now()
        scan_id = f"scan_{scan_start.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🔍 Starting comprehensive security scan: {scan_id}")
        
        vulnerabilities = []
        
        try:
            # Scan dependencies
            logger.info("  📦 Scanning dependencies...")
            dependencies = await self._scan_dependencies()
            vulnerabilities.extend(dependencies)
            
            # Scan system configuration
            logger.info("  ⚙️  Scanning system configuration...")
            system_issues = await self._scan_system_configuration()
            vulnerabilities.extend(system_issues)
            
            # Scan Docker configuration
            logger.info("  🐳 Scanning Docker configuration...")
            docker_issues = await self._scan_docker_configuration()
            vulnerabilities.extend(docker_issues)
            
            # Scan for exposed secrets
            logger.info("  🔐 Scanning for exposed secrets...")
            secret_issues = await self._scan_for_exposed_secrets()
            vulnerabilities.extend(secret_issues)
            
            # Generate summary and recommendations
            summary = self._generate_scan_summary(vulnerabilities)
            recommendations = self._generate_recommendations(vulnerabilities)
            
            scan_duration = (datetime.now() - scan_start).total_seconds()
            
            result = SecurityScanResult(
                timestamp=scan_start.isoformat(),
                scan_id=scan_id,
                vulnerabilities=vulnerabilities,
                summary=summary,
                recommendations=recommendations,
                scan_duration=scan_duration
            )
            
            self.scan_results[scan_id] = result
            self.last_scan = scan_start
            
            logger.info(f"✅ Security scan completed: {len(vulnerabilities)} vulnerabilities found in {scan_duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Security scan failed: {str(e)}")
            raise
    
    async def _scan_dependencies(self) -> List[Vulnerability]:
        """Scan Python and Node.js dependencies for vulnerabilities"""
        vulnerabilities = []
        
        # Scan Python dependencies
        python_vulns = await self._scan_python_dependencies()
        vulnerabilities.extend(python_vulns)
        
        # Scan Node.js dependencies
        node_vulns = await self._scan_node_dependencies()
        vulnerabilities.extend(node_vulns)
        
        return vulnerabilities
    
    async def _scan_python_dependencies(self) -> List[Vulnerability]:
        """Scan Python requirements for vulnerabilities"""
        vulnerabilities = []
        
        try:
            req_files = [
                "backend/requirements/base.txt",
                "backend/requirements/prod.txt", 
                "backend/requirements/dev.txt"
            ]
            
            for req_file in req_files:
                if os.path.exists(req_file):
                    deps = await self._parse_python_dependencies(req_file)
                    for dep_name, dep_version in deps.items():
                        vulns = await self._check_package_vulnerabilities(dep_name, dep_version, "PyPI")
                        vulnerabilities.extend(vulns)
                        
        except Exception as e:
            logger.error(f"❌ Python dependency scan failed: {str(e)}")
        
        return vulnerabilities
    
    async def _scan_node_dependencies(self) -> List[Vulnerability]:
        """Scan Node.js package.json for vulnerabilities"""
        vulnerabilities = []
        
        try:
            package_files = ["frontend/package.json"]
            
            for pkg_file in package_files:
                if os.path.exists(pkg_file):
                    deps = await self._parse_node_dependencies(pkg_file)
                    for dep_name, dep_version in deps.items():
                        vulns = await self._check_package_vulnerabilities(dep_name, dep_version, "npm")
                        vulnerabilities.extend(vulns)
                        
        except Exception as e:
            logger.error(f"❌ Node.js dependency scan failed: {str(e)}")
        
        return vulnerabilities
    
    async def _check_package_vulnerabilities(self, package: str, version: str, ecosystem: str) -> List[Vulnerability]:
        """Check package against OSV vulnerability database"""
        vulnerabilities = []
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                osv_data = {
                    "package": {"name": package, "ecosystem": ecosystem},
                    "version": version
                }
                
                try:
                    async with session.post(
                        self.security_sources["osv"]["url"],
                        json=osv_data
                    ) as response:
                        if response.status == 200:
                            osv_result = await response.json()
                            if "vulns" in osv_result:
                                for vuln in osv_result["vulns"]:
                                    vulnerability = Vulnerability(
                                        id=vuln.get("id", "unknown"),
                                        severity=self._calculate_severity(vuln),
                                        package=package,
                                        version=version,
                                        fixed_version=self._get_fixed_version(vuln),
                                        description=vuln.get("summary", "No description"),
                                        cve_id=vuln.get("aliases", [""])[0] if vuln.get("aliases") else None,
                                        published_date=vuln.get("published", ""),
                                        references=[ref.get("url", "") for ref in vuln.get("references", [])],
                                        affected_components=[f"{ecosystem}:{package}"]
                                    )
                                    vulnerabilities.append(vulnerability)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout checking {package} against OSV")
                except Exception as e:
                    logger.warning(f"Error checking {package}: {str(e)}")
                                
        except Exception as e:
            logger.error(f"❌ Vulnerability check failed for {package}: {str(e)}")
        
        return vulnerabilities
    
    async def _scan_system_configuration(self) -> List[Vulnerability]:
        """Scan system configuration for security issues"""
        vulnerabilities = []
        
        try:
            # Check file permissions
            sensitive_files = [".env", ".env.production", "config/secrets.yml"]
            
            for file_path in sensitive_files:
                if os.path.exists(file_path):
                    try:
                        stat_info = os.stat(file_path)
                        permissions = stat_info.st_mode
                        
                        # Check if world-readable
                        if permissions & 0o004:
                            vulnerabilities.append(Vulnerability(
                                id="file-permission-world-readable",
                                severity="high",
                                package="system",
                                version="*",
                                fixed_version="",
                                description=f"Sensitive file {file_path} is world-readable",
                                affected_components=[f"file:{file_path}"]
                            ))
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"❌ System configuration scan failed: {str(e)}")
        
        return vulnerabilities
    
    async def _scan_docker_configuration(self) -> List[Vulnerability]:
        """Scan Docker configuration for security issues"""
        vulnerabilities = []
        
        try:
            compose_files = ["docker-compose.yml", "infrastructure/docker/docker-compose.yml"]
            
            for compose_file in compose_files:
                if os.path.exists(compose_file):
                    issues = await self._analyze_docker_compose(compose_file)
                    vulnerabilities.extend(issues)
            
        except Exception as e:
            logger.error(f"❌ Docker configuration scan failed: {str(e)}")
        
        return vulnerabilities
    
    async def _scan_for_exposed_secrets(self) -> List[Vulnerability]:
        """Scan for exposed secrets in code and configuration"""
        vulnerabilities = []
        
        try:
            secret_patterns = {
                # Exclude os.getenv(), ${VAR}, and environment variable references
                "api_key": r'(?<!getenv\()[aA][pP][iI][_-]?[kK][eE][yY].*?=.*?[\'\"](?!\$\{)(?!HF_TOKEN)([a-zA-Z0-9_-]{20,})[\'\"]',
                "password": r'(?<!getenv\()[pP][aA][sS][sS][wW][oO][rR][dD].*?=.*?[\'\"](?!\$\{)(?!CHANGE)([^\'\"]{8,})[\'\"]',
                "secret": r'(?<!getenv\()[sS][eE][cC][rR][eE][tT].*?=.*?[\'\"](?!\$\{)(?!CHANGE)([^\'\"]{20,})[\'\"]',
                "token": r'(?<!getenv\()[tT][oO][kK][eE][nN].*?=.*?[\'\"](?!\$\{)(?!your_|CHANGE)([a-zA-Z0-9_-]{20,})[\'\"]',
                "private_key": r'-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----'
            }
            
            scan_extensions = ['.py', '.js', '.json', '.yml', '.yaml', '.env']
            exclusions = ['node_modules', 'venv', '__pycache__', '.git', 'dist', 'build']
            
            scan_files = []
            for root, dirs, files in os.walk("."):
                # Remove excluded directories
                dirs[:] = [d for d in dirs if d not in exclusions]
                
                for file in files:
                    if any(file.endswith(ext) for ext in scan_extensions):
                        scan_files.append(os.path.join(root, file))
                
                if len(scan_files) >= 200:  # Limit for performance
                    break
            
            for file_path in scan_files:
                try:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        for secret_type, pattern in secret_patterns.items():
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                # Filter out common false positives
                                for match in matches:
                                    if isinstance(match, tuple):
                                        match = match[0]
                                    
                                    # Skip if match is empty, too short, or contains template placeholders
                                    if not match or len(match) < 10:
                                        continue
                                    
                                    # Enhanced false positive filtering
                                    false_positives = [
                                        'example', 'test', 'demo', 'placeholder', 'change_me', 
                                        'your_', 'insert_', 'xxx', 'sample', 'template',
                                        'token_here', 'hf_token', 'api_key_env', 'tokenurl'
                                    ]
                                    
                                    if any(fp in match.lower() for fp in false_positives):
                                        continue
                                    
                                    vulnerabilities.append(Vulnerability(
                                        id=f"exposed-secret-{secret_type}",
                                        severity="critical",
                                        package="secrets",
                                        version="*",
                                        fixed_version="",
                                        description=f"Potential {secret_type} exposed in {file_path}",
                                        affected_components=[f"file:{file_path}"]
                                    ))
                                    break  # Only report once per file
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Secret scanning failed: {str(e)}")
        
        return vulnerabilities
    
    async def _analyze_docker_compose(self, compose_file: str) -> List[Vulnerability]:
        """Analyze docker-compose configuration"""
        issues = []
        
        try:
            with open(compose_file, "r") as f:
                content = yaml.safe_load(f)
                
                if content and "services" in content:
                    for service_name, service_config in content["services"].items():
                        # Check for privileged mode
                        if service_config.get("privileged", False):
                            issues.append(Vulnerability(
                                id="docker-privileged-mode",
                                severity="high",
                                package="docker",
                                version="*", 
                                fixed_version="",
                                description=f"Service {service_name} runs in privileged mode",
                                affected_components=[f"compose:{compose_file}"]
                            ))
                        
                        # Check for host network
                        if service_config.get("network_mode") == "host":
                            issues.append(Vulnerability(
                                id="docker-host-network",
                                severity="medium",
                                package="docker",
                                version="*",
                                fixed_version="", 
                                description=f"Service {service_name} uses host network",
                                affected_components=[f"compose:{compose_file}"]
                            ))
                        
                        # Check for default passwords in environment
                        env_vars = service_config.get("environment", [])
                        if isinstance(env_vars, list):
                            for env_var in env_vars:
                                if isinstance(env_var, str) and any(weak in env_var.lower() for weak in ['password=password', 'password=admin', 'password=postgres']):
                                    issues.append(Vulnerability(
                                        id="docker-weak-password",
                                        severity="critical",
                                        package="docker",
                                        version="*",
                                        fixed_version="",
                                        description=f"Service {service_name} uses weak default password",
                                        affected_components=[f"compose:{compose_file}"]
                                    ))
                            
        except Exception as e:
            logger.error(f"❌ Docker compose analysis failed: {str(e)}")
        
        return issues
    
    def _calculate_severity(self, vulnerability: Dict) -> str:
        """Calculate severity based on CVSS score or database severity"""
        # Check database severity first
        if "database_specific" in vulnerability:
            severity = vulnerability["database_specific"].get("severity", "").lower()
            if severity in ["critical", "high", "medium", "low"]:
                return severity
        
        # Fallback to CVSS score
        cvss_score = 0.0
        if "severity" in vulnerability:
            for sev in vulnerability["severity"]:
                if sev.get("type") == "CVSS_V3":
                    score_value = sev.get("score", 0)
                    # Handle both numeric scores and CVSS vector strings
                    try:
                        if isinstance(score_value, (int, float)):
                            cvss_score = float(score_value)
                        elif isinstance(score_value, str):
                            # If it's a CVSS vector string, skip it
                            if score_value.startswith("CVSS:"):
                                continue
                            # Try to parse as float
                            cvss_score = float(score_value)
                    except (ValueError, TypeError):
                        logger.debug(f"Could not parse CVSS score: {score_value}")
                        continue
                    break
        
        if cvss_score >= 9.0:
            return "critical"
        elif cvss_score >= 7.0:
            return "high"
        elif cvss_score >= 4.0:
            return "medium"
        else:
            return "low"
    
    def _get_fixed_version(self, vulnerability: Dict) -> str:
        """Extract fixed version from vulnerability data"""
        for affected in vulnerability.get("affected", []):
            for ranges in affected.get("ranges", []):
                for event in ranges.get("events", []):
                    if "fixed" in event:
                        return event["fixed"]
        return "unknown"
    
    def _generate_scan_summary(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        """Generate summary of scan results"""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for vuln in vulnerabilities:
            severity_counts[vuln.severity] += 1
        
        total_vulns = len(vulnerabilities)
        critical_high = severity_counts["critical"] + severity_counts["high"]
        
        return {
            "total_vulnerabilities": total_vulns,
            "severity_breakdown": severity_counts,
            "critical_high_count": critical_high,
            "risk_level": "critical" if severity_counts["critical"] > 0 else "high" if critical_high > 0 else "medium" if total_vulns > 0 else "low"
        }
    
    def _generate_recommendations(self, vulnerabilities: List[Vulnerability]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        critical_vulns = [v for v in vulnerabilities if v.severity in ["critical", "high"]]
        
        if critical_vulns:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Update critical/high severity packages")
            seen_packages = set()
            for vuln in critical_vulns[:5]:  # Top 5 critical
                if vuln.package not in seen_packages:
                    recommendations.append(f"  • Update {vuln.package} from {vuln.version} to {vuln.fixed_version}")
                    seen_packages.add(vuln.package)
        
        # General recommendations
        recommendations.extend([
            "🔒 Enable automatic security updates where possible",
            "📊 Schedule regular security scans (daily recommended)",
            "🔍 Monitor security advisories for all dependencies",
            "🚫 Implement proper access controls and authentication",
            "📝 Maintain an incident response plan",
            "🔐 Review and rotate all API keys and secrets",
            "🐳 Update Docker base images regularly",
            "📋 Document all security policies and procedures"
        ])
        
        return recommendations
    
    async def _parse_python_dependencies(self, file_path: str) -> Dict[str, str]:
        """Parse Python dependencies from requirements file"""
        deps = {}
        try:
            with open(file_path, "r") as f:
                for line in f:
                    name, version = self._parse_dependency_string(line.strip())
                    if name:
                        deps[name] = version
        except Exception as e:
            logger.error(f"❌ Failed to parse Python dependencies from {file_path}: {str(e)}")
        return deps
    
    async def _parse_node_dependencies(self, file_path: str) -> Dict[str, str]:
        """Parse Node.js dependencies from package.json"""
        deps = {}
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                for name, version in all_deps.items():
                    # Clean version string
                    version = version.lstrip('^~>=<')
                    deps[name] = version
        except Exception as e:
            logger.error(f"❌ Failed to parse Node.js dependencies from {file_path}: {str(e)}")
        return deps
    
    def _parse_dependency_string(self, dep_string: str) -> Tuple[Optional[str], str]:
        """Parse dependency string into name and version"""
        try:
            dep_string = dep_string.split("#")[0].strip()
            if not dep_string or dep_string.startswith("-") or dep_string.startswith("http"):
                return None, ""
            
            if "==" in dep_string:
                name, version = dep_string.split("==", 1)
            elif ">=" in dep_string:
                name, version = dep_string.split(">=", 1)
            elif "@" in dep_string:
                name, version = dep_string.split("@", 1)
            else:
                name, version = dep_string, ""
            
            return name.strip(), version.strip()
        except:
            return None, ""

# Global security scanner instance
security_scanner = SecurityScanner()
