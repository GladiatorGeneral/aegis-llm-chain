#!/usr/bin/env python3
"""
MASTER CODE ANALYSIS SCRIPT
Combined perspectives: Web App Engineer + Security Engineer + Network Engineer + Master Coder
"""
import os
import ast
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    severity: str  # critical, high, medium, low
    category: str  # security, performance, network, code_quality
    file_path: str
    line_number: int
    description: str
    recommendation: str
    perspective: str  # web_app, security, network, code


class MasterCodeAnalyzer:
    """Multi-perspective code analyzer."""

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.issues: List[SecurityIssue] = []
        self.stats = {
            'files_analyzed': 0,
            'issues_found': 0,
            'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'by_perspective': {'web_app': 0, 'security': 0, 'network': 0, 'code': 0},
        }

    def analyze_entire_codebase(self) -> Dict[str, Any]:
        logger.info("🚀 Starting master code analysis...")

        analysis_methods = [
            self._analyze_python_code,
            self._analyze_api_endpoints,
            self._analyze_docker_config,
            self._analyze_dependencies,
            self._analyze_environment_config,
            self._analyze_security_headers,
            self._analyze_authentication,
            self._analyze_network_config,
            self._analyze_performance,
            self._analyze_frontend_integration,
        ]

        for method in analysis_methods:
            try:
                method()
            except Exception as e:  # noqa: BLE001
                logger.error("Analysis method %s failed: %s", method.__name__, e)

        return self._generate_report()

    # --- Core analyzers -------------------------------------------------

    def _analyze_python_code(self) -> None:
        logger.info("🔍 Analyzing Python code...")
        python_files = list(self.codebase_path.rglob("*.py"))

        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                self._analyze_ast_tree(tree, py_file, content)
                self._check_security_issues(content, py_file)
                self._check_code_quality(content, py_file)

                self.stats['files_analyzed'] += 1
            except (SyntaxError, UnicodeDecodeError) as e:
                self._add_issue(
                    severity='medium',
                    category='code_quality',
                    file_path=str(py_file),
                    line_number=0,
                    description=f"File parsing error: {e}",
                    recommendation="Fix syntax errors or encoding issues",
                    perspective='code',
                )

    def _analyze_ast_tree(self, tree: ast.AST, file_path: Path, content: str) -> None:
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, analyzer: 'MasterCodeAnalyzer', file_path: Path, content: str) -> None:
                self.analyzer = analyzer
                self.file_path = file_path
                self.lines = content.split("\n")

            def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id

                    if func_name in {"eval", "exec", "compile"}:
                        self.analyzer._add_issue(
                            severity='critical',
                            category='security',
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            description=f"Dangerous function call: {func_name}",
                            recommendation="Avoid eval/exec/compile in production code",
                            perspective='security',
                        )

                    if func_name in {"execute", "executemany"} and self._is_sql_string(node):
                        self.analyzer._add_issue(
                            severity='high',
                            category='security',
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            description="Potential SQL injection vulnerability",
                            recommendation="Use parameterized queries or an ORM",
                            perspective='security',
                        )

                self.generic_visit(node)

            def _is_sql_string(self, node: ast.Call) -> bool:
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        sql = first_arg.value.lower()
                        return any(op in sql for op in ("select", "insert", "update", "delete"))
                return False

            def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
                if (
                    node.targets
                    and isinstance(node.targets[0], ast.Name)
                    and any(k in node.targets[0].id.lower() for k in ("key", "secret", "password", "token"))
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    self.analyzer._add_issue(
                        severity='high',
                        category='security',
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                        description="Hardcoded secret detected",
                        recommendation="Use environment variables or secret management",
                        perspective='security',
                    )
                self.generic_visit(node)

        SecurityVisitor(self, file_path, content).visit(tree)

    def _check_security_issues(self, content: str, file_path: Path) -> None:
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            lower = line.lower()

            if "jwt" in lower and "secret" in lower and "hardcode" in lower:
                self._add_issue(
                    severity='critical',
                    category='security',
                    file_path=str(file_path),
                    line_number=i,
                    description="Hardcoded JWT secret",
                    recommendation="Use environment variables for JWT secrets",
                    perspective='security',
                )

            if "cors" in lower and 'allow_origins=["*"]' in line:
                self._add_issue(
                    severity='high',
                    category='security',
                    file_path=str(file_path),
                    line_number=i,
                    description="Overly permissive CORS configuration",
                    recommendation="Specify exact origins instead of wildcard",
                    perspective='web_app',
                )

    def _check_code_quality(self, content: str, file_path: Path) -> None:
        lines = content.split("\n")
        function_lines = 0
        in_function = False

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith("def ") or stripped.startswith("async def "):
                if in_function and function_lines > 50:
                    self._add_issue(
                        severity='low',
                        category='code_quality',
                        file_path=str(file_path),
                        line_number=i,
                        description=f"Large function detected ({function_lines} lines)",
                        recommendation="Refactor into smaller, focused functions",
                        perspective='code',
                    )
                in_function = True
                function_lines = 0
            elif in_function:
                if stripped and not stripped.startswith("#"):
                    function_lines += 1
                if stripped == "" and function_lines > 0:
                    in_function = False

    def _analyze_api_endpoints(self) -> None:
        logger.info("🔍 Analyzing API endpoints...")
        main_files = list(self.codebase_path.rglob("main.py")) + list(self.codebase_path.rglob("app.py"))

        for main_file in main_files:
            try:
                content = main_file.read_text(encoding="utf-8")
                if "FastAPI(" in content:
                    if 'CORSMiddleware' not in content:
                        self._add_issue(
                            severity='high',
                            category='security',
                            file_path=str(main_file),
                            line_number=0,
                            description="CORS middleware not configured",
                            recommendation="Add CORSMiddleware for web integration",
                            perspective='web_app',
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", main_file, e)

    def _analyze_docker_config(self) -> None:
        logger.info("🔍 Analyzing Docker configuration...")
        docker_files = list(self.codebase_path.rglob("Dockerfile*")) + list(
            self.codebase_path.rglob("docker-compose*"),
        )

        for docker_file in docker_files:
            try:
                content = docker_file.read_text(encoding="utf-8")
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    lower = line.lower()
                    if "user root" in lower:
                        self._add_issue(
                            severity='high',
                            category='security',
                            file_path=str(docker_file),
                            line_number=i,
                            description="Container runs as root user",
                            recommendation="Create and use a non-root user",
                            perspective='security',
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", docker_file, e)

    def _analyze_dependencies(self) -> None:
        logger.info("🔍 Analyzing dependencies...")
        req_files = list(self.codebase_path.rglob("requirements*.txt")) + list(
            self.codebase_path.rglob("pyproject.toml"),
        )

        for req_file in req_files:
            try:
                if req_file.suffix == ".txt":
                    self._analyze_requirements_txt(req_file)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", req_file, e)

    def _analyze_requirements_txt(self, file_path: Path) -> None:
        content = file_path.read_text(encoding="utf-8")
        for i, line in enumerate(content.split("\n"), 1):
            line = line.strip()
            if line and not line.startswith("#") and "==" not in line and ">=" not in line:
                self._add_issue(
                    severity='medium',
                    category='security',
                    file_path=str(file_path),
                    line_number=i,
                    description=f"Unpinned dependency: {line}",
                    recommendation="Pin to specific versions for reproducible builds",
                    perspective='security',
                )

    def _analyze_environment_config(self) -> None:
        logger.info("🔍 Analyzing environment configuration...")
        env_files = list(self.codebase_path.rglob(".env*"))

        for env_file in env_files:
            try:
                content = env_file.read_text(encoding="utf-8")
                for i, line in enumerate(content.split("\n"), 1):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        key, value = stripped.split("=", 1)
                        if any(w in key.lower() for w in ("secret", "key", "password", "token")):
                            if len(value) < 16 and value not in ("", "${}"):
                                self._add_issue(
                                    severity='high',
                                    category='security',
                                    file_path=str(env_file),
                                    line_number=i,
                                    description=f"Potentially weak secret: {key}",
                                    recommendation="Use strong, randomly generated secrets",
                                    perspective='security',
                                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", env_file, e)

    def _analyze_security_headers(self) -> None:
        main_files = list(self.codebase_path.rglob("main.py"))
        headers = [
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]
        for main_file in main_files:
            try:
                content = main_file.read_text(encoding="utf-8")
                missing = [h for h in headers if h not in content]
                if missing:
                    self._add_issue(
                        severity='medium',
                        category='security',
                        file_path=str(main_file),
                        line_number=0,
                        description=f"Missing security headers: {', '.join(missing)}",
                        recommendation="Implement standard security headers on responses",
                        perspective='web_app',
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze security headers in %s: %s", main_file, e)

    def _analyze_authentication(self) -> None:
        logger.info("🔍 Analyzing authentication...")
        auth_files = list(self.codebase_path.rglob("*auth*.py"))
        for auth_file in auth_files:
            try:
                content = auth_file.read_text(encoding="utf-8").lower()
                if "jwt" in content and "hs256" in content and "secret" in content:
                    if "os.getenv" not in content and "environment" not in content:
                        self._add_issue(
                            severity='high',
                            category='security',
                            file_path=str(auth_file),
                            line_number=0,
                            description="JWT secret may be hardcoded",
                            recommendation="Store JWT secret in environment variables",
                            perspective='security',
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", auth_file, e)

    def _analyze_network_config(self) -> None:
        logger.info("🔍 Analyzing network configuration...")
        compose_files = list(self.codebase_path.rglob("docker-compose*.yml"))
        for compose_file in compose_files:
            try:
                content = compose_file.read_text(encoding="utf-8")
                data = yaml.safe_load(content)
                if data and isinstance(data, dict) and "services" in data:
                    for service_name, cfg in data["services"].items():
                        if "healthcheck" not in cfg:
                            self._add_issue(
                                severity='medium',
                                category='code_quality',
                                file_path=str(compose_file),
                                line_number=0,
                                description=f"Service {service_name} missing healthcheck",
                                recommendation="Add health checks for reliable orchestration",
                                perspective='network',
                            )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", compose_file, e)

    def _analyze_performance(self) -> None:
        logger.info("🔍 Analyzing performance...")
        python_files = list(self.codebase_path.rglob("*.py"))
        sync_calls = ["requests.", "time.sleep", "open("]

        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                if "async def" in content:
                    for call in sync_calls:
                        if call in content:
                            self._add_issue(
                                severity='low',
                                category='performance',
                                file_path=str(py_file),
                                line_number=0,
                                description=f"Synchronous call in async context: {call}",
                                recommendation="Use async alternatives for better performance",
                                perspective='web_app',
                            )
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze performance in %s: %s", py_file, e)

    def _analyze_frontend_integration(self) -> None:
        logger.info("🔍 Analyzing frontend integration...")
        api_root = self.codebase_path / "backend" / "src" / "api"
        if not api_root.exists():
            return
        api_files = list(api_root.rglob("*.py"))
        endpoints: Dict[str, List[str]] = {}

        for api_file in api_files:
            try:
                content = api_file.read_text(encoding="utf-8")
                if "@" in content and "(" in content:
                    for line in content.split("\n"):
                        if "@" in line and '("/' in line:
                            start = line.find('("/') + 2
                            end = line.find('"', start)
                            if end != -1:
                                endpoint = line[start:end]
                                endpoints.setdefault(endpoint, []).append(str(api_file))
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", api_file, e)

        for endpoint, files in endpoints.items():
            if len(files) > 1:
                self._add_issue(
                    severity='low',
                    category='code_quality',
                    file_path=files[0],
                    line_number=0,
                    description=f"Endpoint {endpoint} defined in multiple files",
                    recommendation="Consolidate endpoint definitions",
                    perspective='web_app',
                )

    # --- Helpers --------------------------------------------------------

    def _should_skip_file(self, file_path: Path) -> bool:
        skip_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            "venv",
            ".venv",
            "env",
            "dist",
            "build",
        ]
        return any(p in str(file_path) for p in skip_patterns)

    def _add_issue(
        self,
        severity: str,
        category: str,
        file_path: str,
        line_number: int,
        description: str,
        recommendation: str,
        perspective: str,
    ) -> None:
        issue = SecurityIssue(
            severity=severity,
            category=category,
            file_path=file_path,
            line_number=line_number,
            description=description,
            recommendation=recommendation,
            perspective=perspective,
        )
        self.issues.append(issue)
        self.stats['issues_found'] += 1
        self.stats['by_severity'][severity] += 1
        self.stats['by_perspective'][perspective] += 1

    def _generate_report(self) -> Dict[str, Any]:
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self.issues.sort(key=lambda x: severity_order[x.severity])

        summary = {
            "files_analyzed": self.stats['files_analyzed'],
            "total_issues": self.stats['issues_found'],
            "issues_by_severity": self.stats['by_severity'],
            "issues_by_perspective": self.stats['by_perspective'],
        }

        issues = [
            {
                "severity": i.severity,
                "category": i.category,
                "file": i.file_path,
                "line": i.line_number,
                "description": i.description,
                "recommendation": i.recommendation,
                "perspective": i.perspective,
            }
            for i in self.issues
        ]

        return {"summary": summary, "issues": issues, "recommendations": self._generate_recommendations()}

    def _generate_recommendations(self) -> List[str]:
        recs: List[str] = []
        critical = [i for i in self.issues if i.severity == "critical"]
        security = [i for i in self.issues if i.perspective == "security"]
        performance = [i for i in self.issues if i.category == "performance"]
        codeq = [i for i in self.issues if i.perspective == "code"]
        network = [i for i in self.issues if i.perspective == "network"]

        if critical:
            recs.append("🚨 IMMEDIATE ACTION REQUIRED: Fix critical security issues")
        if security:
            recs.append("🛡️  Enhance security: Address authentication and secret management issues")
        if performance:
            recs.append("⚡ Optimize performance: Fix async operations and resource usage")
        if codeq:
            recs.append("🔧 Improve code quality: Refactor large functions and fix patterns")
        if network:
            recs.append("🌐 Strengthen network: Add health checks and resource limits")
        return recs


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Master Code Analysis Tool")
    parser.add_argument("path", help="Path to codebase to analyze")
    parser.add_argument("--output", "-o", help="Output file for report (JSON)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"❌ Path does not exist: {args.path}")
        return 1

    analyzer = MasterCodeAnalyzer(args.path)
    report = analyzer.analyze_entire_codebase()

    print("\n" + "=" * 80)
    print("🎯 MASTER CODE ANALYSIS REPORT")
    print("=" * 80)

    summary = report["summary"]
    print("\n📊 Summary:")
    print(f"   Files analyzed: {summary['files_analyzed']}")
    print(f"   Total issues: {summary['total_issues']}")
    print(f"   By severity: {summary['issues_by_severity']}")
    print(f"   By perspective: {summary['issues_by_perspective']}")

    critical_high = [i for i in report["issues"] if i["severity"] in {"critical", "high"}]
    if critical_high:
        print(f"\n🚨 CRITICAL/HIGH ISSUES ({len(critical_high)}):")
        for issue in critical_high:
            print(f"   {issue['severity'].upper()}: {issue['file']}:{issue['line']}")
            print(f"      {issue['description']}")
            print(f"      💡 {issue['recommendation']}")
            print()

    if report["recommendations"]:
        print("\n🎯 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Full report saved to: {args.output}")

    critical_count = summary['issues_by_severity']['critical']
    return 1 if critical_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
