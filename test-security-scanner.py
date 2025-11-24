"""
Quick test of the security scanner module
"""
import asyncio
import sys
import os

# Add backend/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def test_security_scanner():
    """Test the security scanner"""
    print("=" * 60)
    print("🛡️  SECURITY SCANNER TEST")
    print("=" * 60)
    
    try:
        from security.security_scanner import security_scanner
        from security.monitoring_service import security_monitor
        
        print("\n✅ Security modules imported successfully!")
        print(f"   📊 Scanner initialized: {security_scanner is not None}")
        print(f"   📊 Monitor initialized: {security_monitor is not None}")
        
        # Run a quick scan
        print("\n" + "=" * 60)
        print("🔍 Running Security Scan...")
        print("=" * 60)
        
        scan_result = await security_scanner.perform_comprehensive_scan()
        
        print(f"\n✅ Scan completed successfully!")
        print(f"   ⏱️  Duration: {scan_result.scan_duration:.2f}s")
        print(f"   🆔 Scan ID: {scan_result.scan_id}")
        print(f"   📊 Total Vulnerabilities: {len(scan_result.vulnerabilities)}")
        
        # Display summary
        print("\n" + "=" * 60)
        print("📊 VULNERABILITY SUMMARY")
        print("=" * 60)
        
        summary = scan_result.summary
        severity_breakdown = summary['severity_breakdown']
        
        print(f"\n   🔴 CRITICAL: {severity_breakdown['critical']}")
        print(f"   🟠 HIGH:     {severity_breakdown['high']}")
        print(f"   🟡 MEDIUM:   {severity_breakdown['medium']}")
        print(f"   🔵 LOW:      {severity_breakdown['low']}")
        print(f"\n   ⚠️  Risk Level: {summary['risk_level'].upper()}")
        
        # Display top critical vulnerabilities
        if severity_breakdown['critical'] > 0:
            print("\n" + "=" * 60)
            print("🚨 TOP CRITICAL VULNERABILITIES")
            print("=" * 60)
            
            critical_vulns = [v for v in scan_result.vulnerabilities if v.severity == "critical"]
            for i, vuln in enumerate(critical_vulns[:5], 1):
                print(f"\n{i}. {vuln.id}")
                print(f"   Package: {vuln.package} {vuln.version}")
                print(f"   Description: {vuln.description}")
                if vuln.fixed_version:
                    print(f"   Fix: Upgrade to {vuln.fixed_version}")
        
        # Display recommendations
        print("\n" + "=" * 60)
        print("💡 RECOMMENDATIONS")
        print("=" * 60)
        
        for i, rec in enumerate(scan_result.recommendations[:8], 1):
            print(f"{i}. {rec}")
        
        # Test monitoring service
        print("\n" + "=" * 60)
        print("📊 MONITORING SERVICE STATUS")
        print("=" * 60)
        
        status = security_monitor.get_monitoring_status()
        print(f"\n   Monitoring Active: {status['is_monitoring']}")
        print(f"   Last Scan: {status['last_scan']}")
        print(f"   Total Vulnerabilities: {status['total_vulnerabilities']}")
        print(f"   Critical Count: {status['critical_count']}")
        print(f"   High Count: {status['high_count']}")
        
        print("\n" + "=" * 60)
        print("✅ SECURITY SCANNER TEST COMPLETE")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {str(e)}")
        print("   Make sure all dependencies are installed:")
        print("   pip install aiohttp pyyaml")
        return False
    except Exception as e:
        print(f"\n❌ Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_security_scanner())
    sys.exit(0 if success else 1)
