#!/usr/bin/env python3
"""
OpenAudit Comprehensive Test Runner

Easy-to-use script for running all test categories with proper reporting.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """Run a command and return its result"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="OpenAudit Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  all         - Run all tests
  unit        - Unit tests only
  integration - Integration tests only
  modular     - Modular system tests only
  performance - Performance tests only
  legacy      - Legacy system tests only
  quick       - Quick smoke tests
  coverage    - Run with coverage reporting

Examples:
  python tests/run_all_tests.py all
  python tests/run_all_tests.py modular --verbose
  python tests/run_all_tests.py coverage --html
        """,
    )

    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=[
            "all",
            "unit",
            "integration",
            "modular",
            "performance",
            "legacy",
            "quick",
            "coverage",
        ],
        help="Test category to run",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--html", action="store_true", help="Generate HTML coverage report"
    )

    parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run tests in parallel"
    )

    parser.add_argument(
        "--failfast", "-f", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["python", "-m", "pytest", "tests/"]

    # Add common options
    if args.verbose:
        base_cmd.append("-v")

    if args.parallel:
        base_cmd.extend(["-n", "auto"])

    if args.failfast:
        base_cmd.append("-x")

    # Test category configurations
    test_configs = {
        "all": {"cmd": base_cmd, "description": "Running All Tests"},
        "unit": {"cmd": base_cmd + ["-m", "unit"], "description": "Running Unit Tests"},
        "integration": {
            "cmd": base_cmd + ["-m", "integration"],
            "description": "Running Integration Tests",
        },
        "modular": {
            "cmd": base_cmd + ["-k", "modular or registry or profile"],
            "description": "Running Modular System Tests",
        },
        "performance": {
            "cmd": base_cmd + ["-m", "performance"],
            "description": "Running Performance Tests",
        },
        "legacy": {
            "cmd": base_cmd
            + [
                "tests/test_enhanced_features.py",
                "tests/test_error_handling.py",
                "tests/test_cv_consistency.py",
                "tests/test_ceteris_paribus.py",
            ],
            "description": "Running Legacy System Tests",
        },
        "quick": {
            "cmd": base_cmd
            + [
                "-x",
                "--tb=short",
                "-q",
                "tests/test_module_registry.py::TestModuleRegistration::test_register_valid_module",
                "tests/test_analysis_profiles.py::TestAnalysisProfile::test_profile_creation",
                "tests/test_modular_integration.py::TestModuleRegistry::test_module_registration",
            ],
            "description": "Running Quick Smoke Tests",
        },
        "coverage": {
            "cmd": base_cmd + ["--cov=core", "--cov-report=term-missing"],
            "description": "Running Tests with Coverage",
        },
    }

    # Add HTML coverage report if requested
    if args.category == "coverage" and args.html:
        test_configs["coverage"]["cmd"].append("--cov-report=html")
        test_configs["coverage"]["description"] += " (HTML Report)"

    # Get the test configuration
    config = test_configs.get(args.category)
    if not config:
        print(f"❌ Unknown test category: {args.category}")
        return 1

    # Print header
    print("🚀 OpenAudit Test Runner")
    print(f"📁 Project Root: {project_root}")
    print(f"🏷️  Test Category: {args.category}")

    # Run the tests
    success = run_command(config["cmd"], config["description"])

    # Print results
    print(f"\n{'='*60}")
    if success:
        print("✅ All tests passed!")
        if args.category == "coverage":
            print("\n📊 Coverage report generated:")
            if args.html:
                print("   - HTML: htmlcov/index.html")
            print("   - Terminal output above")
    else:
        print("❌ Some tests failed!")
        print("\nFor detailed debugging:")
        print("  pytest tests/ -v -s --tb=long")
        print("  pytest tests/failing_test.py::test_name --pdb")

    print(f"{'='*60}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
