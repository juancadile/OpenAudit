#!/usr/bin/env python3
"""
OpenAudit Command Line Interface.

Comprehensive CLI for managing the OpenAudit bias testing platform,
including the new modular analysis system.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from core.analysis_profiles import AnalysisProfile, get_global_profile_manager
from core.external_module_loader import get_global_module_loader
from core.module_registry import get_global_registry

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class OpenAuditCLI:
    """OpenAudit Command Line Interface."""

    def __init__(self):
        """Initialize CLI with global managers."""
        self.registry = get_global_registry()
        self.profile_manager = get_global_profile_manager()
        self.module_loader = get_global_module_loader()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            description="OpenAudit - AI Bias Testing Platform",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s start                          # Start web interface
  %(prog)s modules list                   # List available modules
  %(prog)s modules info enhanced_stats    # Get module information
  %(prog)s profiles list                  # List analysis profiles
  %(prog)s profiles create my_profile     # Create custom profile
  %(prog)s analyze run_123 --profile comprehensive  # Analyze with profile
  %(prog)s test quick                     # Run quick tests
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Start command (existing functionality)
        start_parser = subparsers.add_parser(
            "start", help="Start OpenAudit web interface"
        )
        start_parser.add_argument(
            "--port", "-p", type=int, default=5100, help="Port to run web interface on"
        )
        start_parser.add_argument(
            "--debug", action="store_true", help="Run in debug mode"
        )

        # Module management commands
        modules_parser = subparsers.add_parser(
            "modules", help="Manage analysis modules"
        )
        modules_subparsers = modules_parser.add_subparsers(dest="modules_action")

        # modules list
        modules_subparsers.add_parser("list", help="List available modules")

        # modules info
        info_parser = modules_subparsers.add_parser(
            "info", help="Get module information"
        )
        info_parser.add_argument("module_name", help="Name of module to get info about")

        # modules register (for external modules)
        register_parser = modules_subparsers.add_parser(
            "register", help="Register an external module"
        )
        register_parser.add_argument("name", help="Module name")
        register_parser.add_argument("path", help="Path to module file")
        register_parser.add_argument(
            "--class-name", help="Module class name (if different from filename)"
        )
        register_parser.add_argument(
            "--no-security-check",
            action="store_true",
            help="Skip security validation (use with caution)",
        )

        # modules create (template)
        create_parser = modules_subparsers.add_parser(
            "create", help="Create external module template"
        )
        create_parser.add_argument("name", help="Module name")
        create_parser.add_argument("--author", default="Unknown", help="Module author")
        create_parser.add_argument("--description", help="Module description")

        # modules load-all
        modules_subparsers.add_parser(
            "load-all", help="Load all external modules from directory"
        )

        # modules list-external
        modules_subparsers.add_parser(
            "list-external", help="List loaded external modules"
        )

        # modules test
        test_parser = modules_subparsers.add_parser(
            "test", help="Test module compatibility"
        )
        test_parser.add_argument("module_names", nargs="+", help="Module names to test")

        # Profile management commands
        profiles_parser = subparsers.add_parser(
            "profiles", help="Manage analysis profiles"
        )
        profiles_subparsers = profiles_parser.add_subparsers(dest="profiles_action")

        # profiles list
        profiles_subparsers.add_parser("list", help="List available profiles")

        # profiles show
        show_parser = profiles_subparsers.add_parser(
            "show", help="Show profile details"
        )
        show_parser.add_argument("profile_name", help="Name of profile to show")

        # profiles create
        create_parser = profiles_subparsers.add_parser(
            "create", help="Create new profile"
        )
        create_parser.add_argument("name", help="Profile name")
        create_parser.add_argument("--description", "-d", help="Profile description")
        create_parser.add_argument(
            "--modules",
            "-m",
            nargs="+",
            required=True,
            help="List of modules to include",
        )
        create_parser.add_argument(
            "--params", "-p", help="JSON string of default parameters"
        )

        # profiles validate
        validate_parser = profiles_subparsers.add_parser(
            "validate", help="Validate profile"
        )
        validate_parser.add_argument("profile_name", help="Name of profile to validate")

        # Analysis commands
        analyze_parser = subparsers.add_parser("analyze", help="Run bias analysis")
        analyze_parser.add_argument("run_id", help="Run ID or path to data file")
        analyze_parser.add_argument("--profile", help="Analysis profile to use")
        analyze_parser.add_argument(
            "--modules", "-m", nargs="+", help="Specific modules to use"
        )
        analyze_parser.add_argument("--output", "-o", help="Output file for results")
        analyze_parser.add_argument(
            "--format", choices=["json", "text"], default="text", help="Output format"
        )
        analyze_parser.add_argument(
            "--alpha",
            type=float,
            default=0.05,
            help="Significance level (default: 0.05)",
        )
        analyze_parser.add_argument(
            "--effect-size-threshold",
            type=float,
            default=0.1,
            help="Effect size threshold (default: 0.1)",
        )

        # Test commands
        test_parser = subparsers.add_parser("test", help="Run test suite")
        test_parser.add_argument(
            "suite",
            choices=["quick", "modular", "all"],
            default="quick",
            help="Test suite to run",
        )
        test_parser.add_argument(
            "--verbose", "-v", action="store_true", help="Verbose output"
        )

        # Status command
        subparsers.add_parser("status", help="Show system status")

        # Version command
        subparsers.add_parser("version", help="Show version information")

        return parser

    def cmd_start(self, args) -> int:
        """Start the web interface"""
        print("🚀 Starting OpenAudit - Complete Bias Testing Platform")
        print("=" * 60)
        print("📊 Historical Analysis: ✓ Available")
        print("🔬 Live Experiments: ✓ Available")
        print("🧩 Modular Analysis: ✓ Available")
        print(f"🌐 Dashboard: http://localhost:{args.port}")
        print("=" * 60)
        print("💡 Use Ctrl+C to stop the server")
        print()

        try:
            # Set environment variables for configuration
            os.environ["OPENAUDIT_PORT"] = str(args.port)
            if args.debug:
                os.environ["OPENAUDIT_DEBUG"] = "1"

            # Run the unified interface
            subprocess.run([sys.executable, "unified_interface.py"], check=True)
        except KeyboardInterrupt:
            print("\n👋 OpenAudit stopped successfully!")
        except Exception as e:
            print(f"❌ Error starting OpenAudit: {e}")
            return 1

        return 0

    def cmd_modules_list(self, args) -> int:
        """List available modules"""
        modules = self.registry.get_available_modules()

        if not modules:
            print("📦 No modules currently registered")
            print("\nTo register built-in modules, run:")
            print("  python cli.py modules register-builtin")
            return 0

        print(f"📦 Available Analysis Modules ({len(modules)})")
        print("=" * 50)

        for module_name in sorted(modules):
            info = self.registry.get_module_info(module_name)
            if info:
                print(f"• {module_name}")
                print(f"  Description: {info.get('description', 'No description')}")
                print(f"  Version: {info.get('version', 'Unknown')}")
                print(f"  Capabilities: {', '.join(info.get('capabilities', []))}")
                print()
            else:
                print(f"• {module_name} (info unavailable)")

        return 0

    def cmd_modules_info(self, args) -> int:
        """Get detailed module information"""
        module_name = args.module_name

        if not self.registry.has_module(module_name):
            print(f"❌ Module '{module_name}' not found")
            return 1

        info = self.registry.get_module_info(module_name)
        if not info:
            print(f"❌ Could not retrieve info for module '{module_name}'")
            return 1

        print(f"📋 Module Information: {module_name}")
        print("=" * 50)
        print(f"Name: {info.get('name', 'Unknown')}")
        print(f"Version: {info.get('version', 'Unknown')}")
        print(f"Description: {info.get('description', 'No description')}")
        print(f"Dependencies: {', '.join(info.get('dependencies', [])) or 'None'}")
        print(f"Capabilities: {', '.join(info.get('capabilities', [])) or 'None'}")

        # Test module compatibility
        print("\n🔧 Compatibility Test:")
        compatibility = self.registry.check_module_compatibility(module_name)
        if compatibility["compatible"]:
            print("✅ Module is compatible")
        else:
            print("❌ Module has compatibility issues:")
            for issue in compatibility["issues"]:
                print(f"  • {issue}")

        return 0

    def cmd_modules_test(self, args) -> int:
        """Test module compatibility"""
        module_names = args.module_names

        print(f"🔧 Testing Compatibility: {', '.join(module_names)}")
        print("=" * 50)

        # Check individual modules
        for module_name in module_names:
            if not self.registry.has_module(module_name):
                print(f"❌ Module '{module_name}' not found")
                continue

            compatibility = self.registry.check_module_compatibility(module_name)
            if compatibility["compatible"]:
                print(f"✅ {module_name}: Compatible")
            else:
                print(f"❌ {module_name}: Issues found")
                for issue in compatibility["issues"]:
                    print(f"  • {issue}")

        # Test group compatibility
        print(f"\n🔗 Group Compatibility Test:")
        group_compatibility = self.registry.validate_module_compatibility(module_names)
        if group_compatibility["compatible"]:
            print("✅ All modules are compatible with each other")
        else:
            print("❌ Group compatibility issues:")
            for issue in group_compatibility["issues"]:
                print(f"  • {issue}")

        return 0

    def cmd_modules_register(self, args) -> int:
        """Register an external module"""
        module_path = Path(args.path)
        module_name = args.name
        class_name = getattr(args, "class_name", None)
        skip_security = getattr(args, "no_security_check", False)

        print(f"📦 Registering external module: {module_name}")
        print(f"📁 Source: {module_path}")
        print("=" * 50)

        if not module_path.exists():
            print(f"❌ Module file not found: {module_path}")
            return 1

        # Load the module
        result = self.module_loader.load_module_from_file(
            module_path=module_path,
            module_name=module_name,
            class_name=class_name,
            validate_security=not skip_security,
            auto_register=True,
        )

        if result["success"]:
            print(f"✅ Module '{module_name}' registered successfully!")
            print(f"  Class: {result['class_name']}")
            if result.get("security_warnings"):
                print("⚠️  Security warnings:")
                for warning in result["security_warnings"]:
                    print(f"  • {warning}")
        else:
            print(f"❌ Registration failed: {result['error']}")
            if "security_issues" in result:
                print("\n🔒 Security issues:")
                for issue in result["security_issues"]:
                    print(f"  • {issue}")
            return 1

        return 0

    def cmd_modules_create(self, args) -> int:
        """Create an external module template"""
        module_name = args.name
        author = args.author
        description = args.description or f"Custom analysis module: {module_name}"

        print(f"🛠️  Creating module template: {module_name}")
        print("=" * 50)

        result = self.module_loader.create_module_template(
            module_name=module_name, author=author, description=description
        )

        if result["success"]:
            print(f"✅ Template created successfully!")
            print(f"📁 Module file: {result['module_file']}")
            print(f"📋 Manifest file: {result['manifest_file']}")
            print(f"\n💡 {result['message']}")
            print("\nNext steps:")
            print(f"1. Edit {result['module_file']} to implement your analysis logic")
            print(
                f"2. Test with: python cli.py modules register {module_name} {result['module_file']}"
            )
        else:
            print(f"❌ Template creation failed: {result['error']}")
            return 1

        return 0

    def cmd_modules_load_all(self, args) -> int:
        """Load all external modules from directory"""
        print("📦 Loading all external modules...")
        print("=" * 50)

        results = self.module_loader.load_all_modules()

        # Report successful loads
        if results["successful"]:
            print(f"✅ Successfully loaded {len(results['successful'])} modules:")
            for result in results["successful"]:
                print(f"  • {result['module_name']} ({result['class_name']})")

        # Report failures
        if results["failed"]:
            print(f"\n❌ Failed to load {len(results['failed'])} modules:")
            for failure in results["failed"]:
                print(f"  • {failure['file']}: {failure['error']}")

        # Report warnings
        if results["warnings"]:
            print(f"\n⚠️  Security warnings:")
            for warning in results["warnings"]:
                print(f"  • {warning}")

        if not results["successful"] and not results["failed"]:
            print("📭 No external modules found to load")

        return 0

    def cmd_modules_list_external(self, args) -> int:
        """List loaded external modules"""
        modules = self.module_loader.list_loaded_modules()

        if not modules:
            print("📦 No external modules loaded")
            print("\nTo create a module template:")
            print("  python cli.py modules create my_module")
            print("\nTo load existing modules:")
            print("  python cli.py modules load-all")
            return 0

        print(f"📦 Loaded External Modules ({len(modules)})")
        print("=" * 50)

        for name, info in modules.items():
            manifest = info["manifest"]
            print(f"• {name}")
            print(f"  Author: {manifest['author']}")
            print(f"  Version: {manifest['version']}")
            print(f"  Description: {manifest['description']}")
            print(f"  File: {info['source_path']}")
            print(f"  Registered: {'✅' if info['registered'] else '❌'}")
            print()

        return 0

    def cmd_profiles_list(self, args) -> int:
        """List available profiles"""
        profiles = self.profile_manager.get_all_profiles()

        if not profiles:
            print("📝 No profiles available")
            return 0

        print(f"📝 Available Analysis Profiles ({len(profiles)})")
        print("=" * 50)

        for name, profile in profiles.items():
            print(f"• {name}")
            print(f"  Description: {profile.description}")
            print(f"  Modules: {', '.join(profile.modules)}")
            print(f"  Parameters: {len(profile.default_parameters)} default(s)")
            print()

        return 0

    def cmd_profiles_show(self, args) -> int:
        """Show detailed profile information"""
        profile_name = args.profile_name

        profile = self.profile_manager.get_profile(profile_name)
        if not profile:
            print(f"❌ Profile '{profile_name}' not found")
            return 1

        print(f"📋 Profile Details: {profile_name}")
        print("=" * 50)
        print(f"Name: {profile.name}")
        print(f"Description: {profile.description}")
        print(f"Modules: {', '.join(profile.modules)}")

        if profile.default_parameters:
            print("\nDefault Parameters:")
            for key, value in profile.default_parameters.items():
                print(f"  {key}: {value}")
        else:
            print("\nDefault Parameters: None")

        # Validate profile
        print("\n🔧 Validation:")
        validation = self.profile_manager.validate_profile(profile_name, self.registry)
        if validation["valid"]:
            print("✅ Profile is valid")
        else:
            print("❌ Profile validation issues:")
            for error in validation.get("errors", []):
                print(f"  • {error}")

            invalid_modules = validation.get("invalid_modules", [])
            if invalid_modules:
                print(f"  • Invalid modules: {', '.join(invalid_modules)}")

        return 0

    def cmd_profiles_create(self, args) -> int:
        """Create a new profile"""
        name = args.name
        description = args.description or f"Custom profile: {name}"
        modules = args.modules

        # Parse parameters if provided
        default_parameters = {}
        if args.params:
            try:
                default_parameters = json.loads(args.params)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in parameters: {e}")
                return 1

        # Validate modules exist
        missing_modules = []
        for module in modules:
            if not self.registry.has_module(module):
                missing_modules.append(module)

        if missing_modules:
            print(f"❌ The following modules are not registered:")
            for module in missing_modules:
                print(f"  • {module}")
            print("\nUse 'python cli.py modules list' to see available modules")
            return 1

        # Create profile
        profile = AnalysisProfile(
            name=name,
            description=description,
            modules=modules,
            default_parameters=default_parameters,
        )

        # Validate profile
        validation = profile.validate()
        if not validation["valid"]:
            print("❌ Profile validation failed:")
            for error in validation["errors"]:
                print(f"  • {error}")
            return 1

        # Add to manager
        self.profile_manager.add_profile(profile)

        print(f"✅ Profile '{name}' created successfully!")
        print(f"  Description: {description}")
        print(f"  Modules: {', '.join(modules)}")
        if default_parameters:
            print(f"  Parameters: {len(default_parameters)} default(s)")

        return 0

    def cmd_profiles_validate(self, args) -> int:
        """Validate a profile"""
        profile_name = args.profile_name

        validation = self.profile_manager.validate_profile(profile_name, self.registry)

        print(f"🔧 Validating Profile: {profile_name}")
        print("=" * 50)

        if validation["valid"]:
            print("✅ Profile is valid and ready to use")
        else:
            print("❌ Profile validation failed:")
            for error in validation.get("errors", []):
                print(f"  • {error}")

            invalid_modules = validation.get("invalid_modules", [])
            if invalid_modules:
                print(f"\n❌ Invalid modules:")
                for module in invalid_modules:
                    print(f"  • {module}")

        return 0

    def cmd_analyze(self, args) -> int:
        """Run bias analysis"""
        run_id = args.run_id

        # Load data (placeholder - would need to implement data loading)
        print(f"📊 Loading data for run: {run_id}")

        # For now, show what would happen
        print("=" * 50)

        if args.profile:
            print(f"📝 Using profile: {args.profile}")
            profile = self.profile_manager.get_profile(args.profile)
            if not profile:
                print(f"❌ Profile '{args.profile}' not found")
                return 1
            print(f"  Modules: {', '.join(profile.modules)}")
        elif args.modules:
            print(f"🧩 Using modules: {', '.join(args.modules)}")
        else:
            print("📝 Using default profile: standard")

        print(f"🔬 Analysis parameters:")
        print(f"  Alpha: {args.alpha}")
        print(f"  Effect size threshold: {args.effect_size_threshold}")

        if args.output:
            print(f"💾 Output file: {args.output}")

        print(f"📋 Output format: {args.format}")

        # TODO: Implement actual analysis
        print("\n🚧 Analysis execution not yet implemented in CLI")
        print("   Use the web interface at http://localhost:5100 for analysis")

        return 0

    def cmd_test(self, args) -> int:
        """Run test suite"""
        suite = args.suite

        print(f"🧪 Running {suite} test suite...")

        # Build test command
        test_cmd = [sys.executable, "tests/run_all_tests.py", suite]
        if args.verbose:
            test_cmd.append("--verbose")

        try:
            result = subprocess.run(test_cmd, check=False)
            return result.returncode
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return 1

    def cmd_status(self, args) -> int:
        """Show system status"""
        print("📊 OpenAudit System Status")
        print("=" * 50)

        # Module registry status
        modules = self.registry.get_available_modules()
        print(f"🧩 Modules: {len(modules)} registered")

        # Profile manager status
        profiles = self.profile_manager.get_all_profiles()
        print(f"📝 Profiles: {len(profiles)} available")

        # Check if web interface is running
        try:
            import requests

            requests.get("http://localhost:5100", timeout=2)
            print("🌐 Web Interface: ✅ Running on http://localhost:5100")
        except Exception:
            print("🌐 Web Interface: ❌ Not running")

        # System information
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"📁 Working Directory: {os.getcwd()}")

        return 0

    def cmd_version(self, args) -> int:
        """Show version information"""
        print("OpenAudit - AI Bias Testing Platform")
        print("Version: 2.0.0-modular")
        print(
            "Features: Modular Analysis System, Enhanced Statistics, Cultural Context"
        )
        print("License: MIT")
        return 0

    def run(self, args=None) -> int:
        """Run the CLI"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return 0

        # Route to appropriate command handler
        command_handlers = {
            "start": self.cmd_start,
            "status": self.cmd_status,
            "version": self.cmd_version,
            "analyze": self.cmd_analyze,
            "test": self.cmd_test,
        }

        # Handle subcommands
        if parsed_args.command == "modules":
            if parsed_args.modules_action == "list":
                return self.cmd_modules_list(parsed_args)
            elif parsed_args.modules_action == "info":
                return self.cmd_modules_info(parsed_args)
            elif parsed_args.modules_action == "test":
                return self.cmd_modules_test(parsed_args)
            elif parsed_args.modules_action == "register":
                return self.cmd_modules_register(parsed_args)
            elif parsed_args.modules_action == "create":
                return self.cmd_modules_create(parsed_args)
            elif parsed_args.modules_action == "load-all":
                return self.cmd_modules_load_all(parsed_args)
            elif parsed_args.modules_action == "list-external":
                return self.cmd_modules_list_external(parsed_args)
            else:
                print(f"❌ Unknown modules command: {parsed_args.modules_action}")
                return 1

        elif parsed_args.command == "profiles":
            if parsed_args.profiles_action == "list":
                return self.cmd_profiles_list(parsed_args)
            elif parsed_args.profiles_action == "show":
                return self.cmd_profiles_show(parsed_args)
            elif parsed_args.profiles_action == "create":
                return self.cmd_profiles_create(parsed_args)
            elif parsed_args.profiles_action == "validate":
                return self.cmd_profiles_validate(parsed_args)
            else:
                print(f"❌ Unknown profiles command: {parsed_args.profiles_action}")
                return 1

        # Handle main commands
        handler = command_handlers.get(parsed_args.command)
        if handler:
            return handler(parsed_args)
        else:
            print(f"❌ Unknown command: {parsed_args.command}")
            return 1


def main():
    """Main entry point"""
    cli = OpenAuditCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
