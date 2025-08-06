#!/usr/bin/env python3
"""
OpenAudit Startup Script (Legacy Compatibility)
Launches the complete unified interface with both historical analysis and live experiments

Note: This script now redirects to the new CLI interface.
Use 'python cli.py start' for full functionality.
"""

import subprocess
import sys


def main():
    print("🔄 Redirecting to OpenAudit CLI...")
    print("💡 For full CLI functionality, use: python cli.py --help")
    print()

    try:
        # Redirect to new CLI start command
        subprocess.run([sys.executable, "cli.py", "start"], check=True)
    except KeyboardInterrupt:
        print("\n👋 OpenAudit stopped successfully!")
    except Exception as e:
        print(f"❌ Error starting OpenAudit: {e}")
        print("💡 Try running: python cli.py start")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
