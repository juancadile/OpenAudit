#!/usr/bin/env python3
"""
OpenAudit Startup Script
Launches the complete unified interface with both historical analysis and live experiments
"""

import os
import subprocess
import sys


def main():
    print("🚀 Starting OpenAudit - Complete Bias Testing Platform")
    print("=" * 60)
    print("📊 Historical Analysis: ✓ Available")
    print("🔬 Live Experiments: ✓ Available")
    print("🌐 Dashboard: http://localhost:5100")
    print("=" * 60)
    print("💡 Use Ctrl+C to stop the server")
    print()

    try:
        # Run the unified interface
        subprocess.run([sys.executable, "unified_interface.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 OpenAudit stopped successfully!")
    except Exception as e:
        print(f"❌ Error starting OpenAudit: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
