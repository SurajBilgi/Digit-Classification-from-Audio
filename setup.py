#!/usr/bin/env python3
"""
Setup script for Digit Classification from Audio
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def run_demo():
    """Run the demo script."""
    print("\nRunning demo...")
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running demo: {e}")
        return False


def main():
    """Main setup function."""
    print("🎯 Digit Classification from Audio - Setup")
    print("=" * 45)

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return

    print(f"✅ Python {sys.version.split()[0]} detected")

    # Install requirements
    if not install_requirements():
        return

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    print("✅ Project directories created")

    # Run demo
    print("\n" + "=" * 45)
    print("Running system demo...")
    run_demo()


if __name__ == "__main__":
    main()
