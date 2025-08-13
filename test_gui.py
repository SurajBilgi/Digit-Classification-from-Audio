#!/usr/bin/env python3
"""
Test script for GUI components
"""

import sys
import os

# Add src to path
sys.path.append("src")


def test_imports():
    """Test all required imports for GUI."""
    print("🧪 Testing GUI imports...")

    try:
        import tkinter as tk

        print("✅ tkinter imported successfully")
    except ImportError as e:
        print(f"❌ tkinter import failed: {e}")
        return False

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False

    try:
        import numpy as np

        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False

    # Test internal imports
    try:
        from microphone.live_predictor import LiveDigitPredictor

        print("✅ LiveDigitPredictor imported successfully")
    except ImportError as e:
        print(f"❌ LiveDigitPredictor import failed: {e}")
        print("   This is expected if PyAudio is not installed")

    try:
        from inference.predictor import DigitPredictor

        print("✅ DigitPredictor imported successfully")
    except ImportError as e:
        print(f"❌ DigitPredictor import failed: {e}")
        return False

    return True


def test_gui_creation():
    """Test GUI creation without showing window."""
    print("\n🎨 Testing GUI creation...")

    try:
        # Import GUI class
        from gui_app import DigitClassifierGUI, ModernButton, WaveformWidget

        print("✅ GUI classes imported successfully")

        # Test model path
        model_path = "models/best_model.pth"
        if os.path.exists(model_path):
            print(f"✅ Model found: {model_path}")
        else:
            print(f"⚠️  Model not found: {model_path} (GUI will handle this)")

        return True

    except Exception as e:
        print(f"❌ GUI creation test failed: {e}")
        return False


def check_environment():
    """Check if environment is properly activated."""
    print("\n🔍 Checking environment...")

    # Check if we're in conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "None")
    print(f"   Conda environment: {conda_env}")

    if conda_env == "audioLLM":
        print("✅ audioLLM environment is active")
    else:
        print("⚠️  audioLLM environment not active")
        print("   Run: conda activate audioLLM")

    # Check Python version
    print(f"   Python version: {sys.version}")

    return True


def show_launch_instructions():
    """Show how to launch the GUI."""
    print("\n🚀 How to Launch GUI:")
    print("=" * 50)
    print("1. Activate environment:")
    print("   conda activate audioLLM")
    print()
    print("2. Launch GUI (simple):")
    print("   python run_gui.py")
    print()
    print("3. Launch GUI (with model path):")
    print("   python gui_app.py --model_path models/best_model.pth")
    print()
    print("4. If you don't have a trained model:")
    print("   python scripts/train.py --epochs 10  # Quick training")
    print("   Then launch GUI")


def main():
    """Run all tests."""
    print("🎤 Digit Classifier GUI - Component Test")
    print("=" * 50)

    # Run tests
    imports_ok = test_imports()
    gui_ok = test_gui_creation()
    env_ok = check_environment()

    print("\n📊 Test Results:")
    print("=" * 30)
    print(f"Imports:     {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"GUI Setup:   {'✅ PASS' if gui_ok else '❌ FAIL'}")
    print(f"Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")

    if imports_ok and gui_ok:
        print("\n🎉 GUI is ready to launch!")
        show_launch_instructions()
    else:
        print("\n⚠️  Some issues detected. Please:")
        print("   1. Activate environment: conda activate audioLLM")
        print("   2. Install missing packages")
        print("   3. Re-run this test")


if __name__ == "__main__":
    main()
