#!/usr/bin/env python3
"""
Setup Verification Script for Digit Classification from Audio
============================================================

This script helps diagnose installation issues and provides specific solutions.
Run this after setting up your environment to verify everything works.
"""

import sys
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and 8 <= version.minor <= 11:
        print("   ✅ Python version compatible")
        return True
    else:
        print("   ❌ Python version incompatible")
        print("   💡 Recommended: Python 3.8-3.11")
        return False


def check_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking package installations...")

    packages = {
        "torch": "PyTorch",
        "torchaudio": "TorchAudio",
        "librosa": "Librosa",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "sklearn": "Scikit-learn",
        "datasets": "Hugging Face Datasets",
        "soundfile": "SoundFile",
    }

    results = {}

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
            results[package] = True
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            results[package] = False

    # Check PyAudio separately (optional)
    try:
        import pyaudio

        print("   ✅ PyAudio (microphone support available)")
        results["pyaudio"] = True
    except ImportError:
        print("   ⚠️  PyAudio not installed (microphone features disabled)")
        results["pyaudio"] = False

    return results


def check_dataset_access():
    """Check if dataset can be loaded."""
    print("\n🗂️  Checking dataset access...")

    try:
        from datasets import load_dataset

        print("   Testing MTEB dataset...")

        # Try loading just a small sample first
        dataset = load_dataset("mteb/free-spoken-digit-dataset", split="train[:5]")
        print("   ✅ MTEB dataset accessible")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ MTEB dataset failed: {error_msg[:100]}...")

        # Check for specific errors
        if "torchcodec" in error_msg.lower() or "ffmpeg" in error_msg.lower():
            print("   💡 TorchCodec/FFmpeg issue detected")
            print("   🔧 Solutions:")
            print("      - Install FFmpeg: brew install ffmpeg (macOS)")
            print("      - Use stable PyTorch: conda install pytorch=1.12.0")
            print("      - Try alternative environment with Python 3.8")

        # Try fallback dataset
        try:
            print("   Testing fallback dataset...")
            dataset = load_dataset(
                "Matthijs/free-spoken-digit-dataset", split="train[:5]"
            )
            print("   ✅ Fallback dataset accessible")
            return True
        except Exception as fallback_error:
            print(f"   ❌ Fallback also failed: {str(fallback_error)[:100]}...")
            return False


def check_audio_devices():
    """Check available audio devices."""
    print("\n🎤 Checking audio devices...")

    try:
        import pyaudio

        p = pyaudio.PyAudio()
        device_count = p.get_device_count()

        print(f"   Found {device_count} audio devices:")

        input_devices = 0
        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                input_devices += 1
                if input_devices <= 3:  # Show first 3 input devices
                    print(f"     {i}: {info['name']} ({info['maxInputChannels']} ch)")

        if input_devices > 3:
            print(f"     ... and {input_devices - 3} more input devices")

        p.terminate()

        if input_devices > 0:
            print("   ✅ Microphone support available")
            return True
        else:
            print("   ⚠️  No input devices found")
            return False

    except ImportError:
        print("   ⚠️  PyAudio not installed - microphone features disabled")
        return False
    except Exception as e:
        print(f"   ❌ Audio check failed: {e}")
        return False


def test_core_functionality():
    """Test core project functionality."""
    print("\n⚙️  Testing core functionality...")

    try:
        sys.path.append("src")

        # Test feature extraction
        print("   Testing feature extraction...")
        from features.audio_features import AudioFeatureExtractor
        import numpy as np

        extractor = AudioFeatureExtractor()
        dummy_audio = np.random.randn(8000)
        features = extractor.extract_features(dummy_audio, "mfcc")
        print(f"   ✅ Feature extraction: {features.shape}")

        # Test model creation
        print("   Testing model creation...")
        from models.digit_classifier import create_model

        model = create_model("mini")
        print("   ✅ Model creation successful")

        return True

    except Exception as e:
        print(f"   ❌ Core functionality test failed: {e}")
        return False


def provide_recommendations(results):
    """Provide specific recommendations based on test results."""
    print("\n💡 Recommendations:")
    print("=" * 50)

    if not results.get("python_ok", False):
        print("🔧 Python Version Issue:")
        print("   conda create -n digit-audio python=3.9 -y")
        print("   conda activate digit-audio")
        print()

    missing_packages = []
    if results.get("packages"):
        for pkg, status in results["packages"].items():
            if not status and pkg != "pyaudio":  # PyAudio is optional
                missing_packages.append(pkg)

    if missing_packages:
        print("🔧 Missing Packages:")
        print("   # Install via conda (recommended)")
        if "torch" in missing_packages or "torchaudio" in missing_packages:
            print("   conda install pytorch torchaudio cpuonly -c pytorch -y")
        if "librosa" in missing_packages:
            print("   conda install -c conda-forge librosa -y")
        if any(
            pkg in missing_packages
            for pkg in ["numpy", "pandas", "matplotlib", "sklearn"]
        ):
            print("   conda install numpy pandas matplotlib seaborn scikit-learn -y")
        if any(pkg in missing_packages for pkg in ["datasets", "soundfile"]):
            print("   pip install datasets soundfile tqdm jupyter")
        print()

    if not results.get("dataset_ok", False):
        print("🔧 Dataset Issues:")
        print("   # Try stable environment")
        print("   conda create -n digit-audio python=3.8 -y")
        print("   conda activate digit-audio")
        print("   conda install pytorch=1.12.0 torchaudio=0.12.0 cpuonly -c pytorch -y")
        print("   # ... install other packages")
        print("   # OR install FFmpeg: brew install ffmpeg")
        print()

    if not results.get("audio_ok", False) and results.get("packages", {}).get(
        "pyaudio", False
    ):
        print("🔧 Audio Device Issues:")
        print("   # Check system audio settings")
        print("   # Try: python demo.py --skip-microphone")
        print()

    if not results.get("core_ok", False):
        print("🔧 Core Functionality Issues:")
        print("   # Ensure you're in the project directory")
        print("   # Try: python demo.py --skip-training --skip-microphone")
        print()

    # Success recommendations
    if all(
        [results.get("python_ok"), results.get("dataset_ok"), results.get("core_ok")]
    ):
        print("🎉 System looks good! Next steps:")
        print("   1. python demo.py                    # Full demo")
        print("   2. python scripts/train.py --epochs 5   # Quick training")
        print(
            "   3. python src/microphone/live_predictor.py --model_path models/best_model.pth --interactive"
        )


def main():
    """Run all verification checks."""
    print("🎯 Digit Classification Setup Verification")
    print("=" * 50)

    results = {}

    # Run all checks
    results["python_ok"] = check_python_version()
    results["packages"] = check_packages()
    results["dataset_ok"] = check_dataset_access()
    results["audio_ok"] = check_audio_devices()
    results["core_ok"] = test_core_functionality()

    # Summary
    print("\n📋 Summary:")
    print("=" * 20)

    checks = [
        ("Python Version", results["python_ok"]),
        (
            "Required Packages",
            all(
                results["packages"].get(pkg, False)
                for pkg in ["torch", "librosa", "datasets", "numpy"]
            ),
        ),
        ("Dataset Access", results["dataset_ok"]),
        ("Audio Support", results["audio_ok"]),
        ("Core Functions", results["core_ok"]),
    ]

    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check_name}")

    # Provide recommendations
    provide_recommendations(results)


if __name__ == "__main__":
    main()
