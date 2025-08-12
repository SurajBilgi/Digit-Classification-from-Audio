#!/usr/bin/env python3
"""
Test script to verify all components are working after fixes.
"""

import sys
import os

# Add src to path
sys.path.append("src")


def test_manual_dataset():
    """Test manual dataset loading."""
    print("🧪 Testing manual dataset loading...")

    try:
        from data.manual_loader import load_manual_fsdd

        audio_data, labels = load_manual_fsdd()
        print(f"✅ Manual dataset: {len(audio_data)} files loaded")
        return True
    except Exception as e:
        print(f"❌ Manual dataset failed: {e}")
        return False


def test_dataset_with_fallback():
    """Test main dataset loader with fallback."""
    print("\n🧪 Testing main dataset loader with fallback...")

    try:
        from data.dataset import load_fsdd_dataset

        audio_data, labels = load_fsdd_dataset()
        print(f"✅ Dataset loader: {len(audio_data)} files loaded")
        return True
    except Exception as e:
        print(f"❌ Dataset loader failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\n🧪 Testing feature extraction...")

    try:
        from features.audio_features import AudioFeatureExtractor
        import numpy as np

        extractor = AudioFeatureExtractor()
        dummy_audio = np.random.randn(8000)
        features = extractor.extract_features(dummy_audio, "mfcc")
        print(f"✅ Feature extraction: {features.shape}")
        return True
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\n🧪 Testing model creation...")

    try:
        from models.digit_classifier import create_model

        model = create_model("mini", input_channels=13, input_length=32)
        print("✅ Model creation successful")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_inference_imports():
    """Test inference module imports."""
    print("\n🧪 Testing inference imports...")

    try:
        from inference.predictor import DigitPredictor

        print("✅ Inference imports successful")
        return True
    except Exception as e:
        print(f"❌ Inference imports failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🎯 Testing Fixed Components")
    print("=" * 40)

    tests = [
        ("Manual Dataset", test_manual_dataset),
        ("Dataset with Fallback", test_dataset_with_fallback),
        ("Feature Extraction", test_feature_extraction),
        ("Model Creation", test_model_creation),
        ("Inference Imports", test_inference_imports),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n📋 Test Results:")
    print("=" * 20)

    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All fixes working! Ready to continue training!")
        print("\nNext steps:")
        print("1. Activate your environment: conda activate audioLLM")
        print("2. Resume training: python scripts/train.py --epochs 3 --batch_size 16")
    else:
        print("\n⚠️  Some components still need attention.")
        print("Check error messages above for specific issues.")


if __name__ == "__main__":
    main()
