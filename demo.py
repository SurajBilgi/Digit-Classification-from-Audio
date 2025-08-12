#!/usr/bin/env python3
"""
Digit Classification Demo
=======================

A comprehensive demo showcasing the complete digit classification system:
1. Dataset loading and exploration (with manual dataset support)
2. Model training and evaluation
3. Inference testing
4. Real-time microphone integration

Run this script to experience the full pipeline!
"""

import os
import sys
import time
import argparse

# Add src to path for imports
sys.path.append("src")


def demo_data_loading():
    """Demonstrate dataset loading and exploration with manual fallback."""
    print("🔄 Loading Free Spoken Digit Dataset...")

    try:
        # Try the main dataset loader first (with built-in fallbacks)
        from data.dataset import load_fsdd_dataset, create_data_loaders

        # Load dataset
        audio_data, labels = load_fsdd_dataset()
        train_loader, test_loader = create_data_loaders(audio_data, labels)

        print("✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(audio_data)}")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")

        # Show sample audio info
        sample_audio = audio_data[0]
        print(f"   Sample audio shape: {sample_audio.shape}")
        print("   Sample rate: 8000 Hz")
        print(f"   Duration: {len(sample_audio) / 8000:.1f} seconds")

        # Show label distribution
        import numpy as np

        unique, counts = np.unique(labels, return_counts=True)
        print(f"   Label distribution: {dict(zip(unique, counts))}")

        return True

    except Exception as e:
        print(f"❌ Error with main dataset loader: {e}")

        # Try manual dataset as fallback
        print("\n🔄 Trying manual dataset fallback...")
        try:
            from data.manual_loader import load_manual_fsdd
            from data.dataset import create_data_loaders

            # Load manual dataset
            audio_data, labels = load_manual_fsdd()
            train_loader, test_loader = create_data_loaders(audio_data, labels)

            print("✅ Manual dataset loaded successfully!")
            print(f"   Total samples: {len(audio_data)}")
            print(f"   Training samples: {len(train_loader.dataset)}")
            print(f"   Test samples: {len(test_loader.dataset)}")

            return True

        except Exception as manual_error:
            print(f"❌ Manual dataset also failed: {manual_error}")
            print("\n💡 Solutions:")
            print("   1. Run: python manual_dataset_download.py")
            print("   2. Check environment: python verify_setup.py")
            print("   3. Follow SETUP_GUIDE.md for environment setup")
            return False


def demo_feature_extraction():
    """Demonstrate audio feature extraction."""
    print("\n🎵 Testing Audio Feature Extraction...")

    try:
        import numpy as np
        from features.audio_features import AudioFeatureExtractor

        # Create feature extractor
        extractor = AudioFeatureExtractor()

        # Generate dummy audio (simulating 1 second at 8kHz)
        dummy_audio = np.random.randn(8000)

        # Test different feature types
        feature_types = ["mfcc", "mel", "spectrogram"]

        for feature_type in feature_types:
            features = extractor.extract_features(dummy_audio, feature_type)
            print(f"   {feature_type.upper()}: {features.shape}")

        print("✅ Feature extraction working correctly!")
        return True

    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return False


def demo_model_architectures():
    """Demonstrate different model architectures."""
    print("\n🧠 Testing Model Architectures...")

    try:
        import torch
        from models.digit_classifier import create_model, count_parameters

        model_types = ["lightweight", "mini", "mlp"]

        for model_type in model_types:
            model = create_model(
                model_type, input_channels=13, input_length=32, num_classes=10
            )

            # Count parameters
            param_info = count_parameters(model)

            print(f"   {model_type.capitalize()} CNN:")
            print(f"     Parameters: {param_info['total_parameters']:,}")
            print(f"     Size: {param_info['size_mb']:.2f} MB")

            # Test forward pass
            test_input = torch.randn(4, 13, 32)  # Batch of 4
            output = model(test_input)
            print(f"     Output shape: {output.shape}")

        print("✅ All model architectures working correctly!")
        return True

    except Exception as e:
        print(f"❌ Error testing models: {e}")
        return False


def load_dataset_for_training():
    """Load dataset with fallback options for training."""
    try:
        # Try main loader first
        from data.dataset import load_fsdd_dataset

        return load_fsdd_dataset()
    except Exception:
        # Fallback to manual dataset
        try:
            from data.manual_loader import load_manual_fsdd

            return load_manual_fsdd()
        except Exception as e:
            raise Exception(f"Could not load any dataset: {e}")


def demo_training(quick_mode=True):
    """Demonstrate model training process with dataset fallback."""
    mode_text = "Quick" if quick_mode else "Full"
    print(f"\n🏋️ {mode_text} Training Demo...")

    if quick_mode:
        print("   (Running abbreviated training for demo purposes)")

    try:
        # Import training components
        from data.dataset import create_data_loaders
        from features.audio_features import AudioFeatureExtractor, pad_features
        from models.digit_classifier import create_model
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Load data with fallback
        audio_data, labels = load_dataset_for_training()
        train_loader, test_loader = create_data_loaders(
            audio_data, labels, batch_size=16
        )

        # Extract features for a small subset
        feature_extractor = AudioFeatureExtractor()

        print("   Extracting features...")
        sample_features = []
        sample_labels = []

        # Use only first batch for demo
        for batch_audio, batch_labels in train_loader:
            for i, audio in enumerate(batch_audio[:8]):  # Just 8 samples
                features = feature_extractor.extract_features(audio, "mfcc")
                features = feature_extractor.normalize_features(features)
                features = pad_features(features, 32)
                sample_features.append(features)
                sample_labels.append(batch_labels[i])

            break

        sample_features = torch.stack(sample_features)
        sample_labels = torch.stack(sample_labels)

        print(f"   Training on {len(sample_features)} samples...")

        # Create model
        model = create_model("mini")  # Use mini for faster training

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Quick training loop
        epochs = 3 if quick_mode else 10

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(sample_features)
            loss = criterion(outputs, sample_labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(sample_labels).sum().item()
            accuracy = accuracy / len(sample_labels)

            print(
                f"     Epoch {epoch+1}/{epochs}: "
                f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.3f}"
            )

        print("✅ Training demo completed successfully!")

        # Save demo model
        os.makedirs("models", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": "mini",
                "feature_type": "mfcc",
                "input_channels": 13,
                "input_length": 32,
            },
            "models/demo_model.pth",
        )

        print("   Demo model saved to models/demo_model.pth")
        return True

    except Exception as e:
        print(f"❌ Error in training demo: {e}")
        print("\n💡 Possible solutions:")
        print("   1. Check if dataset is loaded: python manual_dataset_download.py")
        print("   2. Verify environment: python verify_setup.py")
        return False


def demo_inference():
    """Demonstrate inference pipeline."""
    print("\n⚡ Testing Inference Pipeline...")

    if not os.path.exists("models/demo_model.pth"):
        print("   ⚠️  Demo model not found. Running training first...")
        if not demo_training(quick_mode=True):
            print("   ❌ Could not create demo model for inference")
            return False

    try:
        from inference.predictor import DigitPredictor
        import numpy as np

        # Load predictor
        predictor = DigitPredictor("models/demo_model.pth")

        print("   Model loaded successfully!")

        # Test with random audio
        print("   Testing inference speed...")

        # Multiple predictions for timing
        times = []
        for i in range(10):
            test_audio = np.random.randn(8000)
            start_time = time.time()
            digit, confidence = predictor.predict(test_audio)
            inference_time = time.time() - start_time
            times.append(inference_time)

            if i == 0:  # Show first prediction
                print(
                    f"     Sample prediction: Digit {digit} "
                    f"(confidence: {confidence:.3f})"
                )

        # Show performance stats
        avg_time = np.mean(times) * 1000  # Convert to ms
        print(f"   Average inference time: {avg_time:.1f} ms")

        # Get predictor stats
        stats = predictor.get_performance_stats()
        pps = stats["predictions_per_second"]
        print(f"   Predictions per second: {pps:.1f}")

        print("✅ Inference pipeline working correctly!")
        return True

    except Exception as e:
        print(f"❌ Error in inference demo: {e}")
        return False


def demo_microphone():
    """Demonstrate microphone integration."""
    print("\n🎤 Microphone Integration Demo...")

    try:
        from microphone.live_predictor import LiveDigitPredictor

        if not os.path.exists("models/demo_model.pth"):
            print("   ⚠️  Demo model not found. Skipping microphone demo.")
            return False

        print("   Creating live predictor...")

        # Create predictor (but don't actually record for demo)
        with LiveDigitPredictor("models/demo_model.pth") as predictor:
            print("   ✅ Live predictor initialized successfully!")

            # Show available audio devices
            print("   Available audio devices:")
            devices = predictor.list_audio_devices()

            if devices:
                for device in devices[:3]:  # Show first 3
                    name = device["name"]
                    idx = device["index"]
                    print(f"     {idx}: {name}")
                if len(devices) > 3:
                    more_count = len(devices) - 3
                    print(f"     ... and {more_count} more")
            else:
                print("     No audio input devices found")

            print("\n   To test microphone integration, run:")
            mic_cmd = (
                "python src/microphone/live_predictor.py "
                "--model_path models/demo_model.pth --interactive"
            )
            print(f"   {mic_cmd}")

        return True

    except Exception as e:
        print(f"❌ Error in microphone demo: {e}")
        print("   Note: This might be due to missing PyAudio or audio drivers")
        print("   Install PyAudio: conda install -c anaconda pyaudio")
        return False


def show_setup_status():
    """Show the current setup status and provide guidance."""
    print("\n🔍 Checking Setup Status...")

    # Check if manual dataset exists
    manual_dataset_path = "data/fsdd_manual/recordings"
    if os.path.exists(manual_dataset_path):
        import glob

        wav_count = len(glob.glob(f"{manual_dataset_path}/*.wav"))
        print(f"   ✅ Manual dataset found: {wav_count} files")
    else:
        print("   ❌ Manual dataset not found")
        print("      Run: python manual_dataset_download.py")

    # Check if demo model exists
    if os.path.exists("models/demo_model.pth"):
        print("   ✅ Demo model found")
    else:
        print("   ⚠️  Demo model not found (will be created during training)")

    # Check verification script
    if os.path.exists("verify_setup.py"):
        print("   ✅ Setup verification available")
        print("      Run: python verify_setup.py")

    print()


def main():
    """Run the complete demo."""
    parser = argparse.ArgumentParser(description="Digit Classification Demo")
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip the training demonstration"
    )
    parser.add_argument(
        "--skip-microphone",
        action="store_true",
        help="Skip the microphone demonstration",
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Show setup status and guidance",
    )

    args = parser.parse_args()

    print("🎯 Digit Classification from Audio - Complete Demo")
    print("=" * 55)

    # Show setup status if requested or if it looks like first run
    if args.check_setup or not os.path.exists("data/fsdd_manual/recordings"):
        show_setup_status()
        if args.check_setup:
            return

    demos = [
        ("Data Loading", demo_data_loading),
        ("Feature Extraction", demo_feature_extraction),
        ("Model Architectures", demo_model_architectures),
    ]

    if not args.skip_training:
        demos.append(("Training", lambda: demo_training(quick_mode=True)))

    demos.append(("Inference", demo_inference))

    if not args.skip_microphone:
        demos.append(("Microphone", demo_microphone))

    # Run all demos
    results = []
    start_time = time.time()

    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, success))
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    total_time = time.time() - start_time
    print(f"\n📋 Demo Summary ({total_time:.1f}s total):")
    print("=" * 35)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nOverall: {successful}/{total} components working")

    if successful == total:
        print("\n🎉 All systems operational! Ready for digit classification!")
        print("\nNext steps:")
        print("   1. Train a full model: python scripts/train.py")
        cmd = (
            "   2. Test with microphone: "
            "python src/microphone/live_predictor.py"
            " --model_path models/best_model.pth --interactive"
        )
        print(cmd)
    else:
        print("\n⚠️  Some components had issues. Check error messages above.")
        print("\n💡 Troubleshooting:")
        print("   1. Verify environment: python verify_setup.py")
        print("   2. Download dataset: python manual_dataset_download.py")
        print("   3. Check setup guide: SETUP_GUIDE.md")
        print("   4. Manual dataset guide: MANUAL_DATASET_GUIDE.md")


if __name__ == "__main__":
    main()
