#!/usr/bin/env python3
"""
Model Comparison and Recommendation Tool
=======================================

Compare different model architectures and get recommendations for your use case.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from prettytable import PrettyTable

# Add src to path
sys.path.append("src")

try:
    from models.digit_classifier import create_model
    from models.advanced_digit_classifier import get_model_info
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you're in the project root directory")
    sys.exit(1)


def benchmark_model(model, device, input_shape, num_tests=100):
    """Benchmark model inference speed."""
    model.eval()
    model = model.to(device)

    # Warmup
    dummy_input = torch.randn(1, *input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_tests):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        "avg_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
    }


def compare_models():
    """Compare all available models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    print(f"🧪 Comparing models on device: {device}")
    print("=" * 70)

    # Model types to compare
    model_types = ["lightweight", "mini", "mlp", "advanced", "efficient", "transformer"]

    # Feature dimensions (typical for MFCC)
    input_channels, input_length = 13, 32
    input_shape = (input_channels, input_length)

    # Results table
    table = PrettyTable()
    table.field_names = [
        "Model",
        "Parameters",
        "Size (MB)",
        "Speed (ms)",
        "Memory",
        "Complexity",
        "Accuracy*",
    ]

    results = []

    for model_type in model_types:
        print(f"\n🧠 Testing {model_type} model...")

        try:
            # Create model
            model = create_model(
                model_type=model_type,
                input_channels=input_channels,
                input_length=input_length,
                num_classes=10,
            )

            # Get model info
            info = get_model_info(model)

            # Benchmark speed
            speed_info = benchmark_model(model, device, input_shape)

            # Estimate memory usage (rough)
            param_memory = info["model_size_mb"]
            activation_memory = np.prod(input_shape) * 4 / 1024 / 1024  # Rough estimate
            total_memory = param_memory + activation_memory

            # Complexity rating (subjective)
            complexity_map = {
                "mlp": "⭐",
                "mini": "⭐⭐",
                "lightweight": "⭐⭐⭐",
                "efficient": "⭐⭐⭐⭐",
                "advanced": "⭐⭐⭐⭐⭐",
                "transformer": "⭐⭐⭐⭐⭐",
                "ensemble": "⭐⭐⭐⭐⭐⭐",
            }

            # Expected accuracy (rough estimates based on complexity)
            accuracy_map = {
                "mlp": "~75%",
                "mini": "~80%",
                "lightweight": "~85%",
                "efficient": "~90%",
                "advanced": "~92%",
                "transformer": "~88%",
                "ensemble": "~94%",
            }

            # Add to table
            table.add_row(
                [
                    model_type.title(),
                    f"{info['total_params']:,}",
                    f"{info['model_size_mb']:.1f}",
                    f"{speed_info['avg_time_ms']:.2f}",
                    f"{total_memory:.1f} MB",
                    complexity_map.get(model_type, "⭐⭐⭐"),
                    accuracy_map.get(model_type, "~85%"),
                ]
            )

            results.append(
                {
                    "model_type": model_type,
                    "info": info,
                    "speed": speed_info,
                    "memory": total_memory,
                }
            )

            print(
                f"✅ {model_type}: {info['total_params']:,} params, "
                f"{speed_info['avg_time_ms']:.2f}ms"
            )

        except Exception as e:
            print(f"❌ Failed to test {model_type}: {e}")
            table.add_row(
                [model_type.title(), "Error", "Error", "Error", "Error", "❌", "Error"]
            )

    print(f"\n📊 Model Comparison Results:")
    print(table)
    print("\n* Accuracy estimates based on model complexity and typical performance")

    return results


def get_recommendation(use_case="balanced"):
    """Get model recommendation based on use case."""
    print(f"\n🎯 Recommendation for '{use_case}' use case:")
    print("=" * 50)

    recommendations = {
        "speed": {
            "model": "mini",
            "reason": "Fastest inference with reasonable accuracy",
            "pros": ["Very fast", "Small memory footprint", "Good for real-time"],
            "cons": ["Lower accuracy", "Simple architecture"],
        },
        "accuracy": {
            "model": "advanced",
            "reason": "Best single-model accuracy with ResNet + attention",
            "pros": ["High accuracy", "Robust features", "Good generalization"],
            "cons": ["Slower inference", "More memory", "Complex"],
        },
        "balanced": {
            "model": "efficient",
            "reason": "Good balance of speed and accuracy",
            "pros": ["Fast inference", "High accuracy", "Moderate size"],
            "cons": ["More complex than basic models"],
        },
        "research": {
            "model": "transformer",
            "reason": "Cutting-edge architecture for temporal patterns",
            "pros": ["State-of-the-art", "Attention mechanism", "Interpretable"],
            "cons": ["Slower", "More memory", "Complex training"],
        },
        "production": {
            "model": "efficient",
            "reason": "Reliable performance with good efficiency",
            "pros": ["Stable", "Fast enough", "Good accuracy", "Battle-tested"],
            "cons": ["Not the absolute fastest or most accurate"],
        },
        "ensemble": {
            "model": "ensemble",
            "reason": "Highest possible accuracy from multiple models",
            "pros": ["Highest accuracy", "Robust predictions", "Error reduction"],
            "cons": ["Slowest", "Most memory", "Complex deployment"],
        },
    }

    rec = recommendations.get(use_case, recommendations["balanced"])

    print(f"🏆 Recommended Model: {rec['model'].upper()}")
    print(f"📋 Reason: {rec['reason']}")
    print(f"\n✅ Pros:")
    for pro in rec["pros"]:
        print(f"   • {pro}")
    print(f"\n⚠️  Cons:")
    for con in rec["cons"]:
        print(f"   • {con}")

    return rec["model"]


def training_recommendations():
    """Provide training recommendations for better performance."""
    print(f"\n🚀 Training Tips for Better Performance:")
    print("=" * 50)

    tips = [
        "🎯 **Use Advanced Training Script**: `python scripts/train_advanced.py --model_type advanced`",
        "📊 **Try Different Features**: MFCC (default), Mel-spectrogram, or raw spectrogram",
        "⏰ **Train Longer**: Use 100+ epochs with early stopping for best results",
        "🔧 **Hyperparameter Tuning**: Try different learning rates (0.001, 0.0003, 0.0001)",
        "📈 **Use Cosine Scheduling**: `--scheduler cosine` for better convergence",
        "🎨 **Label Smoothing**: `--label_smoothing 0.1` for better generalization",
        "💪 **Mixed Precision**: `--mixed_precision` for faster training (if GPU available)",
        "🧠 **Model Ensemble**: Combine multiple models for highest accuracy",
        "📊 **Data Augmentation**: More data always helps (consider noise, speed variations)",
        "⚡ **Advanced Optimizers**: Use AdamW for better weight decay handling",
    ]

    for tip in tips:
        print(f"   {tip}")

    print(f"\n🏆 **Quick Start Commands:**")
    print(f"   # Best accuracy (advanced model)")
    print(f"   python scripts/train_advanced.py --model_type advanced --epochs 100")
    print(f"   ")
    print(f"   # Balanced performance")
    print(f"   python scripts/train_advanced.py --model_type efficient --epochs 80")
    print(f"   ")
    print(f"   # Fast training")
    print(f"   python scripts/train_advanced.py --model_type lightweight --epochs 50")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Model Comparison Tool")
    parser.add_argument(
        "--use_case",
        default="balanced",
        choices=["speed", "accuracy", "balanced", "research", "production", "ensemble"],
        help="Your use case for model selection",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run full model comparison"
    )
    parser.add_argument(
        "--recommend_only", action="store_true", help="Only show recommendations"
    )

    args = parser.parse_args()

    print("🎤 Digit Classifier - Model Selection Tool")
    print("=" * 50)

    if args.recommend_only:
        # Just show recommendations
        recommended_model = get_recommendation(args.use_case)
        training_recommendations()
    elif args.compare:
        # Full comparison
        results = compare_models()
        recommended_model = get_recommendation(args.use_case)
        training_recommendations()
    else:
        # Quick recommendation
        recommended_model = get_recommendation(args.use_case)
        print(f"\n💡 To see full comparison, run: python {__file__} --compare")
        training_recommendations()

    print(f"\n🎉 **Ready to train your {recommended_model} model!**")
    print(
        f"   Next step: python scripts/train_advanced.py --model_type {recommended_model}"
    )


if __name__ == "__main__":
    main()
