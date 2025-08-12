"""
Training script for digit classification from audio.
Supports multiple model architectures and feature types.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.dataset import load_fsdd_dataset, FSSDDataset
from features.audio_features import AudioFeatureExtractor, pad_features
from models.digit_classifier import create_model, count_parameters


def extract_features_from_loader(
    data_loader: DataLoader,
    feature_extractor: AudioFeatureExtractor,
    feature_type: str = "mfcc",
    max_length: int = 32,
) -> tuple:
    """Extract features from a data loader."""
    all_features = []
    all_labels = []

    print(f"Extracting {feature_type} features...")
    for batch_audio, batch_labels in data_loader:
        batch_features = []
        for audio in batch_audio:
            features = feature_extractor.extract_features(audio, feature_type)
            features = feature_extractor.normalize_features(features)
            features = pad_features(features, max_length)
            batch_features.append(features)

        batch_features = torch.stack(batch_features)
        all_features.append(batch_features)
        all_labels.append(batch_labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


def train_model(
    model: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    args,
) -> dict:
    """Train the model and return training history."""

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    best_val_acc = 0.0
    best_model_state = None

    print(f"Training on device: {device}")
    print(f"Model parameters: {count_parameters(model)}")

    for epoch in range(args.epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0

        # Create batches manually for training
        dataset_size = len(train_features)
        batch_size = args.batch_size

        for i in range(0, dataset_size, batch_size):
            end_idx = min(i + batch_size, dataset_size)
            batch_features = train_features[i:end_idx].to(device)
            batch_labels = train_labels[i:end_idx].to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            val_dataset_size = len(val_features)
            for i in range(0, val_dataset_size, batch_size):
                end_idx = min(i + batch_size, val_dataset_size)
                batch_features = val_features[i:end_idx].to(device)
                batch_labels = val_labels[i:end_idx].to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_labels).sum().item()

        # Calculate metrics
        train_acc = 100.0 * train_correct / len(train_features)
        val_acc = 100.0 * val_correct / len(val_features)
        epoch_time = time.time() - start_time

        # Update history
        history["train_loss"].append(train_loss / (dataset_size // batch_size))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / (val_dataset_size // batch_size))
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Update learning rate
        scheduler.step(val_acc)

        # Print progress
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}:")
            print(
                f"  Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Train Acc: {train_acc:.2f}%"
            )
            print(
                f"  Val Loss: {history['val_loss'][-1]:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )
            print(f"  Time: {epoch_time:.2f}s")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    return history


def evaluate_model(
    model: nn.Module, test_features: torch.Tensor, test_labels: torch.Tensor, args
) -> dict:
    """Evaluate the model and return detailed metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        batch_size = args.batch_size
        dataset_size = len(test_features)

        for i in range(0, dataset_size, batch_size):
            end_idx = min(i + batch_size, dataset_size)
            batch_features = test_features[i:end_idx].to(device)
            batch_labels = test_labels[i:end_idx]

            # Measure inference time
            start_time = time.time()
            outputs = model(batch_features)
            inference_time = time.time() - start_time

            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            inference_times.append(inference_time / len(batch_features))

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_inference_time = np.mean(inference_times)

    # Classification report
    class_report = classification_report(
        all_labels,
        all_predictions,
        target_names=[f"Digit {i}" for i in range(10)],
        output_dict=True,
    )

    results = {
        "accuracy": accuracy * 100,
        "avg_inference_time_ms": avg_inference_time * 1000,
        "classification_report": class_report,
        "predictions": all_predictions,
        "true_labels": all_labels,
    }

    return results


def plot_training_history(history: dict, save_path: str = None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curves
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Accuracy curves
    axes[0, 1].plot(history["train_acc"], label="Train Acc")
    axes[0, 1].plot(history["val_acc"], label="Val Acc")
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()

    # Epoch times
    axes[1, 0].plot(history["epoch_times"])
    axes[1, 0].set_title("Training Time per Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Time (seconds)")

    # Learning rate (if available)
    if "learning_rates" in history:
        axes[1, 1].plot(history["learning_rates"])
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Learning Rate\nNot Tracked",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train digit classifier")
    parser.add_argument(
        "--model_type",
        default="lightweight",
        choices=["lightweight", "mini", "mlp"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--feature_type",
        default="mfcc",
        choices=["mfcc", "mel", "spectrogram"],
        help="Type of audio features to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--save_model",
        default="models/best_model.pth",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--save_plots",
        default="training_curves.png",
        help="Path to save training plots",
    )

    args = parser.parse_args()

    print("=== Digit Classification Training ===")
    print(f"Model: {args.model_type}")
    print(f"Features: {args.feature_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    # Load dataset
    print("\n1. Loading dataset...")
    audio_data, labels = load_fsdd_dataset()

    # Create data loaders for original audio
    from data.dataset import create_data_loaders

    train_loader, test_loader = create_data_loaders(
        audio_data, labels, batch_size=args.batch_size
    )

    # Initialize feature extractor
    print("\n2. Setting up feature extraction...")
    feature_extractor = AudioFeatureExtractor()

    # Extract features
    print("\n3. Extracting features...")
    train_features, train_labels = extract_features_from_loader(
        train_loader, feature_extractor, args.feature_type
    )
    test_features, test_labels = extract_features_from_loader(
        test_loader, feature_extractor, args.feature_type
    )

    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    # Create model
    print("\n4. Creating model...")
    input_channels = train_features.shape[1]
    input_length = train_features.shape[2]

    model = create_model(
        args.model_type,
        input_channels=input_channels,
        input_length=input_length,
        num_classes=10,
    )

    print(f"Model: {model.__class__.__name__}")
    param_info = count_parameters(model)
    print(f"Parameters: {param_info['total_parameters']:,}")
    print(f"Model size: {param_info['size_mb']:.2f} MB")

    # Split train into train/val
    val_split = 0.2
    val_size = int(len(train_features) * val_split)
    indices = torch.randperm(len(train_features))

    val_features = train_features[indices[:val_size]]
    val_labels = train_labels[indices[:val_size]]
    train_features = train_features[indices[val_size:]]
    train_labels = train_labels[indices[val_size:]]

    print(
        f"Final split - Train: {len(train_features)}, "
        f"Val: {len(val_features)}, Test: {len(test_features)}"
    )

    # Train model
    print("\n5. Training model...")
    history = train_model(
        model, train_features, train_labels, val_features, val_labels, args
    )

    # Evaluate model
    print("\n6. Evaluating model...")
    results = evaluate_model(model, test_features, test_labels, args)

    print(f"\nFinal Results:")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Avg Inference Time: {results['avg_inference_time_ms']:.2f} ms")

    # Print detailed classification report
    print("\nPer-class Performance:")
    for i in range(10):
        class_metrics = results["classification_report"][f"Digit {i}"]
        print(
            f"  Digit {i}: Precision: {class_metrics['precision']:.3f}, "
            f"Recall: {class_metrics['recall']:.3f}, "
            f"F1: {class_metrics['f1-score']:.3f}"
        )

    # Save model
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "feature_type": args.feature_type,
            "input_channels": input_channels,
            "input_length": input_length,
            "results": results,
            "args": args,
        },
        args.save_model,
    )

    print(f"\nModel saved to: {args.save_model}")

    # Plot training curves
    plot_training_history(history, args.save_plots)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
