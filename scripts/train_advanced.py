#!/usr/bin/env python3
"""
Advanced Training Script for Digit Classification
===============================================

Enhanced training with state-of-the-art optimization techniques:
- Advanced model architectures (ResNet, EfficientNet, Transformer, Ensemble)
- Cosine annealing with warm restarts
- Label smoothing
- Mixed precision training
- Advanced data augmentation
- Early stopping with patience
- Model ensembling
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.dataset import load_fsdd_dataset, create_data_loaders
from features.audio_features import AudioFeatureExtractor
from models.digit_classifier import create_model


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""

    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art techniques."""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # Loss function with label smoothing
        if config.get("label_smoothing", 0.0) > 0:
            self.criterion = LabelSmoothingLoss(
                config["num_classes"], config["label_smoothing"]
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if config.get("mixed_precision", False)
            else None
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0

    def _create_optimizer(self):
        """Create optimizer with advanced settings."""
        if self.config["optimizer"] == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif self.config["optimizer"] == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                momentum=0.9,
                weight_decay=self.config["weight_decay"],
                nesterov=True,
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config["epochs"] // 4,
                T_mult=2,
                eta_min=self.config["learning_rate"] * 0.01,
            )
        elif self.config["scheduler"] == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config["epochs"] // 3, gamma=0.1
            )
        elif self.config["scheduler"] == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
            )
        else:
            return None

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Print progress
            if batch_idx % 20 == 0:
                print(
                    f"    Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc, all_predictions, all_targets

    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"\n🚀 Starting advanced training with {self.config['model_type']} model")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.config['optimizer']}")
        print(f"Scheduler: {self.config['scheduler']}")
        print(f"Mixed precision: {self.config.get('mixed_precision', False)}")
        print(f"Label smoothing: {self.config.get('label_smoothing', 0.0)}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(self.config["epochs"]):
            epoch_start = time.time()

            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)

            # Update learning rate
            if self.scheduler:
                if self.config["scheduler"] == "plateau":
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"🎯 New best validation accuracy: {val_acc:.2f}%")
            else:
                self.patience_counter += 1

            # Print epoch results
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {lr:.6f}")
            print(f"Epoch Time: {epoch_time:.2f}s")

            # Early stopping
            if self.patience_counter >= self.config.get("patience", 10):
                print(
                    f"\n⏰ Early stopping after {self.patience_counter} epochs without improvement"
                )
                break

        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.2f}s")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}%")

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print("📥 Loaded best model weights")

        return self.history

    def plot_training_history(self, save_path="training_history.png"):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        ax1.plot(self.history["train_loss"], label="Train Loss", color="blue")
        ax1.plot(self.history["val_loss"], label="Val Loss", color="red")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(self.history["train_acc"], label="Train Acc", color="blue")
        ax2.plot(self.history["val_acc"], label="Val Acc", color="red")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        # Learning Rate
        ax3.plot(self.history["learning_rates"], color="green")
        ax3.set_title("Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_yscale("log")
        ax3.grid(True)

        # Placeholder for confusion matrix or other metrics
        ax4.text(
            0.5,
            0.5,
            f"Best Val Acc:\n{self.best_val_acc:.2f}%",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax4.transAxes,
        )
        ax4.set_title("Best Performance")
        ax4.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📈 Training history saved to {save_path}")


def create_advanced_config(args):
    """Create advanced training configuration."""
    return {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "num_classes": 10,
        "mixed_precision": args.mixed_precision,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "feature_type": args.feature_type,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Advanced Digit Classification Training"
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        default="advanced",
        choices=[
            "advanced",
            "efficient",
            "transformer",
            "ensemble",
            "lightweight",
            "mini",
            "mlp",
        ],
        help="Model architecture",
    )
    parser.add_argument(
        "--feature_type",
        default="mfcc",
        choices=["mfcc", "mel", "spectrogram"],
        help="Feature extraction type",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # Optimization
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="LR scheduler",
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing"
    )

    # Paths
    parser.add_argument("--save_dir", default="models", help="Model save directory")
    parser.add_argument("--plots_dir", default="plots", help="Plots save directory")

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Load data
    print("📂 Loading dataset...")
    audio_data, labels = load_fsdd_dataset()

    # Create feature extractor
    feature_extractor = AudioFeatureExtractor()

    # Split data into train/val/test using raw audio
    from sklearn.model_selection import train_test_split

    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        audio_data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Second split: separate train/val from remaining 80% (16% val, 64% train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    # Create datasets
    from data.dataset import FSSDDataset
    from torch.utils.data import DataLoader

    train_dataset = FSSDDataset(X_train, y_train)
    val_dataset = FSSDDataset(X_val, y_val)
    test_dataset = FSSDDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"✅ Dataset loaded: {len(audio_data)} samples")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Extract features to get dimensions
    def extract_features_from_data(data_loader, feature_extractor, feature_type):
        """Extract features from data loader."""
        features = []
        labels = []

        for audio_batch, label_batch in data_loader:
            for i in range(len(audio_batch)):
                audio = audio_batch[i].numpy()
                feature = feature_extractor.extract_features(audio, feature_type)
                features.append(feature)
                labels.append(label_batch[i].item())

        return torch.stack(features), torch.tensor(labels)

    # Extract features for all splits
    print("🔧 Extracting features...")
    train_features, train_labels = extract_features_from_data(
        train_loader, feature_extractor, args.feature_type
    )
    val_features, val_labels = extract_features_from_data(
        val_loader, feature_extractor, args.feature_type
    )
    test_features, test_labels = extract_features_from_data(
        test_loader, feature_extractor, args.feature_type
    )

    print(f"✅ Feature extraction complete")
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    # Get feature dimensions
    input_channels, input_length = train_features.shape[1], train_features.shape[2]

    print(f"🎯 Feature shape: ({input_channels}, {input_length})")

    # Create model
    print(f"🧠 Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        input_channels=input_channels,
        input_length=input_length,
        num_classes=10,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Create trainer
    config = create_advanced_config(args)
    trainer = AdvancedTrainer(model, device, config)

    # Train model using feature tensors
    class FeatureDataLoader:
        def __init__(self, features, labels, batch_size, shuffle=False):
            self.features = features
            self.labels = labels
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __len__(self):
            return (len(self.features) + self.batch_size - 1) // self.batch_size

        def __iter__(self):
            indices = (
                torch.randperm(len(self.features))
                if self.shuffle
                else torch.arange(len(self.features))
            )
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                yield self.features[batch_indices], self.labels[batch_indices]

    # Create feature-based data loaders
    train_feature_loader = FeatureDataLoader(
        train_features, train_labels, args.batch_size, shuffle=True
    )
    val_feature_loader = FeatureDataLoader(
        val_features, val_labels, args.batch_size, shuffle=False
    )
    test_feature_loader = FeatureDataLoader(
        test_features, test_labels, args.batch_size, shuffle=False
    )

    # Train model
    history = trainer.train(train_feature_loader, val_feature_loader)

    # Save model
    model_name = f"{args.model_type}_{args.feature_type}_advanced.pth"
    model_path = os.path.join(args.save_dir, model_name)

    torch.save(
        {
            "model_type": args.model_type,
            "feature_type": args.feature_type,
            "input_channels": input_channels,
            "input_length": input_length,
            "num_classes": 10,
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "best_val_acc": trainer.best_val_acc,
        },
        model_path,
    )

    print(f"💾 Model saved to: {model_path}")

    # Save training plots
    plot_path = os.path.join(
        args.plots_dir, f"{args.model_type}_{args.feature_type}_history.png"
    )
    trainer.plot_training_history(plot_path)

    # Test evaluation
    print("\n🧪 Final evaluation on test set...")
    test_loss, test_acc, test_preds, test_targets = trainer.validate(
        test_feature_loader
    )
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Classification report
    print("\n📋 Classification Report:")
    print(
        classification_report(
            test_targets, test_preds, target_names=[f"Digit {i}" for i in range(10)]
        )
    )

    # Save results
    results = {
        "model_type": args.model_type,
        "feature_type": args.feature_type,
        "config": config,
        "best_val_acc": trainer.best_val_acc,
        "test_acc": test_acc,
        "total_params": total_params,
    }

    results_path = os.path.join(args.save_dir, f"{args.model_type}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"📊 Results saved to: {results_path}")
    print(
        f"\n🎉 Training completed! Best model: {trainer.best_val_acc:.2f}% validation accuracy"
    )


if __name__ == "__main__":
    main()
