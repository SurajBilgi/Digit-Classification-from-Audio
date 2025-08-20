"""
Lightweight CNN models for digit classification from audio features.
Optimized for fast inference and minimal latency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class LightweightCNN(nn.Module):
    """Lightweight CNN for digit classification from audio features."""

    def __init__(
        self,
        input_channels: int = 13,
        num_classes: int = 10,
        input_length: int = 32,
        dropout: float = 0.3,
    ):
        """
        Initialize the CNN model.

        Args:
            input_channels: Number of input feature channels (e.g., 13 for
                           MFCC)
            num_classes: Number of output classes (10 for digits 0-9)
            input_length: Input sequence length (time dimension)
            dropout: Dropout rate for regularization
        """
        super(LightweightCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_length = input_length

        # First convolutional block
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # Calculate the size after convolutions
        conv_output_size = self._calculate_conv_output_size()

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after all convolutions."""
        # Simulate forward pass to get output size
        x = torch.zeros(1, self.input_channels, self.input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class MiniCNN(nn.Module):
    """Ultra-lightweight CNN for very fast inference."""

    def __init__(
        self, input_channels: int = 13, num_classes: int = 10, input_length: int = 32
    ):
        """
        Initialize the mini CNN model.

        Args:
            input_channels: Number of input feature channels
            num_classes: Number of output classes
            input_length: Input sequence length
        """
        super(MiniCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_length = input_length

        # Single convolutional block
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(4)

        # Second conv block
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        # Calculate output size
        conv_output_size = self._calculate_conv_output_size()

        # Simple fully connected layer
        self.fc = nn.Linear(conv_output_size, num_classes)

    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after all convolutions."""
        x = torch.zeros(1, self.input_channels, self.input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the mini model."""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SimpleMLMLP(nn.Module):
    """Simple Multi-Layer Perceptron for baseline comparison."""

    def __init__(
        self,
        input_size: int,
        num_classes: int = 10,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        """
        Initialize the MLP model.

        Args:
            input_size: Size of flattened input features
            num_classes: Number of output classes
            hidden_size: Size of hidden layers
            dropout: Dropout rate
        """
        super(SimpleMLMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def create_model(
    model_type="advanced", input_channels=13, input_length=32, num_classes=10
):
    """
    Create a model for digit classification.

    Args:
        model_type: Model architecture type
        input_channels: Number of input feature channels
        input_length: Length of input sequence
        num_classes: Number of output classes

    Returns:
        PyTorch model
    """
    # Import advanced models
    try:
        from .advanced_digit_classifier import create_advanced_model

        # Check if it's an advanced model type
        advanced_types = ["advanced", "efficient", "transformer", "ensemble"]
        if model_type in advanced_types:
            return create_advanced_model(
                model_type, input_channels, input_length, num_classes
            )
    except ImportError:
        print("⚠️  Advanced models not available, falling back to basic models")

    # Original basic models
    if model_type == "lightweight":
        return LightweightCNN(input_channels, num_classes, input_length)
    elif model_type == "mini":
        return MiniCNN(input_channels, num_classes, input_length)
    elif model_type == "mlp":
        input_size = input_channels * input_length
        return SimpleMLMLP(input_size, num_classes)
    else:
        # Default to advanced if available, otherwise lightweight
        try:
            from .advanced_digit_classifier import create_advanced_model

            return create_advanced_model(
                "advanced", input_channels, input_length, num_classes
            )
        except ImportError:
            print(f"⚠️  Unknown model type '{model_type}', using lightweight CNN")
            return LightweightCNN(input_channels, num_classes, input_length)


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts and model size info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB (assuming float32)
    size_mb = total_params * 4 / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "size_mb": size_mb,
    }


if __name__ == "__main__":
    # Test different models
    models = {
        "LightweightCNN": create_model("lightweight"),
        "MiniCNN": create_model("mini"),
        "SimpleMLP": create_model("mlp"),
    }

    # Test input
    batch_size = 4
    input_channels = 13
    sequence_length = 32
    test_input = torch.randn(batch_size, input_channels, sequence_length)

    for name, model in models.items():
        print(f"\n{name}:")

        # Forward pass
        if name == "SimpleMLP":
            # MLP expects flattened input
            output = model(test_input)
        else:
            output = model(test_input)

        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

        # Parameter count
        params_info = count_parameters(model)
        print(f"  Total parameters: {params_info['total_parameters']:,}")
        print(f"  Model size: {params_info['size_mb']:.2f} MB")
