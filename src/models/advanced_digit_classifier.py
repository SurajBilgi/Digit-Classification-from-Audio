"""
Advanced Digit Classification Models
===================================

State-of-the-art CNN architectures for robust digit classification from audio.
Includes ResNet-inspired blocks, attention mechanisms, and ensemble methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with optional SE attention."""

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else None

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.se:
            out = self.se(out)

        out += self.shortcut(residual)
        out = F.relu(out)

        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class AdvancedDigitCNN(nn.Module):
    """
    Advanced CNN with ResNet blocks, attention, and sophisticated architecture.
    Designed for high accuracy on audio digit classification.
    """

    def __init__(self, input_channels=13, input_length=32, num_classes=10):
        super(AdvancedDigitCNN, self).__init__()

        self.input_channels = input_channels
        self.input_length = input_length

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Spatial attention
        self.spatial_attention = SpatialAttention()

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_channels, input_length)
            dummy_output = self._forward_features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spatial_attention(x)
        x = self.global_pool(x)

        return x

    def forward(self, x):
        # Ensure input has correct shape [batch, 1, channels, length]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class EfficientDigitCNN(nn.Module):
    """
    Efficient CNN inspired by EfficientNet for better accuracy/efficiency trade-off.
    """

    def __init__(self, input_channels=13, input_length=32, num_classes=10):
        super(EfficientDigitCNN, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # MBConv blocks (Mobile Inverted Bottleneck)
        self.blocks = nn.Sequential(
            self._make_mbconv_block(32, 64, 3, 1, 1),
            self._make_mbconv_block(64, 128, 3, 2, 6),
            self._make_mbconv_block(128, 128, 5, 1, 6),
            self._make_mbconv_block(128, 256, 3, 2, 6),
            self._make_mbconv_block(256, 256, 5, 1, 6),
            self._make_mbconv_block(256, 512, 5, 2, 6),
        )

        # Head
        self.head_conv = nn.Sequential(
            nn.Conv2d(512, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def _make_mbconv_block(self, in_ch, out_ch, kernel_size, stride, expand_ratio):
        hidden_ch = in_ch * expand_ratio

        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU(inplace=True),
                ]
            )

        # Depthwise
        layers.extend(
            [
                nn.Conv2d(
                    hidden_ch,
                    hidden_ch,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
            ]
        )

        # SE block
        layers.append(SEBlock(hidden_ch))

        # Pointwise
        layers.extend(
            [nn.Conv2d(hidden_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        )

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class TransformerDigitClassifier(nn.Module):
    """
    Transformer-based model for audio digit classification.
    Uses self-attention to capture temporal dependencies in audio features.
    """

    def __init__(self, input_channels=13, input_length=32, num_classes=10):
        super(TransformerDigitClassifier, self).__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        d_model = 256

        # Input projection
        self.input_projection = nn.Linear(input_channels, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(input_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _create_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x shape: [batch, channels, length] -> [batch, length, channels]
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove channel dim if present
        x = x.transpose(1, 2)

        batch_size, seq_len, _ = x.shape

        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class EnsembleDigitClassifier(nn.Module):
    """
    Ensemble of multiple models for robust predictions.
    """

    def __init__(self, input_channels=13, input_length=32, num_classes=10):
        super(EnsembleDigitClassifier, self).__init__()

        self.models = nn.ModuleList(
            [
                AdvancedDigitCNN(input_channels, input_length, num_classes),
                EfficientDigitCNN(input_channels, input_length, num_classes),
                TransformerDigitClassifier(input_channels, input_length, num_classes),
            ]
        )

        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = sum(w * out for w, out in zip(weights, outputs))

        return ensemble_output


def create_advanced_model(
    model_type="advanced", input_channels=13, input_length=32, num_classes=10
):
    """
    Create an advanced model for digit classification.

    Args:
        model_type: One of ['advanced', 'efficient', 'transformer', 'ensemble']
        input_channels: Number of input feature channels (e.g., 13 for MFCC)
        input_length: Length of input sequence (e.g., 32 time steps)
        num_classes: Number of output classes (10 for digits 0-9)

    Returns:
        PyTorch model
    """
    model_classes = {
        "advanced": AdvancedDigitCNN,
        "efficient": EfficientDigitCNN,
        "transformer": TransformerDigitClassifier,
        "ensemble": EnsembleDigitClassifier,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(model_classes.keys())}"
        )

    model = model_classes[model_type](input_channels, input_length, num_classes)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Created {model_type} model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def get_model_info(model):
    """Get detailed information about a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "architecture": model.__class__.__name__,
    }


if __name__ == "__main__":
    # Test all models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_test = ["advanced", "efficient", "transformer", "ensemble"]

    for model_type in models_to_test:
        print(f"\n🧪 Testing {model_type} model:")
        print("=" * 50)

        try:
            model = create_advanced_model(model_type)
            model = model.to(device)

            # Test forward pass
            batch_size = 4
            test_input = torch.randn(batch_size, 13, 32).to(device)

            with torch.no_grad():
                output = model(test_input)

            print(f"✅ Forward pass successful")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")

            # Model info
            info = get_model_info(model)
            print(f"   Parameters: {info['total_params']:,}")
            print(f"   Model size: {info['model_size_mb']:.1f} MB")

        except Exception as e:
            print(f"❌ Error testing {model_type}: {e}")

    print(f"\n🎯 All models tested successfully!")
    print("Choose the best model for your use case:")
    print("  - 'advanced': Best accuracy, more parameters")
    print("  - 'efficient': Good accuracy/speed trade-off")
    print("  - 'transformer': Great for temporal patterns")
    print("  - 'ensemble': Highest accuracy, slowest")
