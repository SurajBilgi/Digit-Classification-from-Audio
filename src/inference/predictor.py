"""
Fast inference pipeline for digit classification from audio.
Optimized for minimal latency in real-time applications.
"""

import os
import sys
import time
import torch
import numpy as np
import librosa
from typing import Union, Tuple
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.audio_features import AudioFeatureExtractor, pad_features
from models.digit_classifier import create_model

warnings.filterwarnings("ignore")


class DigitPredictor:
    """Fast digit predictor optimized for real-time inference."""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the saved model
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model = None
        self.feature_extractor = None
        self.model_config = None

        # Performance tracking
        self.inference_times = []
        self.preprocessing_times = []

        # Load model
        self._load_model(model_path)

    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_device = torch.device(device)
        print(f"Inference device: {torch_device}")
        return torch_device

    def _load_model(self, model_path: str):
        """Load the trained model and configuration."""
        # Load with weights_only=False for compatibility with PyTorch 2.6+
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Extract model configuration
        self.model_config = {
            "model_type": checkpoint["model_type"],
            "feature_type": checkpoint["feature_type"],
            "input_channels": checkpoint["input_channels"],
            "input_length": checkpoint["input_length"],
        }

        # Create and load model
        self.model = create_model(
            self.model_config["model_type"],
            input_channels=self.model_config["input_channels"],
            input_length=self.model_config["input_length"],
            num_classes=10,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Setup feature extractor
        self.feature_extractor = AudioFeatureExtractor()

        print(f"Loaded {self.model_config['model_type']} model")
        print(f"Feature type: {self.model_config['feature_type']}")
        print(
            f"Input shape: ({self.model_config['input_channels']}, "
            f"{self.model_config['input_length']})"
        )

        # Warm up the model
        self._warmup()

    def _warmup(self, num_warmup: int = 5):
        """Warm up the model for consistent timing."""
        print("Warming up model...")
        dummy_audio = np.random.randn(8000)  # 1 second at 8kHz

        for _ in range(num_warmup):
            _ = self.predict(dummy_audio, return_confidence=False)

        # Clear timing statistics from warmup
        self.inference_times.clear()
        self.preprocessing_times.clear()

        print("Model warmed up!")

    def preprocess_audio(
        self, audio: Union[np.ndarray, str], target_sr: int = 8000
    ) -> np.ndarray:
        """
        Preprocess audio for prediction.

        Args:
            audio: Audio array or path to audio file
            target_sr: Target sample rate

        Returns:
            Preprocessed audio array
        """
        start_time = time.time()

        # Load audio if path is provided
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=target_sr)
        else:
            # Assume it's already an array at correct sample rate
            audio = np.array(audio, dtype=np.float32)

        # Ensure we have exactly 1 second of audio (8000 samples at 8kHz)
        target_length = target_sr

        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            # Take the middle portion
            start_idx = (len(audio) - target_length) // 2
            audio = audio[start_idx : start_idx + target_length]

        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        preprocessing_time = time.time() - start_time
        self.preprocessing_times.append(preprocessing_time)

        return audio

    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract features from preprocessed audio."""
        features = self.feature_extractor.extract_features(
            audio, self.model_config["feature_type"]
        )

        # Normalize features
        features = self.feature_extractor.normalize_features(features)

        # Pad to expected length
        features = pad_features(features, self.model_config["input_length"])

        # Add batch dimension and move to device
        features = features.unsqueeze(0).to(self.device)

        return features

    def predict(
        self, audio: Union[np.ndarray, str], return_confidence: bool = True
    ) -> Union[int, Tuple[int, float]]:
        """
        Predict digit from audio.

        Args:
            audio: Audio array or path to audio file
            return_confidence: Whether to return confidence score

        Returns:
            Predicted digit (and confidence if requested)
        """
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio)

        # Extract features
        features = self.extract_features(processed_audio)

        # Inference
        start_time = time.time()

        with torch.no_grad():
            logits = self.model(features)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_digit = predicted.item()
            confidence_score = confidence.item()

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        if return_confidence:
            return predicted_digit, confidence_score
        else:
            return predicted_digit

    def predict_batch(self, audio_list: list) -> list:
        """
        Predict digits for a batch of audio samples.

        Args:
            audio_list: List of audio arrays or file paths

        Returns:
            List of (digit, confidence) tuples
        """
        results = []

        for audio in audio_list:
            digit, confidence = self.predict(audio, return_confidence=True)
            results.append((digit, confidence))

        return results

    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {"message": "No predictions made yet"}

        inference_times_ms = [t * 1000 for t in self.inference_times]
        preprocessing_times_ms = [t * 1000 for t in self.preprocessing_times]

        return {
            "avg_inference_time_ms": np.mean(inference_times_ms),
            "max_inference_time_ms": np.max(inference_times_ms),
            "min_inference_time_ms": np.min(inference_times_ms),
            "avg_preprocessing_time_ms": np.mean(preprocessing_times_ms),
            "total_predictions": len(self.inference_times),
            "predictions_per_second": 1.0 / np.mean(self.inference_times),
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times.clear()
        self.preprocessing_times.clear()


class BatchPredictor:
    """Optimized batch predictor for processing multiple audio files."""

    def __init__(self, model_path: str, batch_size: int = 32, device: str = "auto"):
        """
        Initialize batch predictor.

        Args:
            model_path: Path to saved model
            batch_size: Batch size for inference
            device: Computation device
        """
        self.batch_size = batch_size
        device_str = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(device_str)

        # Load model using single predictor for configuration
        self.single_predictor = DigitPredictor(model_path, device)
        self.model = self.single_predictor.model
        self.feature_extractor = self.single_predictor.feature_extractor
        self.model_config = self.single_predictor.model_config

    def predict_batch(self, audio_list: list) -> list:
        """
        Predict digits for a batch of audio samples efficiently.

        Args:
            audio_list: List of audio arrays or file paths

        Returns:
            List of (digit, confidence) tuples
        """
        results = []

        # Process in batches
        for i in range(0, len(audio_list), self.batch_size):
            batch = audio_list[i : i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results

    def _process_batch(self, audio_batch: list) -> list:
        """Process a single batch of audio samples."""
        # Preprocess all audio in batch
        processed_audio = []
        for audio in audio_batch:
            processed = self.single_predictor.preprocess_audio(audio)
            processed_audio.append(processed)

        # Extract features for all audio
        features_list = []
        for audio in processed_audio:
            features = self.single_predictor.extract_features(audio)
            features_list.append(features.squeeze(0))  # Remove batch dim

        # Stack into batch tensor
        batch_features = torch.stack(features_list).to(self.device)

        # Batch inference
        with torch.no_grad():
            logits = self.model(batch_features)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

        # Convert to list of tuples
        results = []
        for i in range(len(audio_batch)):
            digit = predicted[i].item()
            confidence = confidences[i].item()
            results.append((digit, confidence))

        return results


def load_predictor(model_path: str, device: str = "auto") -> DigitPredictor:
    """
    Convenience function to load a digit predictor.

    Args:
        model_path: Path to saved model
        device: Computation device

    Returns:
        Initialized DigitPredictor
    """
    return DigitPredictor(model_path, device)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test digit predictor")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--audio_file", help="Path to audio file to test")
    parser.add_argument("--device", default="auto", help="Device for inference")

    args = parser.parse_args()

    # Load predictor
    predictor = load_predictor(args.model_path, args.device)

    if args.audio_file:
        # Test single prediction
        digit, confidence = predictor.predict(args.audio_file)
        print(f"Predicted digit: {digit}")
        print(f"Confidence: {confidence:.3f}")

        # Show performance stats
        stats = predictor.get_performance_stats()
        inf_time = stats["avg_inference_time_ms"]
        print(f"Inference time: {inf_time:.2f} ms")
    else:
        # Test with random audio
        test_audio = np.random.randn(8000)
        digit, confidence = predictor.predict(test_audio)
        print(f"Test prediction - Digit: {digit}, " f"Confidence: {confidence:.3f}")

        # Performance test
        print("\nPerformance test (100 predictions)...")
        for _ in range(100):
            test_audio = np.random.randn(8000)
            _ = predictor.predict(test_audio, return_confidence=False)

        stats = predictor.get_performance_stats()
        avg_time = stats["avg_inference_time_ms"]
        pred_per_sec = stats["predictions_per_second"]
        max_time = stats["max_inference_time_ms"]
        min_time = stats["min_inference_time_ms"]

        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Predictions per second: {pred_per_sec:.1f}")
        print(f"Max inference time: {max_time:.2f} ms")
        print(f"Min inference time: {min_time:.2f} ms")
