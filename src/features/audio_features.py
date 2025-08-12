"""
Audio feature extraction for digit classification.
Provides multiple feature extraction methods optimized for lightweight
inference.
"""

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


class AudioFeatureExtractor:
    """Lightweight audio feature extractor for digit classification."""

    def __init__(
        self,
        sample_rate: int = 8000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 64,
        n_mfcc: int = 13,
    ):
        """
        Initialize the feature extractor.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Audio array

        Returns:
            MFCC features of shape (n_mfcc, time_frames)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfccs

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.

        Args:
            audio: Audio array

        Returns:
            Mel spectrogram of shape (n_mels, time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract regular spectrogram from audio.

        Args:
            audio: Audio array

        Returns:
            Spectrogram of shape (freq_bins, time_frames)
        """
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(stft)
        # Convert to log scale
        log_spec = librosa.amplitude_to_db(spectrogram, ref=np.max)
        return log_spec

    def extract_features(
        self, audio: Union[np.ndarray, torch.Tensor], feature_type: str = "mfcc"
    ) -> torch.Tensor:
        """
        Extract features from audio with specified type.

        Args:
            audio: Audio data (numpy array or torch tensor)
            feature_type: Type of features ('mfcc', 'mel', 'spectrogram')

        Returns:
            Feature tensor of shape (1, features, time) for single audio
            or (batch, features, time) for batch of audio
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Handle batch dimension
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]  # Add batch dimension
            single_audio = True
        else:
            single_audio = False

        features_list = []
        for i in range(audio_np.shape[0]):
            audio_sample = audio_np[i]

            if feature_type == "mfcc":
                features = self.extract_mfcc(audio_sample)
            elif feature_type == "mel":
                features = self.extract_mel_spectrogram(audio_sample)
            elif feature_type == "spectrogram":
                features = self.extract_spectrogram(audio_sample)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            features_list.append(features)

        # Stack features
        features_array = np.stack(features_list, axis=0)
        features_tensor = torch.FloatTensor(features_array)

        if single_audio:
            features_tensor = features_tensor.squeeze(0)

        return features_tensor

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features for better training stability.

        Args:
            features: Feature tensor

        Returns:
            Normalized feature tensor
        """
        # Normalize along feature dimension
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True) + 1e-8
        normalized = (features - mean) / std
        return normalized


def pad_features(features: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Pad or truncate features to fixed length.

    Args:
        features: Feature tensor of shape (..., time)
        max_length: Target time length

    Returns:
        Padded/truncated features
    """
    current_length = features.shape[-1]

    if current_length < max_length:
        # Pad with zeros
        pad_size = max_length - current_length
        features = F.pad(features, (0, pad_size))
    elif current_length > max_length:
        # Truncate
        features = features[..., :max_length]

    return features


if __name__ == "__main__":
    # Test feature extraction
    extractor = AudioFeatureExtractor()

    # Generate dummy audio
    dummy_audio = np.random.randn(8000)  # 1 second at 8kHz

    # Test different feature types
    for feature_type in ["mfcc", "mel", "spectrogram"]:
        features = extractor.extract_features(dummy_audio, feature_type)
        print(f"{feature_type} features shape: {features.shape}")

        # Test normalization
        normalized = extractor.normalize_features(features)
        print(
            f"Normalized {feature_type} mean: {normalized.mean():.3f}, "
            f"std: {normalized.std():.3f}"
        )

        # Test padding
        padded = pad_features(features, 50)
        print(f"Padded {feature_type} shape: {padded.shape}")
        print()
