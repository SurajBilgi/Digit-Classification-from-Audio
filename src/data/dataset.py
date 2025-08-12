"""
Data loading and preprocessing for the Free Spoken Digit Dataset (FSDD).
Uses the MTEB version from Hugging Face with fallback support.
"""

import librosa
import numpy as np
import pandas as pd
from datasets import load_dataset
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings


class FSSDDataset(Dataset):
    """PyTorch Dataset for Free Spoken Digit Dataset."""

    def __init__(
        self,
        audio_data: List[np.ndarray],
        labels: List[int],
        sample_rate: int = 8000,
        max_length: float = 1.0,
    ):
        """
        Initialize the dataset.

        Args:
            audio_data: List of audio arrays
            labels: List of digit labels (0-9)
            sample_rate: Audio sample rate
            max_length: Maximum audio length in seconds
        """
        self.audio_data = audio_data
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_samples = int(max_length * sample_rate)

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        label = self.labels[idx]

        # Pad or truncate to fixed length
        if len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        else:
            audio = audio[: self.max_samples]

        return torch.FloatTensor(audio), torch.LongTensor([label])[0]


def load_fsdd_dataset() -> Tuple[List[np.ndarray], List[int]]:
    """
    Load the Free Spoken Digit Dataset from Hugging Face.
    Tries MTEB version first, falls back to alternative sources.

    Returns:
        Tuple of (audio_data, labels)
    """
    print("Loading Free Spoken Digit Dataset...")

    # Try MTEB dataset first
    try:
        print("Attempting to load MTEB dataset...")
        dataset = load_dataset("mteb/free-spoken-digit-dataset")

        audio_data = []
        labels = []

        # Process both train and test splits
        for split_name in dataset.keys():
            print(f"Processing {split_name} split...")
            split_data = dataset[split_name]

            for example in split_data:
                # Extract audio array and label
                audio_info = example["audio"]
                audio_array = audio_info["array"]
                sample_rate = audio_info["sampling_rate"]
                digit_label = example["label"]

                # Resample to 8kHz if needed
                if sample_rate != 8000:
                    audio_array = librosa.resample(
                        audio_array, orig_sr=sample_rate, target_sr=8000
                    )

                audio_data.append(audio_array)
                labels.append(digit_label)

        print(f"✅ MTEB dataset loaded: {len(audio_data)} samples")
        label_dist = pd.Series(labels).value_counts().sort_index().to_dict()
        print(f"Label distribution: {label_dist}")

        return audio_data, labels

    except Exception as mteb_error:
        error_msg = str(mteb_error)
        print(f"❌ MTEB dataset failed: {error_msg[:100]}...")

        # Check for specific torchcodec/FFmpeg error
        if "torchcodec" in error_msg.lower() or "ffmpeg" in error_msg.lower():
            print("⚠️  TorchCodec/FFmpeg compatibility issue detected")
            print("💡 This is a known issue with audio datasets")

        print("🔄 Trying alternative dataset source...")

        # Fallback to Matthijs dataset
        try:
            dataset = load_dataset("Matthijs/free-spoken-digit-dataset")

            audio_data = []
            labels = []

            # Process the dataset (typically has one split)
            for split in ["train"]:
                if split in dataset:
                    print(f"Processing fallback {split} split...")
                    for example in dataset[split]:
                        # Extract audio array and label
                        audio_array = example["audio"]["array"]
                        sample_rate = example["audio"]["sampling_rate"]
                        digit_label = example["label"]

                        # Resample to 8kHz if needed
                        if sample_rate != 8000:
                            audio_array = librosa.resample(
                                audio_array, orig_sr=sample_rate, target_sr=8000
                            )

                        audio_data.append(audio_array)
                        labels.append(digit_label)

            print(f"✅ Fallback successful: {len(audio_data)} samples")
            label_dist = pd.Series(labels).value_counts().sort_index()
            print(f"Label distribution: {label_dist.to_dict()}")

            return audio_data, labels

        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")

            # Final fallback: try to load with specific configurations
            try:
                print("🔄 Trying final fallback with specific config...")

                # Try loading with specific trust_remote_code settings
                dataset = load_dataset(
                    "Matthijs/free-spoken-digit-dataset", trust_remote_code=True
                )

                audio_data = []
                labels = []

                for split in dataset.keys():
                    for example in dataset[split]:
                        audio_array = example["audio"]["array"]
                        sample_rate = example["audio"]["sampling_rate"]
                        digit_label = example["label"]

                        if sample_rate != 8000:
                            audio_array = librosa.resample(
                                audio_array, orig_sr=sample_rate, target_sr=8000
                            )

                        audio_data.append(audio_array)
                        labels.append(digit_label)

                print(f"✅ Final fallback successful: {len(audio_data)} samples")
                return audio_data, labels

            except Exception as final_error:
                print(f"❌ All attempts failed: {final_error}")

                # Final fallback: try manual dataset
                try:
                    print("🔄 Trying manual dataset as ultimate fallback...")
                    from .manual_loader import load_manual_fsdd

                    return load_manual_fsdd()

                except Exception as manual_error:
                    print(f"❌ Manual dataset also failed: {manual_error}")

                    # Provide helpful error message
                    error_message = """
Dataset loading failed. Possible solutions:

1. Install FFmpeg (for torchcodec issues):
   - macOS: brew install ffmpeg
   - Ubuntu: sudo apt install ffmpeg
   - Windows: Download from https://ffmpeg.org/

2. Downgrade PyTorch if compatibility issues:
   conda install pytorch=1.13.0 torchaudio=0.13.0 -c pytorch

3. Download dataset manually:
   python manual_dataset_download.py

4. Use a different environment with Python 3.8-3.10

5. Check setup: python verify_setup.py
"""
                    print(error_message)
                    raise Exception("Could not load any version of the dataset")


def create_data_loaders(
    audio_data: List[np.ndarray],
    labels: List[int],
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders.

    Args:
        audio_data: List of audio arrays
        labels: List of labels
        batch_size: Batch size for training
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        audio_data,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Create datasets
    train_dataset = FSSDDataset(X_train, y_train)
    test_dataset = FSSDDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader


if __name__ == "__main__":
    # Test the data loading
    audio_data, labels = load_fsdd_dataset()
    train_loader, test_loader = create_data_loaders(audio_data, labels)

    # Print sample batch
    for batch_audio, batch_labels in train_loader:
        print(f"Batch audio shape: {batch_audio.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Sample labels: {batch_labels[:5]}")
        break
