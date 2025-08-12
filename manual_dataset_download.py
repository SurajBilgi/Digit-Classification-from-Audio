#!/usr/bin/env python3
"""
Manual Dataset Download Script for Free Spoken Digit Dataset (FSDD)
==================================================================

This script downloads the FSDD dataset directly from GitHub as a backup
when Hugging Face datasets fail to load (torchcodec/FFmpeg issues).

Usage:
    python manual_dataset_download.py

The script will:
1. Download the dataset from GitHub
2. Extract and organize the files
3. Create the proper folder structure
4. Test the dataset loading
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import sys


def download_with_progress(url, filename):
    """Download a file with progress indicator."""

    def report_progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r{filename}: {percent}% complete")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, filename, report_progress)
    print()  # New line after progress


def download_fsdd_manual():
    """
    Download and setup the Free Spoken Digit Dataset manually.
    """
    print("🎯 Manual Free Spoken Digit Dataset Download")
    print("=" * 50)

    # Create data directory
    data_dir = Path("data")
    fsdd_dir = data_dir / "fsdd_manual"
    recordings_dir = fsdd_dir / "recordings"

    print(f"📁 Creating directory structure...")
    recordings_dir.mkdir(parents=True, exist_ok=True)

    # Download URLs
    github_repo_url = "https://github.com/Jakobovski/free-spoken-digit-dataset"
    download_url = f"{github_repo_url}/archive/refs/heads/master.zip"
    zip_file = "fsdd_master.zip"

    print(f"⬇️  Downloading FSDD from GitHub...")
    print(f"Source: {github_repo_url}")

    try:
        download_with_progress(download_url, zip_file)
        print(f"✅ Download complete: {zip_file}")

        # Extract the zip file
        print(f"📦 Extracting dataset...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall("temp_fsdd")

        # Move recordings to proper location
        temp_recordings = Path("temp_fsdd/free-spoken-digit-dataset-master/recordings")
        if temp_recordings.exists():
            print(f"📂 Organizing files...")

            # Copy all .wav files
            wav_files = list(temp_recordings.glob("*.wav"))
            print(f"Found {len(wav_files)} audio files")

            for wav_file in wav_files:
                dest_file = recordings_dir / wav_file.name
                shutil.copy2(wav_file, dest_file)

            print(f"✅ Copied {len(wav_files)} files to {recordings_dir}")

            # Also copy metadata if it exists
            metadata_file = Path(
                "temp_fsdd/free-spoken-digit-dataset-master/metadata.py"
            )
            if metadata_file.exists():
                shutil.copy2(metadata_file, fsdd_dir / "metadata.py")
                print(f"✅ Copied metadata file")

        else:
            print(f"❌ Could not find recordings directory in download")
            return False

        # Cleanup
        print(f"🧹 Cleaning up temporary files...")
        os.remove(zip_file)
        shutil.rmtree("temp_fsdd")

        print(f"✅ Manual download complete!")
        print(f"📍 Dataset location: {fsdd_dir.absolute()}")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Cleanup on failure
        if os.path.exists(zip_file):
            os.remove(zip_file)
        if os.path.exists("temp_fsdd"):
            shutil.rmtree("temp_fsdd")
        return False


def analyze_dataset_structure():
    """Analyze the downloaded dataset structure."""
    print("\n📊 Analyzing Dataset Structure")
    print("=" * 30)

    recordings_dir = Path("data/fsdd_manual/recordings")

    if not recordings_dir.exists():
        print("❌ Dataset not found. Run download first.")
        return

    # Count files by digit
    digit_counts = {}
    speaker_counts = {}
    total_files = 0

    for wav_file in recordings_dir.glob("*.wav"):
        total_files += 1

        # Parse filename: {digit}_{speaker}_{index}.wav
        parts = wav_file.stem.split("_")
        if len(parts) >= 2:
            digit = parts[0]
            speaker = parts[1]

            digit_counts[digit] = digit_counts.get(digit, 0) + 1
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

    print(f"📈 Dataset Statistics:")
    print(f"   Total files: {total_files}")
    print(f"   Unique digits: {len(digit_counts)}")
    print(f"   Unique speakers: {len(speaker_counts)}")

    print(f"\n🔢 Files per digit:")
    for digit in sorted(digit_counts.keys()):
        print(f"   Digit {digit}: {digit_counts[digit]} files")

    print(f"\n🎤 Files per speaker:")
    for speaker in sorted(speaker_counts.keys()):
        print(f"   Speaker {speaker}: {speaker_counts[speaker]} files")


def create_dataset_loader():
    """Create a simple dataset loader for the manual dataset."""
    print("\n⚙️  Creating Dataset Loader")
    print("=" * 25)

    # Create a simple loader script
    loader_script = '''
import os
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, List

def load_manual_fsdd() -> Tuple[List[np.ndarray], List[int]]:
    """
    Load manually downloaded FSDD dataset.
    
    Returns:
        Tuple of (audio_data, labels)
    """
    recordings_dir = Path("data/fsdd_manual/recordings")
    
    if not recordings_dir.exists():
        raise FileNotFoundError(
            "Manual dataset not found. Run 'python manual_dataset_download.py' first"
        )
    
    audio_data = []
    labels = []
    
    print(f"Loading manual FSDD dataset from {recordings_dir}")
    
    for wav_file in sorted(recordings_dir.glob("*.wav")):
        try:
            # Parse filename: {digit}_{speaker}_{index}.wav
            digit = int(wav_file.stem.split('_')[0])
            
            # Load audio file
            audio, sr = librosa.load(wav_file, sr=8000)
            
            audio_data.append(audio)
            labels.append(digit)
            
        except Exception as e:
            print(f"Warning: Could not load {wav_file}: {e}")
            continue
    
    print(f"✅ Loaded {len(audio_data)} files successfully")
    return audio_data, labels

if __name__ == "__main__":
    # Test the loader
    audio_data, labels = load_manual_fsdd()
    
    print(f"Sample statistics:")
    print(f"  Audio files: {len(audio_data)}")
    print(f"  Labels range: {min(labels)} to {max(labels)}")
    print(f"  Sample audio shape: {audio_data[0].shape if audio_data else 'None'}")
'''

    # Write the loader script
    with open("src/data/manual_loader.py", "w") as f:
        f.write(loader_script)

    print("✅ Created manual dataset loader: src/data/manual_loader.py")


def update_main_dataset_loader():
    """Update the main dataset loader to include manual fallback."""
    print("\n🔧 Updating Main Dataset Loader")
    print("=" * 30)

    print("✅ The main dataset loader already has fallback mechanisms.")
    print("💡 To use manual dataset, modify src/data/dataset.py:")
    print("   - Import: from .manual_loader import load_manual_fsdd")
    print("   - Add as final fallback in the exception handling")


def main():
    """Main function to run the manual download process."""
    print("🎯 Free Spoken Digit Dataset - Manual Download")
    print("=" * 50)

    # Check if dataset already exists
    fsdd_dir = Path("data/fsdd_manual/recordings")
    if fsdd_dir.exists() and len(list(fsdd_dir.glob("*.wav"))) > 0:
        print("📁 Manual dataset already exists!")
        print("🔍 Analyzing existing dataset...")
        analyze_dataset_structure()

        response = input("\n🤔 Re-download dataset? (y/N): ").lower().strip()
        if response != "y":
            print("✅ Using existing dataset")
            create_dataset_loader()
            return

    # Download the dataset
    success = download_fsdd_manual()

    if success:
        # Analyze the downloaded dataset
        analyze_dataset_structure()

        # Create the dataset loader
        create_dataset_loader()

        # Show next steps
        print("\n🎉 Manual Download Complete!")
        print("=" * 30)
        print("📋 Next Steps:")
        print("1. Test the dataset:")
        print(
            '   python -c "from src.data.manual_loader import load_manual_fsdd; load_manual_fsdd()"'
        )
        print()
        print("2. Use in your code:")
        print("   from src.data.manual_loader import load_manual_fsdd")
        print("   audio_data, labels = load_manual_fsdd()")
        print()
        print("3. Train a model:")
        print("   python scripts/train.py")

    else:
        print("\n❌ Manual download failed")
        print("💡 Alternative options:")
        print("1. Check your internet connection")
        print("2. Try the Hugging Face datasets with stable environment")
        print(
            "3. Download manually from: https://github.com/Jakobovski/free-spoken-digit-dataset"
        )


if __name__ == "__main__":
    main()
