
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
