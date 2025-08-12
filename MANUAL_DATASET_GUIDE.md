# Manual Dataset Download Guide
## Free Spoken Digit Dataset (FSDD)

### 🎯 Quick Answer: YES, you can download manually!

If you're experiencing the **torchcodec/FFmpeg error** with Hugging Face datasets, you can download the FSDD dataset directly from GitHub.

---

## 📁 **Folder Structure After Manual Download**

```
Digit-Classification-from-Audio/
├── data/
│   └── fsdd_manual/
│       ├── recordings/
│       │   ├── 0_jackson_0.wav
│       │   ├── 0_jackson_1.wav
│       │   ├── 0_nicolas_0.wav
│       │   ├── 1_jackson_0.wav
│       │   ├── 1_nicolas_0.wav
│       │   ├── 2_jackson_0.wav
│       │   └── ... (3,000 total files)
│       └── metadata.py (speaker information)
├── src/
│   └── data/
│       ├── dataset.py (main loader)
│       └── manual_loader.py (manual dataset loader)
└── manual_dataset_download.py (download script)
```

### 📄 **File Naming Convention**

All audio files follow this pattern:
```
{digit}_{speaker}_{recording_number}.wav
```

**Examples:**
- `0_jackson_0.wav` = Digit "0", spoken by "jackson", recording #0
- `5_nicolas_23.wav` = Digit "5", spoken by "nicolas", recording #23
- `9_yweweler_49.wav` = Digit "9", spoken by "yweweler", recording #49

---

## 🚀 **Option 1: Automated Download (Recommended)**

### Step 1: Run the Download Script
```bash
# Navigate to your project directory
cd /Users/surajbilgi/Documents/MyWork/My_LLM/Digit-Classification-from-Audio

# Run the manual download script
python manual_dataset_download.py
```

This script will:
- ✅ Download the dataset from GitHub (~20MB)
- ✅ Extract and organize files
- ✅ Create the proper folder structure
- ✅ Generate a custom dataset loader
- ✅ Analyze the dataset statistics

### Step 2: Test the Dataset
```bash
# Test the manual loader
python -c "from src.data.manual_loader import load_manual_fsdd; load_manual_fsdd()"

# Run verification
python verify_setup.py
```

---

## 🔧 **Option 2: Manual Download (DIY)**

### Step 1: Download from GitHub
```bash
# Method A: Using curl
curl -L https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip -o fsdd.zip

# Method B: Using wget
wget https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip -O fsdd.zip

# Method C: Browser download
# Visit: https://github.com/Jakobovski/free-spoken-digit-dataset
# Click "Code" → "Download ZIP"
```

### Step 2: Extract and Organize
```bash
# Create the directory structure
mkdir -p data/fsdd_manual/recordings

# Extract the zip file
unzip fsdd.zip

# Copy the audio files
cp free-spoken-digit-dataset-master/recordings/*.wav data/fsdd_manual/recordings/

# Copy metadata (optional)
cp free-spoken-digit-dataset-master/metadata.py data/fsdd_manual/

# Cleanup
rm -rf free-spoken-digit-dataset-master fsdd.zip
```

### Step 3: Create Simple Loader
Create `src/data/manual_loader.py`:
```python
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, List

def load_manual_fsdd() -> Tuple[List[np.ndarray], List[int]]:
    recordings_dir = Path("data/fsdd_manual/recordings")
    audio_data, labels = [], []
    
    for wav_file in sorted(recordings_dir.glob("*.wav")):
        digit = int(wav_file.stem.split('_')[0])
        audio, _ = librosa.load(wav_file, sr=8000)
        audio_data.append(audio)
        labels.append(digit)
    
    return audio_data, labels
```

---

## 📊 **Dataset Statistics**

| Attribute | Value |
|-----------|-------|
| **Total Files** | ~3,000 audio files |
| **Digits** | 0-9 (10 classes) |
| **Speakers** | 6 speakers |
| **Files per Digit** | ~300 per digit |
| **Audio Format** | WAV, 8kHz, mono |
| **File Size** | ~20MB total |
| **Duration** | 0.1 - 2.0 seconds per file |

### Speakers Information
- **jackson** - Male speaker
- **nicolas** - Male speaker  
- **yweweler** - Male speaker
- **theo** - Male speaker
- **george** - Male speaker
- **lucas** - Male speaker

---

## 🔌 **Integration with Your Project**

### Update Main Dataset Loader
Modify `src/data/dataset.py` to include manual fallback:

```python
def load_fsdd_dataset() -> Tuple[List[np.ndarray], List[int]]:
    # ... existing MTEB and fallback attempts ...
    
    except Exception as final_error:
        print("🔄 Trying manual dataset as final fallback...")
        try:
            from .manual_loader import load_manual_fsdd
            return load_manual_fsdd()
        except Exception as manual_error:
            print(f"❌ Manual dataset also failed: {manual_error}")
            raise Exception("All dataset loading methods failed")
```

### Use in Training Scripts
```python
# In your training code
from src.data.manual_loader import load_manual_fsdd

# Load the dataset
audio_data, labels = load_manual_fsdd()

# Continue with normal training...
from src.data import create_data_loaders
train_loader, test_loader = create_data_loaders(audio_data, labels)
```

---

## ✅ **Verification Commands**

After manual download, run these to verify everything works:

```bash
# 1. Check folder structure
ls -la data/fsdd_manual/recordings/ | head -10

# 2. Count files
find data/fsdd_manual/recordings -name "*.wav" | wc -l

# 3. Test loading
python -c "
from src.data.manual_loader import load_manual_fsdd
audio_data, labels = load_manual_fsdd()
print(f'Loaded {len(audio_data)} files')
print(f'Labels: {min(labels)} to {max(labels)}')
print(f'Sample shape: {audio_data[0].shape}')
"

# 4. Run full verification
python verify_setup.py

# 5. Test training (quick)
python scripts/train.py --epochs 2 --batch_size 16
```

---

## 🎉 **Advantages of Manual Download**

✅ **No dependency issues** - Works without torchcodec/FFmpeg  
✅ **Fast and reliable** - Direct download from GitHub  
✅ **Full control** - You manage the files locally  
✅ **Offline capable** - Works without internet after download  
✅ **Easy debugging** - Can inspect files directly  

---

## 🆘 **Troubleshooting**

### Issue: "Manual dataset not found"
```bash
# Check if download completed
ls -la data/fsdd_manual/recordings/

# Re-run download if needed
python manual_dataset_download.py
```

### Issue: "ImportError: manual_loader"
```bash
# Ensure the loader was created
ls -la src/data/manual_loader.py

# Create manually if needed (see Option 2 above)
```

### Issue: Audio loading errors
```bash
# Check librosa installation
python -c "import librosa; print('✅ Librosa works')"

# Install if needed
conda install -c conda-forge librosa
```

---

## 📋 **Next Steps After Manual Download**

1. **Verify setup**: `python verify_setup.py`
2. **Quick training**: `python scripts/train.py --epochs 5`
3. **Full demo**: `python demo.py --skip-microphone`
4. **Live prediction**: Set up PyAudio for microphone features

The manual download gives you complete control and bypasses all the torchcodec/FFmpeg issues! 🎯 