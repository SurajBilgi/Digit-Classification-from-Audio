# Complete Setup Guide for Digit Classification from Audio

## 🐍 Python Version Requirement
**Required Python Version: 3.8 - 3.11** (Recommended: Python 3.9)

## 🚀 Quick Setup with Anaconda

### 1. Create and Activate Conda Environment

```bash
# Create new environment with Python 3.9
conda create -n digit-audio python=3.9 -y

# Activate the environment
conda activate digit-audio

# Verify Python version
python --version  # Should show Python 3.9.x
```

### 2. Install Core Dependencies via Conda

```bash
# Install PyTorch (CPU version - adjust for your system)
conda install pytorch torchaudio cpuonly -c pytorch -y

# For GPU support (if you have CUDA):
# conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install scientific computing packages
conda install numpy pandas matplotlib seaborn scikit-learn -y

# Install audio processing
conda install -c conda-forge librosa -y
```

### 3. Install Remaining Dependencies via Pip

```bash
# Install the remaining packages
pip install datasets soundfile pyaudio tqdm jupyter

# Verify installation
python -c "import torch, librosa, datasets; print('✅ All packages installed successfully!')"
```

### 4. Clone and Setup Project

```bash
# Clone the repository
git clone <your-repo-url>
cd Digit-Classification-from-Audio

# Run the demo to verify everything works
python demo.py --skip-microphone  # Skip microphone if no audio device
```

## 🔧 Alternative Setup Methods

### Method 1: All via Pip (if conda is not available)

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (macOS/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Method 2: Using requirements.txt with conda

```bash
# After creating and activating conda environment
conda activate digit-audio

# Install from requirements file
pip install -r requirements.txt
```

## 🎵 Audio Setup (PyAudio Installation Issues)

PyAudio can be tricky to install. Here are platform-specific solutions:

### Windows:
```bash
# Option 1: Install via conda
conda install -c anaconda pyaudio -y

# Option 2: Install pre-compiled wheel
pip install pipwin
pipwin install pyaudio
```

### macOS:
```bash
# Install PortAudio first
brew install portaudio

# Then install PyAudio
pip install pyaudio
```

### Linux (Ubuntu/Debian):
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio -y

# Install PyAudio
pip install pyaudio
```

## 🧪 Verify Installation

Run this verification script:

```bash
python -c "
import sys
print(f'Python version: {sys.version}')

packages = ['torch', 'torchaudio', 'librosa', 'numpy', 'pandas', 
           'matplotlib', 'sklearn', 'datasets', 'soundfile']

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError as e:
        print(f'❌ {pkg}: {e}')

# Test PyAudio separately (might not be available on all systems)
try:
    import pyaudio
    print('✅ pyaudio (microphone support available)')
except ImportError:
    print('⚠️  pyaudio not installed (microphone features disabled)')

# Test dataset loading
try:
    from datasets import load_dataset
    print('Testing MTEB dataset access...')
    dataset = load_dataset('mteb/free-spoken-digit-dataset', split='train[:5]')
    print('✅ MTEB dataset access successful')
except Exception as e:
    print(f'⚠️  MTEB dataset test failed: {e}')
"
```

## 🚀 Quick Start Commands

After successful installation:

```bash
# 1. Run complete demo
python demo.py

# 2. Train a model (quick test)
python scripts/train.py --epochs 5 --batch_size 16

# 3. Test inference (after training)
python src/inference/predictor.py --model_path models/best_model.pth

# 4. Real-time prediction (if microphone works)
python src/microphone/live_predictor.py --model_path models/best_model.pth --interactive
```

## 🐛 Common Issues and Solutions

### Issue 1: "No module named 'datasets'"
```bash
pip install datasets
```

### Issue 2: TorchCodec/FFmpeg Error (Most Common)
If you see "Could not load libtorchcodec" or FFmpeg-related errors:

**Solution A: Install FFmpeg**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/ and add to PATH
```

**Solution B: Use Compatible PyTorch Version**
```bash
# Downgrade to compatible versions
conda install pytorch=1.13.0 torchaudio=0.13.0 -c pytorch -y

# Or try PyTorch 2.0 series
conda install pytorch=2.0.1 torchaudio=2.0.2 -c pytorch -y
```

**Solution C: Alternative Environment Setup**
```bash
# Remove current environment
conda env remove -n digit-audio

# Create with Python 3.8 (more stable)
conda create -n digit-audio python=3.8 -y
conda activate digit-audio

# Install older, more stable versions
conda install pytorch=1.12.0 torchaudio=0.12.0 cpuonly -c pytorch -y
conda install numpy pandas matplotlib seaborn scikit-learn -y
conda install -c conda-forge librosa -y
pip install datasets==2.10.0 soundfile pyaudio tqdm jupyter
```

### Issue 3: MTEB Dataset Loading Fails
```bash
# Test dataset access
python -c "from datasets import load_dataset; dataset = load_dataset('mteb/free-spoken-digit-dataset'); print('Dataset loaded successfully!')"

# If this fails, the code automatically falls back to alternative sources
# Check internet connection and try:
pip install --upgrade datasets
```

### Issue 4: PyTorch installation fails
```bash
# Use conda instead of pip for PyTorch
conda install pytorch torchaudio -c pytorch
```

### Issue 5: "Microsoft Visual C++ 14.0 is required" (Windows)
```bash
# Install Visual Studio Build Tools or use conda
conda install -c anaconda pyaudio
```

### Issue 6: "Failed building wheel for pyaudio" (macOS)
```bash
brew install portaudio
export CPPFLAGS=-I/opt/homebrew/include
export LDFLAGS=-L/opt/homebrew/lib
pip install pyaudio
```

### Issue 7: Audio device not found
```bash
# List audio devices
python -c "
try:
    import pyaudio
    p = pyaudio.PyAudio()
    print(f'Found {p.get_device_count()} audio devices')
    p.terminate()
except:
    print('PyAudio not available - microphone features disabled')
"
```

### Issue 8: Dataset Authentication (if required)
```bash
# Login to Hugging Face if needed
pip install huggingface_hub
python -c "from huggingface_hub import login; login()"
```

## 🛠️ Alternative Installation Methods

### Method 1: Stable/Conservative Setup (Recommended for issues)
```bash
# Create environment with older Python
conda create -n digit-audio python=3.8 -y
conda activate digit-audio

# Install stable versions
conda install pytorch=1.12.0 torchaudio=0.12.0 cpuonly -c pytorch -y
conda install numpy=1.21.0 pandas=1.4.0 matplotlib=3.5.0 seaborn=0.11.0 scikit-learn=1.1.0 -y
conda install -c conda-forge librosa=0.9.2 -y
pip install datasets==2.10.0 soundfile==0.12.1 tqdm==4.60.0 jupyter==1.0.0

# Test without PyAudio first
python demo.py --skip-microphone

# Install PyAudio separately if needed
conda install -c anaconda pyaudio -y
```

### Method 2: CPU-Only Environment (No CUDA issues)
```bash
conda create -n digit-audio python=3.9 -y
conda activate digit-audio

# Explicitly install CPU-only versions
conda install pytorch torchaudio cpuonly -c pytorch -y
conda install numpy pandas matplotlib seaborn scikit-learn -y
conda install -c conda-forge librosa -y
pip install datasets soundfile tqdm jupyter

# Skip PyAudio if microphone not needed
```

### Method 3: Minimal Installation
```bash
# Only essential packages
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install librosa datasets numpy scikit-learn pandas matplotlib
pip install soundfile tqdm

# Test core functionality
python -c "
import torch, librosa, datasets
print('✅ Core packages working')
from datasets import load_dataset
try:
    ds = load_dataset('Matthijs/free-spoken-digit-dataset', split='train[:5]')
    print('✅ Dataset loading working')
except:
    print('⚠️ Dataset issue - check internet connection')
"
```

## 📋 Environment Export/Import

### Export your working environment:
```bash
# Export conda environment
conda env export > environment.yml

# Export pip requirements
pip freeze > requirements-frozen.txt
```

### Import environment on another machine:
```bash
# Import conda environment
conda env create -f environment.yml

# Or use pip
pip install -r requirements-frozen.txt
```

## 🎯 Project Structure After Setup

```
Digit-Classification-from-Audio/
├── src/                    # Source code
├── scripts/               # Training scripts
├── models/                # Saved models (created after training)
├── data/                  # Data storage (created automatically)
├── requirements.txt       # Python dependencies
├── demo.py               # Complete system demo
├── setup.py              # Automated setup script
└── README.md             # Main documentation
```

## ✅ Verify Everything is Working

```bash
# Run the complete verification
python setup.py

# Or run individual tests
python demo.py --skip-training --skip-microphone  # Basic test
python demo.py                                    # Full test
```

## 🆘 Get Help

If you encounter issues:

1. **Check Python version**: `python --version` (should be 3.8-3.11)
2. **Check conda environment**: `conda info --envs`
3. **Run verification script** (provided above)
4. **Check specific error messages** and refer to solutions above

## 🔄 Clean Installation (if needed)

```bash
# Remove conda environment
conda env remove -n digit-audio

# Start fresh
conda create -n digit-audio python=3.9 -y
conda activate digit-audio

# Follow setup steps again
```

---

**Recommended Setup Path:**
1. Use **Anaconda/Miniconda** with **Python 3.9**
2. Install **PyTorch via conda**
3. Install **other packages via pip**
4. **Test with demo.py** 