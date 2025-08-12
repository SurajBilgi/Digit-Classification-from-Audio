# 🚀 Quick Start Guide - Manual Dataset
## Digit Classification from Audio

### ✅ **You now have the dataset downloaded manually!**

This guide shows you how to get started with your manually downloaded FSDD dataset and run the complete digit classification system.

---

## 📊 **Current Status**

✅ **Manual dataset downloaded**: 3,000 audio files  
✅ **Project structure created**: All files organized  
✅ **Demo updated**: Now works with manual dataset  
✅ **Verification script available**: `verify_setup.py`  
✅ **Comprehensive guides available**: Multiple setup options  

---

## 🎯 **Next Steps (Choose Your Path)**

### **Path A: Quick Demo (No Dependencies)**
```bash
# Check what's working
python demo.py --check-setup

# See current status
python verify_setup.py
```

### **Path B: Set Up Environment & Run Full Demo**
```bash
# 1. Create stable environment
conda create -n digit-audio python=3.8 -y
conda activate digit-audio

# 2. Install stable packages
conda install pytorch=1.12.0 torchaudio=0.12.0 cpuonly -c pytorch -y
conda install numpy pandas matplotlib seaborn scikit-learn -y
conda install -c conda-forge librosa -y
pip install datasets==2.10.0 soundfile tqdm jupyter

# 3. Run full demo
python demo.py

# 4. Train a model
python scripts/train.py --epochs 10
```

### **Path C: Quick Environment Setup**
```bash
# Use the provided environment file
conda env create -f environment.yml
conda activate digit-audio

# Test everything
python demo.py
```

---

## 📁 **What's Already Set Up**

### **Dataset Location**
```
data/fsdd_manual/recordings/
├── 0_jackson_0.wav
├── 0_jackson_1.wav
├── 1_jackson_0.wav
└── ... (3,000 total files)
```

### **Key Files Created**
- ✅ `manual_dataset_download.py` - Dataset download script
- ✅ `src/data/manual_loader.py` - Manual dataset loader
- ✅ `verify_setup.py` - Environment verification
- ✅ `demo.py` - Updated with manual dataset support
- ✅ `SETUP_GUIDE.md` - Comprehensive setup instructions
- ✅ `MANUAL_DATASET_GUIDE.md` - Manual dataset guide

### **Integration Features**
- ✅ **Automatic fallback**: Main loader falls back to manual dataset
- ✅ **Error handling**: Clear error messages and solutions
- ✅ **Multiple paths**: Works with or without Hugging Face
- ✅ **Compatibility**: Bypasses torchcodec/FFmpeg issues

---

## 🧪 **Testing Commands**

### **1. Verify Setup**
```bash
# Check environment and dependencies
python verify_setup.py

# Quick setup status
python demo.py --check-setup
```

### **2. Test Manual Dataset Loading**
```bash
# Test the manual loader directly
python -c "from src.data.manual_loader import load_manual_fsdd; load_manual_fsdd()"

# Test with main dataset loader (includes fallbacks)
python -c "from src.data.dataset import load_fsdd_dataset; load_fsdd_dataset()"
```

### **3. Run Demo Components**
```bash
# Run individual components
python demo.py --skip-training --skip-microphone  # Basic test
python demo.py --skip-microphone                  # With training
python demo.py                                    # Full demo
```

### **4. Training Commands**
```bash
# Quick training test
python scripts/train.py --epochs 5 --batch_size 16

# Full training
python scripts/train.py --model_type lightweight --epochs 50

# Different model types
python scripts/train.py --model_type mini --feature_type mel
```

---

## 📈 **Expected Results**

### **Dataset Statistics**
- **Total Files**: 3,000 audio samples
- **Digits**: 0-9 (300 files each)
- **Speakers**: 6 speakers (500 files each)
- **Format**: WAV, 8kHz, mono
- **Duration**: 0.1-2.0 seconds per file

### **Model Performance (Expected)**
| Model | Accuracy | Parameters | Inference Time |
|-------|----------|------------|----------------|
| Lightweight CNN | ~94% | 98K | ~15ms |
| Mini CNN | ~92% | 53K | ~9ms |
| Simple MLP | ~87% | 67K | ~12ms |

---

## 🔧 **Troubleshooting**

### **Issue: "No module named 'librosa'"**
```bash
# Solution: Install missing packages
conda install -c conda-forge librosa -y
# or
pip install librosa
```

### **Issue: "Manual dataset not found"**
```bash
# Check if download completed
ls data/fsdd_manual/recordings/ | wc -l  # Should show 3000

# Re-download if needed
python manual_dataset_download.py
```

### **Issue: Training fails**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Reinstall if needed
conda install pytorch torchaudio cpuonly -c pytorch -y
```

### **Issue: Microphone doesn't work**
```bash
# Install PyAudio
conda install -c anaconda pyaudio -y

# Test without microphone
python demo.py --skip-microphone
```

---

## 🎯 **Recommended Workflow**

### **Day 1: Environment Setup**
```bash
1. conda create -n digit-audio python=3.8 -y
2. conda activate digit-audio
3. conda install pytorch=1.12.0 torchaudio=0.12.0 cpuonly -c pytorch -y
4. conda install numpy pandas matplotlib seaborn scikit-learn -y
5. conda install -c conda-forge librosa -y
6. pip install datasets==2.10.0 soundfile tqdm jupyter
7. python verify_setup.py
```

### **Day 1: First Run**
```bash
1. python demo.py --check-setup  # Verify everything
2. python demo.py               # Full demo
3. python scripts/train.py --epochs 5  # Quick training test
```

### **Day 2: Full Training**
```bash
1. python scripts/train.py --epochs 50  # Full training
2. python src/inference/predictor.py --model_path models/best_model.pth
3. python src/microphone/live_predictor.py --model_path models/best_model.pth --interactive
```

---

## 🎉 **Key Advantages**

✅ **No torchcodec/FFmpeg issues** - Manual dataset bypasses all compatibility problems  
✅ **Works offline** - Dataset is local, no internet required after setup  
✅ **Fast and reliable** - Direct file access, no API dependencies  
✅ **Easy debugging** - You can inspect files directly  
✅ **Multiple fallback options** - System tries multiple approaches  
✅ **Clear error messages** - Specific solutions for each issue  

---

## 📚 **Additional Resources**

- **`SETUP_GUIDE.md`** - Comprehensive environment setup
- **`MANUAL_DATASET_GUIDE.md`** - Detailed manual dataset instructions  
- **`README.md`** - Full project documentation
- **`verify_setup.py`** - Diagnose installation issues
- **`demo.py`** - Test all components

---

## 🚨 **Important Notes**

1. **Environment**: Use Python 3.8-3.10 for best compatibility
2. **PyTorch**: Use stable versions (1.12.0-1.13.0) to avoid torchcodec issues
3. **Dataset**: Manual dataset is fully functional and recommended
4. **Testing**: Always run `verify_setup.py` after environment setup
5. **Updates**: The project automatically detects and uses manual dataset

---

**🎯 Ready to get started? Run these commands:**

```bash
# Quick test
python demo.py --check-setup

# Set up environment and run full demo  
conda create -n digit-audio python=3.8 -y && conda activate digit-audio
conda install pytorch=1.12.0 torchaudio=0.12.0 cpuonly -c pytorch -y
conda install -c conda-forge librosa numpy pandas matplotlib scikit-learn -y
pip install datasets==2.10.0 soundfile tqdm jupyter
python demo.py
```

**Your manual dataset is ready to use! 🎉** 