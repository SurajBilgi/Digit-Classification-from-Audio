# Quick Start Guide

Get up and running with digit classification in 5 minutes!

## 🚀 1-Minute Setup

```bash
# Clone and setup
git clone <repository-url>
cd Digit-Classification-from-Audio

# Install and run demo
python setup.py
```

## 🎯 Quick Demo

```bash
# Run complete system demo
python demo.py

# Skip microphone test if you don't have one
python demo.py --skip-microphone
```

## 🏋️ Train Your First Model

```bash
# Train with default settings (takes ~10 minutes)
python scripts/train.py

# Quick training for testing
python scripts/train.py --epochs 10 --batch_size 16
```

## 🎤 Test Real-time Prediction

```bash
# Interactive microphone session
python src/microphone/live_predictor.py \
    --model_path models/best_model.pth \
    --interactive

# Single prediction
python src/microphone/live_predictor.py \
    --model_path models/best_model.pth
```

## 📊 Expected Results

- **Training Time**: 5-15 minutes on CPU
- **Model Accuracy**: 90-95% on test set
- **Inference Speed**: 10-50ms per prediction
- **Model Size**: 0.2-0.4 MB

## 🛠️ Troubleshooting

**"No module named 'datasets'"**
```bash
pip install datasets
```

**"Audio device not found"**
```bash
# List available devices
python -c "
from src.microphone.live_predictor import LiveDigitPredictor
with LiveDigitPredictor('models/demo_model.pth') as p:
    p.print_audio_devices()
"
```

**Poor prediction accuracy**
- Speak clearly and loudly
- Reduce background noise
- Ensure microphone is working: `python -c "import pyaudio; print(pyaudio.PyAudio().get_device_count())"`

## 🎨 Try Different Models

```bash
# Fast but less accurate
python scripts/train.py --model_type mini

# Slow but more accurate  
python scripts/train.py --model_type lightweight

# Different audio features
python scripts/train.py --feature_type mel
python scripts/train.py --feature_type spectrogram
```

## 📈 Next Steps

1. **Experiment**: Try different model architectures and features
2. **Optimize**: Use GPU training with CUDA
3. **Deploy**: Export model for production use
4. **Extend**: Add noise robustness or support for more digits

---

For detailed documentation, see [README.md](README.md) 