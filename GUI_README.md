# 🎤 AI Digit Classifier GUI

## Beautiful, Modern Interface for Real-Time Audio Digit Recognition

A comprehensive GUI application featuring all the functionality you requested with animations, professional styling, and real-time visualization.

---

## 🚀 **Features**

### **Core Functionality**
✅ **[1] Predict Single Digit** - Record once, get prediction  
✅ **[2] Continuous Prediction** - Live streaming recognition  
✅ **[3] Prediction History** - View all past predictions with timestamps  
✅ **[4] Performance Statistics** - Detailed analytics and charts  
✅ **[5] Settings Panel** - Confidence threshold, recording duration, audio device  
✅ **[6] Quit Option** - Clean exit with confirmation  

### **Additional Features**
🎨 **Modern UI Design** - Professional color scheme and animations  
📊 **Real-Time Visualizations** - Live waveform display  
📈 **Performance Charts** - Confidence over time, digit distribution  
⚙️ **Advanced Settings** - Audio device selection, customizable thresholds  
💾 **Model Management** - Load different models via file dialog  
📋 **Session Management** - Clear history, export data  

---

## 🎯 **Quick Start**

### **1. Run the GUI**
```bash
# Activate your environment
conda activate audioLLM

# Launch the GUI (simple launcher)
python run_gui.py

# Or directly
python gui_app.py --model_path models/best_model.pth
```

### **2. First Time Setup**
1. **Load Model**: If no model is found, click "📁 Load Model"
2. **Check Settings**: Click "⚙️ Settings" to configure recording duration
3. **Test Single Prediction**: Click "🎤 Predict Single Digit"
4. **Start Continuous Mode**: Click "🔄 Start Continuous" for live recognition

---

## 🎮 **How to Use Each Feature**

### **🎤 [1] Predict Single Digit**
- Click the blue button
- Speak a digit (0-9) clearly
- View prediction with confidence score
- See result in the large display

### **🔄 [2] Continuous Prediction**
- Click green "Start Continuous" button
- Speak digits continuously
- Button changes to "Stop Continuous"
- Real-time predictions appear automatically

### **📋 [3] View History**
- Click orange "View History" button
- See table with timestamps, digits, confidence
- Filter by high/low confidence
- Clear history option available

### **📊 [4] Performance Stats**
- Click purple "Performance Stats" button
- View 3 tabs:
  - **Overall**: Timing, accuracy metrics
  - **Accuracy**: Detailed analysis
  - **Charts**: Visual graphs and distributions

### **⚙️ [5] Settings**
- Click gray "Settings" button
- Adjust:
  - **Confidence Threshold** (0.0-1.0)
  - **Record Duration** (0.5-3.0 seconds)
  - **Audio Device** (from dropdown)
- Save changes

### **❌ [6] Quit**
- Click red "Quit" button
- Confirmation dialog appears
- Clean shutdown of all threads

---

## 🎨 **GUI Layout**

```
┌─────────────────────────────────────────────────────────────┐
│                  🎤 AI Digit Classifier                    │
│              Real-time Audio Digit Recognition              │
├─────────────┬───────────────────────────┬─────────────────── │
│ 🎛️ Control  │     📈 Live Visual        │   📊 Statistics   │
│   Panel     │                          │                   │
│             │    ┌─────────────┐        │  Performance:     │
│ ● Ready     │    │      ?      │        │  - Speed metrics  │
│             │    │             │        │  - Session stats  │
│ 🎤 Single   │    └─────────────┘        │  - Settings       │
│ 🔄 Continuous│                          │  - Recent preds   │
│ 📋 History   │    Confidence: ████      │                   │
│ 📊 Stats     │                          │  🕒 Recent:       │
│ ⚙️ Settings │    🌊 Waveform ~~~~       │  • 7 (0.95)      │
│ 📁 Load Model│                          │  • 3 (0.82)      │
│ ❌ Quit     │                          │  • 1 (0.78)      │
└─────────────┴───────────────────────────┴───────────────────┘
│                Model: best_model.pth                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 **Visual Indicators**

### **Status Colors**
- 🔵 **Blue**: Ready/Info
- 🟢 **Green**: Success/High confidence  
- 🟡 **Yellow**: Warning/Medium confidence
- 🔴 **Red**: Error/Low confidence/Recording

### **Prediction Display**
- **Green Background**: High confidence (≥ threshold)
- **Red Background**: Low confidence (< threshold)
- **Large Font**: Easy to read predicted digit

### **Confidence Bar**
- **Animated Progress Bar**: Visual confidence indicator
- **Percentage Display**: Exact confidence value

---

## 🔧 **Customization Options**

### **Settings Panel**
- **Confidence Threshold**: 0.0 to 1.0 (default: 0.5)
- **Record Duration**: 0.5 to 3.0 seconds (default: 1.0)
- **Audio Device**: Select from available microphones

### **Model Loading**
- Load any `.pth` model file
- Automatic configuration detection
- Model info displayed in footer

---

## 🎪 **Performance Features**

### **Real-Time Metrics**
- Inference speed (ms)
- Predictions per second
- Preprocessing time
- Total predictions

### **Session Analytics**
- High vs low confidence predictions
- Digit distribution analysis
- Confidence trends over time

### **Visual Charts**
- **Confidence Timeline**: Track confidence over session
- **Digit Histogram**: See which digits predicted most
- **Performance Graphs**: Speed and accuracy metrics

---

## 🛠️ **Troubleshooting**

### **Model Not Loading**
```
Error: Model file not found
Solution: Click "📁 Load Model" and select your .pth file
```

### **No Audio Input**
```
Error: No microphone detected
Solution: Check Settings → Audio Device, select correct microphone
```

### **Low Accuracy**
```
Issue: Poor predictions
Solution: 
1. Speak clearly and slowly
2. Adjust Settings → Record Duration
3. Train model with more epochs
```

### **Import Errors**
```
Error: Module not found
Solution: Activate environment: conda activate audioLLM
```

---

## 🎉 **Pro Tips**

1. **Best Recording**: Speak clearly, avoid background noise
2. **Optimal Settings**: Start with 1.0s duration, 0.5 confidence threshold
3. **Performance**: Use continuous mode for real-time experience
4. **Analytics**: Check history to see prediction patterns
5. **Customization**: Adjust settings based on your voice and environment

---

## 🚀 **Launch Commands**

```bash
# Basic launch
python run_gui.py

# With specific model
python gui_app.py --model_path path/to/your/model.pth

# With environment activation
conda activate audioLLM && python run_gui.py
```

---

## 🎯 **Ready to Use!**

Your GUI includes everything you requested:
- ✅ All 6 core features  
- ✅ Beautiful modern interface
- ✅ Real-time animations  
- ✅ Professional styling
- ✅ Comprehensive functionality

**Launch it now and start recognizing digits!** 🎤🚀 