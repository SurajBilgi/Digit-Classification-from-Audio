#!/usr/bin/env python3
"""
Digit Classification GUI Application
==================================

A beautiful, modern GUI for real-time digit classification from audio.
Features microphone integration, prediction history, performance stats, and more.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import json

# Add src to path
sys.path.append("src")

try:
    from microphone.live_predictor import LiveDigitPredictor
    from inference.predictor import DigitPredictor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to activate your environment: conda activate audioLLM")
    sys.exit(1)


class ModernButton(tk.Button):
    """Custom modern-looking button with hover effects."""

    def __init__(
        self,
        parent,
        text="",
        command=None,
        bg_color="#4CAF50",
        hover_color="#45a049",
        text_color="white",
        **kwargs,
    ):
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=text_color,
            relief="flat",
            font=("Arial", 10, "bold"),
            cursor="hand2",
            padx=20,
            pady=10,
            **kwargs,
        )

        self.bg_color = bg_color
        self.hover_color = hover_color

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=self.hover_color)

    def on_leave(self, e):
        self.config(bg=self.bg_color)


class WaveformWidget(tk.Frame):
    """Real-time waveform display widget."""

    def __init__(self, parent, width=400, height=100):
        super().__init__(parent, bg="#2c3e50")

        self.figure = Figure(
            figsize=(width / 100, height / 100), facecolor="#2c3e50", edgecolor="none"
        )
        self.ax = self.figure.add_subplot(111, facecolor="#34495e")
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-1, 1)
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize empty waveform
        (self.line,) = self.ax.plot([], [], color="#3498db", linewidth=2)

    def update_waveform(self, audio_data):
        """Update waveform display with new audio data."""
        if len(audio_data) > 0:
            # Downsample for display
            step = max(1, len(audio_data) // 100)
            x_data = np.arange(0, 100)
            y_data = audio_data[::step][:100]

            if len(y_data) < 100:
                y_data = np.pad(y_data, (0, 100 - len(y_data)))

            self.line.set_data(x_data, y_data)
            self.ax.set_ylim(min(y_data) - 0.1, max(y_data) + 0.1)
            self.canvas.draw()


class DigitClassifierGUI:
    """Main GUI application for digit classification."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.predictor = None
        self.live_predictor = None
        self.is_recording = False
        self.is_continuous = False

        # Data storage
        self.prediction_history = []
        self.performance_stats = []
        self.settings = {
            "confidence_threshold": 0.5,
            "record_duration": 1.0,
            "device_index": None,
        }

        # Threading
        self.prediction_queue = queue.Queue()
        self.recording_thread = None

        # Initialize GUI
        self.setup_gui()
        self.load_model()

    def setup_gui(self):
        """Setup the main GUI interface."""
        self.root = tk.Tk()
        self.root.title("🎤 Digit Classifier - AI Audio Recognition")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        # Configure style
        self.setup_styles()

        # Create main layout
        self.create_header()
        self.create_main_content()
        self.create_footer()

        # Start update loop
        self.update_gui()

    def setup_styles(self):
        """Setup modern GUI styles."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure colors
        self.colors = {
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "warning": "#f39c12",
            "danger": "#e74c3c",
            "dark": "#2c3e50",
            "light": "#ecf0f1",
            "text": "#2c3e50",
        }

        # Configure ttk styles
        self.style.configure(
            "Title.TLabel",
            font=("Arial", 24, "bold"),
            foreground=self.colors["light"],
            background=self.colors["dark"],
        )

        self.style.configure(
            "Subtitle.TLabel",
            font=("Arial", 12),
            foreground=self.colors["light"],
            background=self.colors["dark"],
        )

    def create_header(self):
        """Create the application header."""
        header_frame = tk.Frame(self.root, bg=self.colors["dark"], height=100)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        header_frame.pack_propagate(False)

        # Title
        title_label = ttk.Label(
            header_frame, text="🎤 AI Digit Classifier", style="Title.TLabel"
        )
        title_label.pack(pady=10)

        # Subtitle
        subtitle_label = ttk.Label(
            header_frame,
            text="Real-time Audio Digit Recognition System",
            style="Subtitle.TLabel",
        )
        subtitle_label.pack()

    def create_main_content(self):
        """Create the main content area."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors["dark"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Controls
        self.create_control_panel(main_frame)

        # Center panel - Visualization
        self.create_visualization_panel(main_frame)

        # Right panel - Statistics
        self.create_stats_panel(main_frame)

    def create_control_panel(self, parent):
        """Create the control panel with all features."""
        control_frame = tk.Frame(parent, bg=self.colors["light"], relief="raised", bd=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Panel title
        tk.Label(
            control_frame,
            text="🎛️ Control Panel",
            font=("Arial", 16, "bold"),
            bg=self.colors["light"],
            fg=self.colors["text"],
        ).pack(pady=20)

        # Status indicator
        self.status_frame = tk.Frame(control_frame, bg=self.colors["light"])
        self.status_frame.pack(pady=10)

        self.status_label = tk.Label(
            self.status_frame,
            text="● Ready",
            font=("Arial", 12, "bold"),
            fg=self.colors["secondary"],
            bg=self.colors["light"],
        )
        self.status_label.pack()

        # Main control buttons
        button_frame = tk.Frame(control_frame, bg=self.colors["light"])
        button_frame.pack(pady=20, padx=20, fill=tk.X)

        # Button 1: Predict Single Digit
        self.btn_single = ModernButton(
            button_frame,
            text="🎤 Predict Single Digit",
            command=self.predict_single_digit,
            bg_color=self.colors["primary"],
            hover_color="#2980b9",
        )
        self.btn_single.pack(pady=5, fill=tk.X)

        # Button 2: Continuous Prediction
        self.btn_continuous = ModernButton(
            button_frame,
            text="🔄 Start Continuous",
            command=self.toggle_continuous_prediction,
            bg_color=self.colors["secondary"],
            hover_color="#27ae60",
        )
        self.btn_continuous.pack(pady=5, fill=tk.X)

        # Button 3: Show History
        self.btn_history = ModernButton(
            button_frame,
            text="📋 View History",
            command=self.show_prediction_history,
            bg_color=self.colors["warning"],
            hover_color="#e67e22",
        )
        self.btn_history.pack(pady=5, fill=tk.X)

        # Button 4: Performance Stats
        self.btn_stats = ModernButton(
            button_frame,
            text="📊 Performance Stats",
            command=self.show_performance_stats,
            bg_color="#9b59b6",
            hover_color="#8e44ad",
        )
        self.btn_stats.pack(pady=5, fill=tk.X)

        # Button 5: Settings
        self.btn_settings = ModernButton(
            button_frame,
            text="⚙️ Settings",
            command=self.open_settings,
            bg_color="#34495e",
            hover_color="#2c3e50",
        )
        self.btn_settings.pack(pady=5, fill=tk.X)

        # Load Model Button
        self.btn_load_model = ModernButton(
            button_frame,
            text="📁 Load Model",
            command=self.load_model_dialog,
            bg_color="#16a085",
            hover_color="#138d75",
        )
        self.btn_load_model.pack(pady=5, fill=tk.X)

        # Quit Button
        self.btn_quit = ModernButton(
            button_frame,
            text="❌ Quit",
            command=self.quit_application,
            bg_color=self.colors["danger"],
            hover_color="#c0392b",
        )
        self.btn_quit.pack(pady=(20, 5), fill=tk.X)

    def create_visualization_panel(self, parent):
        """Create the visualization panel."""
        viz_frame = tk.Frame(parent, bg=self.colors["light"], relief="raised", bd=2)
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Title
        tk.Label(
            viz_frame,
            text="📈 Live Visualization",
            font=("Arial", 16, "bold"),
            bg=self.colors["light"],
            fg=self.colors["text"],
        ).pack(pady=20)

        # Prediction display
        self.prediction_frame = tk.Frame(viz_frame, bg=self.colors["light"])
        self.prediction_frame.pack(pady=20)

        # Large prediction display
        self.prediction_display = tk.Label(
            self.prediction_frame,
            text="?",
            font=("Arial", 72, "bold"),
            bg="#ffffff",
            fg=self.colors["primary"],
            relief="raised",
            bd=3,
            width=3,
            height=2,
        )
        self.prediction_display.pack(pady=10)

        # Confidence bar
        self.confidence_frame = tk.Frame(viz_frame, bg=self.colors["light"])
        self.confidence_frame.pack(pady=10)

        tk.Label(
            self.confidence_frame,
            text="Confidence:",
            font=("Arial", 12),
            bg=self.colors["light"],
        ).pack()

        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(
            self.confidence_frame,
            variable=self.confidence_var,
            maximum=100,
            length=300,
            mode="determinate",
        )
        self.confidence_bar.pack(pady=5)

        self.confidence_label = tk.Label(
            self.confidence_frame,
            text="0.0%",
            font=("Arial", 12, "bold"),
            bg=self.colors["light"],
        )
        self.confidence_label.pack()

        # Waveform display
        waveform_frame = tk.Frame(viz_frame, bg=self.colors["light"])
        waveform_frame.pack(pady=20, fill=tk.X, padx=20)

        tk.Label(
            waveform_frame,
            text="🌊 Audio Waveform",
            font=("Arial", 12, "bold"),
            bg=self.colors["light"],
        ).pack()

        self.waveform_widget = WaveformWidget(waveform_frame, width=400, height=100)
        self.waveform_widget.pack(pady=10, fill=tk.X)

    def create_stats_panel(self, parent):
        """Create the statistics panel."""
        stats_frame = tk.Frame(
            parent, bg=self.colors["light"], relief="raised", bd=2, width=300
        )
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y)
        stats_frame.pack_propagate(False)

        # Title
        tk.Label(
            stats_frame,
            text="📊 Statistics",
            font=("Arial", 16, "bold"),
            bg=self.colors["light"],
            fg=self.colors["text"],
        ).pack(pady=20)

        # Statistics display
        self.stats_text = tk.Text(
            stats_frame,
            height=20,
            width=35,
            font=("Consolas", 10),
            state="disabled",
            bg="#f8f9fa",
            fg=self.colors["text"],
        )
        self.stats_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Recent predictions
        tk.Label(
            stats_frame,
            text="🕒 Recent Predictions",
            font=("Arial", 12, "bold"),
            bg=self.colors["light"],
        ).pack(pady=(10, 5))

        self.recent_frame = tk.Frame(stats_frame, bg=self.colors["light"])
        self.recent_frame.pack(pady=5, fill=tk.X, padx=10)

        self.recent_predictions = []
        self.update_recent_display()

    def create_footer(self):
        """Create the application footer."""
        footer_frame = tk.Frame(self.root, bg=self.colors["dark"], height=40)
        footer_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
        footer_frame.pack_propagate(False)

        # Model info
        self.model_info_label = ttk.Label(
            footer_frame, text="Model: Not loaded", style="Subtitle.TLabel"
        )
        self.model_info_label.pack(side=tk.LEFT, pady=10)

        # Version info
        version_label = ttk.Label(
            footer_frame, text="v1.0 | AI Digit Classifier", style="Subtitle.TLabel"
        )
        version_label.pack(side=tk.RIGHT, pady=10)

    def load_model(self):
        """Load the digit classification model."""
        try:
            if not os.path.exists(self.model_path):
                messagebox.showerror(
                    "Error", f"Model file not found: {self.model_path}"
                )
                return False

            self.predictor = DigitPredictor(self.model_path)
            self.live_predictor = LiveDigitPredictor(
                self.model_path,
                confidence_threshold=self.settings["confidence_threshold"],
                record_seconds=self.settings["record_duration"],
            )

            # Update model info
            self.model_info_label.config(
                text=f"Model: {os.path.basename(self.model_path)}"
            )
            self.update_status("Model loaded successfully", "success")

            # Update stats
            self.update_stats_display()

            return True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return False

    def load_model_dialog(self):
        """Open file dialog to load a model."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
        )

        if file_path:
            self.model_path = file_path
            self.load_model()

    def predict_single_digit(self):
        """Predict a single digit from microphone."""
        if not self.live_predictor:
            messagebox.showerror("Error", "Please load a model first")
            return

        if self.is_recording:
            messagebox.showwarning("Warning", "Already recording!")
            return

        # Start prediction in separate thread
        self.recording_thread = threading.Thread(target=self._single_prediction_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def _single_prediction_worker(self):
        """Worker thread for single prediction."""
        try:
            self.is_recording = True
            self.update_status("🎤 Recording...", "recording")

            # Record and predict
            digit, confidence, audio_data = self.live_predictor.predict_from_microphone(
                self.settings["device_index"]
            )

            # Update GUI
            self.prediction_queue.put(
                {
                    "type": "single_prediction",
                    "digit": digit,
                    "confidence": confidence,
                    "audio": audio_data,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            self.prediction_queue.put({"type": "error", "message": str(e)})
        finally:
            self.is_recording = False

    def toggle_continuous_prediction(self):
        """Toggle continuous prediction mode."""
        if not self.live_predictor:
            messagebox.showerror("Error", "Please load a model first")
            return

        if self.is_continuous:
            self.stop_continuous_prediction()
        else:
            self.start_continuous_prediction()

    def start_continuous_prediction(self):
        """Start continuous prediction mode."""
        self.is_continuous = True
        self.btn_continuous.config(text="⏹️ Stop Continuous")
        self.update_status("🔄 Continuous mode active", "recording")

        # Start continuous thread
        self.recording_thread = threading.Thread(
            target=self._continuous_prediction_worker
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def stop_continuous_prediction(self):
        """Stop continuous prediction mode."""
        self.is_continuous = False
        self.btn_continuous.config(text="🔄 Start Continuous")
        self.update_status("⏹️ Continuous mode stopped", "success")

    def _continuous_prediction_worker(self):
        """Worker thread for continuous prediction."""
        while self.is_continuous:
            try:
                if not self.is_recording:
                    self.is_recording = True

                    # Record and predict
                    digit, confidence, audio_data = (
                        self.live_predictor.predict_from_microphone(
                            self.settings["device_index"]
                        )
                    )

                    # Update GUI
                    self.prediction_queue.put(
                        {
                            "type": "continuous_prediction",
                            "digit": digit,
                            "confidence": confidence,
                            "audio": audio_data,
                            "timestamp": time.time(),
                        }
                    )

                    self.is_recording = False

                    # Brief pause between predictions
                    time.sleep(0.5)

            except Exception as e:
                self.prediction_queue.put({"type": "error", "message": str(e)})
                break

    def show_prediction_history(self):
        """Show prediction history in a new window."""
        history_window = tk.Toplevel(self.root)
        history_window.title("📋 Prediction History")
        history_window.geometry("600x400")
        history_window.configure(bg=self.colors["light"])

        # Title
        tk.Label(
            history_window,
            text="📋 Prediction History",
            font=("Arial", 16, "bold"),
            bg=self.colors["light"],
        ).pack(pady=20)

        # History table
        columns = ("Time", "Digit", "Confidence", "Status")
        tree = ttk.Treeview(history_window, columns=columns, show="headings", height=15)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)

        # Populate data
        for i, pred in enumerate(reversed(self.prediction_history[-50:])):  # Last 50
            time_str = time.strftime("%H:%M:%S", time.localtime(pred["timestamp"]))
            status = (
                "✓ High"
                if pred["confidence"] >= self.settings["confidence_threshold"]
                else "⚠ Low"
            )
            tree.insert(
                "",
                "end",
                values=(time_str, pred["digit"], f"{pred['confidence']:.3f}", status),
            )

        tree.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Clear history button
        ModernButton(
            history_window,
            text="🗑️ Clear History",
            command=lambda: self.clear_history(history_window),
            bg_color=self.colors["danger"],
        ).pack(pady=10)

    def clear_history(self, window):
        """Clear prediction history."""
        result = messagebox.askyesno("Confirm", "Clear all prediction history?")
        if result:
            self.prediction_history.clear()
            self.update_recent_display()
            window.destroy()

    def show_performance_stats(self):
        """Show detailed performance statistics."""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 Performance Statistics")
        stats_window.geometry("800x600")
        stats_window.configure(bg=self.colors["light"])

        # Title
        tk.Label(
            stats_window,
            text="📊 Performance Statistics",
            font=("Arial", 16, "bold"),
            bg=self.colors["light"],
        ).pack(pady=20)

        # Create notebook for tabs
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Overall stats tab
        self.create_overall_stats_tab(notebook)

        # Accuracy analysis tab
        self.create_accuracy_stats_tab(notebook)

        # Performance charts tab
        self.create_charts_tab(notebook)

    def create_overall_stats_tab(self, notebook):
        """Create overall statistics tab."""
        frame = tk.Frame(notebook, bg=self.colors["light"])
        notebook.add(frame, text="📈 Overall")

        if self.predictor:
            pred_stats = self.predictor.get_performance_stats()

            stats_text = f"""
🎯 Prediction Statistics:
  Total Predictions: {len(self.prediction_history)}
  Average Inference Time: {pred_stats.get('avg_inference_time_ms', 0):.2f} ms
  Predictions Per Second: {pred_stats.get('predictions_per_second', 0):.1f}

🎪 Confidence Analysis:
  High Confidence (≥{self.settings['confidence_threshold']}): {sum(1 for p in self.prediction_history if p['confidence'] >= self.settings['confidence_threshold'])}
  Low Confidence: {sum(1 for p in self.prediction_history if p['confidence'] < self.settings['confidence_threshold'])}

📊 Digit Distribution:
"""

            # Calculate digit distribution
            digit_counts = {}
            for pred in self.prediction_history:
                digit = pred["digit"]
                digit_counts[digit] = digit_counts.get(digit, 0) + 1

            for digit in range(10):
                count = digit_counts.get(digit, 0)
                stats_text += f"  Digit {digit}: {count} predictions\n"

        else:
            stats_text = "No statistics available. Please load a model first."

        text_widget = tk.Text(
            frame,
            font=("Consolas", 12),
            state="normal",
            bg="#f8f9fa",
            fg=self.colors["text"],
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert("1.0", stats_text)
        text_widget.config(state="disabled")

    def create_accuracy_stats_tab(self, notebook):
        """Create accuracy analysis tab."""
        frame = tk.Frame(notebook, bg=self.colors["light"])
        notebook.add(frame, text="🎯 Accuracy")

        tk.Label(
            frame,
            text="Accuracy analysis would go here",
            font=("Arial", 14),
            bg=self.colors["light"],
        ).pack(pady=50)

    def create_charts_tab(self, notebook):
        """Create performance charts tab."""
        frame = tk.Frame(notebook, bg=self.colors["light"])
        notebook.add(frame, text="📊 Charts")

        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), facecolor=self.colors["light"])

        # Confidence over time chart
        ax1 = fig.add_subplot(211)
        if self.prediction_history:
            times = [
                p["timestamp"] - self.prediction_history[0]["timestamp"]
                for p in self.prediction_history
            ]
            confidences = [p["confidence"] for p in self.prediction_history]
            ax1.plot(times, confidences, "b-", linewidth=2)
            ax1.set_title("Confidence Over Time")
            ax1.set_ylabel("Confidence")
            ax1.grid(True, alpha=0.3)

        # Digit distribution chart
        ax2 = fig.add_subplot(212)
        if self.prediction_history:
            digits = [p["digit"] for p in self.prediction_history]
            ax2.hist(digits, bins=range(11), alpha=0.7, color="green")
            ax2.set_title("Digit Distribution")
            ax2.set_xlabel("Digit")
            ax2.set_ylabel("Count")
            ax2.set_xticks(range(10))

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def open_settings(self):
        """Open settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Settings")
        settings_window.geometry("500x400")
        settings_window.configure(bg=self.colors["light"])

        # Title
        tk.Label(
            settings_window,
            text="⚙️ Application Settings",
            font=("Arial", 16, "bold"),
            bg=self.colors["light"],
        ).pack(pady=20)

        # Settings frame
        settings_frame = tk.Frame(settings_window, bg=self.colors["light"])
        settings_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)

        # Confidence threshold
        tk.Label(
            settings_frame,
            text="Confidence Threshold:",
            font=("Arial", 12),
            bg=self.colors["light"],
        ).grid(row=0, column=0, sticky="w", pady=10)

        conf_var = tk.DoubleVar(value=self.settings["confidence_threshold"])
        conf_scale = tk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            variable=conf_var,
            length=200,
        )
        conf_scale.grid(row=0, column=1, pady=10, padx=10)

        # Record duration
        tk.Label(
            settings_frame,
            text="Record Duration (seconds):",
            font=("Arial", 12),
            bg=self.colors["light"],
        ).grid(row=1, column=0, sticky="w", pady=10)

        duration_var = tk.DoubleVar(value=self.settings["record_duration"])
        duration_scale = tk.Scale(
            settings_frame,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            variable=duration_var,
            length=200,
        )
        duration_scale.grid(row=1, column=1, pady=10, padx=10)

        # Audio device selection
        tk.Label(
            settings_frame,
            text="Audio Device:",
            font=("Arial", 12),
            bg=self.colors["light"],
        ).grid(row=2, column=0, sticky="w", pady=10)

        device_var = tk.StringVar(value="Default")
        device_combo = ttk.Combobox(
            settings_frame,
            textvariable=device_var,
            values=["Default"] + self.get_audio_devices(),
        )
        device_combo.grid(row=2, column=1, pady=10, padx=10, sticky="ew")

        # Buttons
        button_frame = tk.Frame(settings_window, bg=self.colors["light"])
        button_frame.pack(pady=20)

        def save_settings():
            self.settings["confidence_threshold"] = conf_var.get()
            self.settings["record_duration"] = duration_var.get()

            if self.live_predictor:
                self.live_predictor.confidence_threshold = conf_var.get()
                self.live_predictor.record_seconds = duration_var.get()

            messagebox.showinfo("Success", "Settings saved successfully!")
            settings_window.destroy()

        ModernButton(
            button_frame,
            text="💾 Save Settings",
            command=save_settings,
            bg_color=self.colors["secondary"],
        ).pack(side=tk.LEFT, padx=10)

        ModernButton(
            button_frame,
            text="❌ Cancel",
            command=settings_window.destroy,
            bg_color=self.colors["danger"],
        ).pack(side=tk.LEFT, padx=10)

    def get_audio_devices(self):
        """Get list of available audio devices."""
        if self.live_predictor:
            devices = self.live_predictor.list_audio_devices()
            return [f"{d['index']}: {d['name']}" for d in devices]
        return []

    def update_status(self, message, status_type="info"):
        """Update status display."""
        colors = {
            "info": "#3498db",
            "success": "#2ecc71",
            "warning": "#f39c12",
            "error": "#e74c3c",
            "recording": "#e74c3c",
        }

        self.status_label.config(
            text=f"● {message}", fg=colors.get(status_type, "#3498db")
        )

    def update_prediction_display(self, digit, confidence):
        """Update the main prediction display."""
        # Update digit display with animation
        self.prediction_display.config(text=str(digit))

        # Color coding based on confidence
        if confidence >= self.settings["confidence_threshold"]:
            self.prediction_display.config(bg="#d5f4e6", fg="#27ae60")
        else:
            self.prediction_display.config(bg="#fadbd8", fg="#e74c3c")

        # Update confidence bar
        self.confidence_var.set(confidence * 100)
        self.confidence_label.config(text=f"{confidence:.1%}")

        # Add to history
        self.prediction_history.append(
            {"digit": digit, "confidence": confidence, "timestamp": time.time()}
        )

        # Update recent display
        self.update_recent_display()

    def update_recent_display(self):
        """Update the recent predictions display."""
        # Clear existing
        for widget in self.recent_frame.winfo_children():
            widget.destroy()

        # Show last 5 predictions
        recent = self.prediction_history[-5:]
        for i, pred in enumerate(reversed(recent)):
            color = (
                "#27ae60"
                if pred["confidence"] >= self.settings["confidence_threshold"]
                else "#e74c3c"
            )

            pred_label = tk.Label(
                self.recent_frame,
                text=f"{pred['digit']} ({pred['confidence']:.2f})",
                font=("Arial", 10),
                fg=color,
                bg=self.colors["light"],
            )
            pred_label.pack(anchor="w")

    def update_stats_display(self):
        """Update the statistics text display."""
        if self.predictor:
            try:
                pred_stats = self.predictor.get_performance_stats()

                stats_text = f"""📊 Model Performance:
  
🚀 Speed Metrics:
  Avg Inference: {pred_stats.get('avg_inference_time_ms', 0):.1f} ms
  Max Inference: {pred_stats.get('max_inference_time_ms', 0):.1f} ms
  Min Inference: {pred_stats.get('min_inference_time_ms', 0):.1f} ms
  Pred/Second: {pred_stats.get('predictions_per_second', 0):.1f}

📈 Session Stats:
  Total Predictions: {len(self.prediction_history)}
  High Confidence: {sum(1 for p in self.prediction_history if p['confidence'] >= self.settings['confidence_threshold'])}
  Low Confidence: {sum(1 for p in self.prediction_history if p['confidence'] < self.settings['confidence_threshold'])}

⚙️ Current Settings:
  Confidence Threshold: {self.settings['confidence_threshold']:.2f}
  Record Duration: {self.settings['record_duration']:.1f}s
  Device: {"Default" if not self.settings['device_index'] else f"Device {self.settings['device_index']}"}

🎯 Recent Activity:
"""

                # Add recent predictions
                for pred in self.prediction_history[-5:]:
                    time_str = time.strftime(
                        "%H:%M:%S", time.localtime(pred["timestamp"])
                    )
                    stats_text += (
                        f"  {time_str}: {pred['digit']} ({pred['confidence']:.2f})\n"
                    )

            except Exception as e:
                stats_text = f"Error getting stats: {e}"
        else:
            stats_text = "Please load a model to see statistics."

        # Update text widget
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats_text)
        self.stats_text.config(state="disabled")

    def update_gui(self):
        """Update GUI with any pending data."""
        try:
            while not self.prediction_queue.empty():
                data = self.prediction_queue.get_nowait()

                if data["type"] in ["single_prediction", "continuous_prediction"]:
                    self.update_prediction_display(data["digit"], data["confidence"])
                    self.waveform_widget.update_waveform(data["audio"])
                    self.update_stats_display()

                    if data["type"] == "single_prediction":
                        self.update_status("✅ Prediction complete", "success")

                elif data["type"] == "error":
                    self.update_status(f"❌ Error: {data['message']}", "error")
                    messagebox.showerror("Error", data["message"])

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.update_gui)

    def quit_application(self):
        """Quit the application."""
        result = messagebox.askyesno("Confirm Exit", "Are you sure you want to quit?")
        if result:
            # Stop any running threads
            self.is_continuous = False
            self.is_recording = False

            # Cleanup resources
            if self.live_predictor:
                self.live_predictor.cleanup()

            self.root.quit()
            self.root.destroy()

    def run(self):
        """Run the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit_application()


def main():
    """Main function to run the GUI application."""
    import argparse

    parser = argparse.ArgumentParser(description="Digit Classifier GUI")
    parser.add_argument(
        "--model_path",
        default="models/best_model.pth",
        help="Path to the trained model",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"⚠️  Model file not found: {args.model_path}")
        print("You can load a model using the GUI or specify a different path.")

    # Initialize and run GUI
    try:
        app = DigitClassifierGUI(args.model_path)
        app.run()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        print("Make sure you have activated your environment: conda activate audioLLM")


if __name__ == "__main__":
    main()
