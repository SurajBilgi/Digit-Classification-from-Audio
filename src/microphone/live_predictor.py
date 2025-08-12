"""
Real-time microphone integration for live digit prediction.
Captures audio from microphone and provides instant digit classification.
"""

import os
import sys
import time
import threading
import queue
import numpy as np
import pyaudio
from typing import Optional, Callable
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from inference.predictor import DigitPredictor


class LiveDigitPredictor:
    """Real-time digit predictor using microphone input."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 8000,
        chunk_size: int = 1024,
        record_seconds: float = 1.0,
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ):
        """
        Initialize live predictor.

        Args:
            model_path: Path to trained model
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size for recording
            record_seconds: Duration to record for each prediction
            confidence_threshold: Minimum confidence for valid prediction
            device: Computation device
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.record_seconds = record_seconds
        self.confidence_threshold = confidence_threshold

        # Audio recording setup
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.audio_queue = queue.Queue()

        # Load digit predictor
        self.predictor = DigitPredictor(model_path, device)

        # Performance tracking
        self.prediction_history = []
        self.timing_history = []

        print(f"Live predictor initialized with:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Record duration: {record_seconds} seconds")
        print(f"  Confidence threshold: {confidence_threshold}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()
        self.audio.terminate()
        print("Audio resources cleaned up")

    def list_audio_devices(self) -> list:
        """List available audio input devices."""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:  # Input device
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "rate": info["defaultSampleRate"],
                    }
                )
        return devices

    def print_audio_devices(self):
        """Print available audio devices."""
        devices = self.list_audio_devices()
        print("\nAvailable audio input devices:")
        for device in devices:
            print(
                f"  {device['index']}: {device['name']} "
                f"({device['channels']} ch, {device['rate']:.0f} Hz)"
            )

    def record_audio(self, device_index: Optional[int] = None) -> np.ndarray:
        """
        Record audio for the specified duration.

        Args:
            device_index: Audio device index (None for default)

        Returns:
            Recorded audio array
        """
        # Calculate total frames to record
        frames_to_record = int(self.sample_rate * self.record_seconds)

        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
        )

        print(f"Recording for {self.record_seconds} seconds...")

        # Record audio
        audio_data = []
        frames_recorded = 0

        try:
            while frames_recorded < frames_to_record:
                frames_to_read = min(
                    self.chunk_size, frames_to_record - frames_recorded
                )
                data = stream.read(frames_to_read)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                audio_data.extend(audio_chunk)
                frames_recorded += len(audio_chunk)

        except Exception as e:
            print(f"Error during recording: {e}")

        finally:
            stream.stop_stream()
            stream.close()

        audio_array = np.array(audio_data, dtype=np.float32)
        print(f"Recorded {len(audio_array)} samples")

        return audio_array

    def predict_from_microphone(
        self, device_index: Optional[int] = None, show_waveform: bool = False
    ) -> tuple:
        """
        Record audio and predict digit.

        Args:
            device_index: Audio device index
            show_waveform: Whether to show audio waveform

        Returns:
            Tuple of (digit, confidence, audio_array)
        """
        start_time = time.time()

        # Record audio
        audio_array = self.record_audio(device_index)

        # Show waveform if requested
        if show_waveform:
            self._show_waveform(audio_array)

        # Make prediction
        digit, confidence = self.predictor.predict(audio_array)

        total_time = time.time() - start_time

        # Store results
        self.prediction_history.append((digit, confidence))
        self.timing_history.append(total_time)

        return digit, confidence, audio_array

    def _show_waveform(self, audio_array: np.ndarray):
        """Display simple ASCII waveform."""
        # Downsample for display
        display_samples = 50
        step = len(audio_array) // display_samples
        downsampled = audio_array[::step][:display_samples]

        # Normalize to 0-20 range for display
        normalized = ((downsampled + 1) / 2 * 20).astype(int)

        print("\nAudio waveform:")
        for i, val in enumerate(normalized):
            bar = "█" * val + "░" * (20 - val)
            print(f"{i:2d}: {bar}")
        print()

    def continuous_prediction(
        self,
        device_index: Optional[int] = None,
        callback: Optional[Callable] = None,
        max_predictions: int = 10,
    ):
        """
        Run continuous prediction loop.

        Args:
            device_index: Audio device index
            callback: Callback function for predictions
            max_predictions: Maximum number of predictions (-1 for infinite)
        """
        print(f"\nStarting continuous prediction...")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Press Ctrl+C to stop\n")

        prediction_count = 0

        try:
            while max_predictions == -1 or prediction_count < max_predictions:
                print(f"--- Prediction {prediction_count + 1} ---")

                # Get prediction
                digit, confidence, audio = self.predict_from_microphone(device_index)

                # Check confidence
                if confidence >= self.confidence_threshold:
                    status = "✓ HIGH CONFIDENCE"
                    print(f"🔢 Predicted Digit: {digit}")
                    print(f"📊 Confidence: {confidence:.3f} {status}")
                else:
                    status = "⚠ LOW CONFIDENCE"
                    print(f"🔢 Predicted Digit: {digit} (uncertain)")
                    print(f"📊 Confidence: {confidence:.3f} {status}")

                # Call callback if provided
                if callback:
                    callback(digit, confidence, audio)

                # Show timing stats
                if len(self.timing_history) > 0:
                    avg_time = np.mean(self.timing_history[-10:])  # Last 10
                    print(f"⏱️  Avg Time: {avg_time:.2f}s")

                prediction_count += 1
                print()

                # Small pause between predictions
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n🛑 Stopping continuous prediction...")

    def interactive_session(self, device_index: Optional[int] = None):
        """Run interactive prediction session."""
        print("\n🎤 Interactive Digit Prediction Session")
        print("=====================================")

        # Show available devices
        self.print_audio_devices()

        # Main loop
        while True:
            print("\nOptions:")
            print("  [1] Predict single digit")
            print("  [2] Continuous prediction")
            print("  [3] Show prediction history")
            print("  [4] Show performance stats")
            print("  [5] Change settings")
            print("  [q] Quit")

            choice = input("\nEnter choice: ").strip().lower()

            if choice == "1":
                try:
                    digit, confidence, _ = self.predict_from_microphone(
                        device_index, show_waveform=True
                    )
                    if confidence >= self.confidence_threshold:
                        print(f"✅ Predicted: {digit} (confidence: {confidence:.3f})")
                    else:
                        print(
                            f"❓ Uncertain prediction: {digit} "
                            f"(confidence: {confidence:.3f})"
                        )
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif choice == "2":
                try:
                    num_predictions = input(
                        "Number of predictions (Enter for 10): "
                    ).strip()
                    max_pred = int(num_predictions) if num_predictions else 10
                    self.continuous_prediction(device_index, max_predictions=max_pred)
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif choice == "3":
                self._show_prediction_history()

            elif choice == "4":
                self._show_performance_stats()

            elif choice == "5":
                self._change_settings()

            elif choice == "q":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice")

    def _show_prediction_history(self):
        """Show recent prediction history."""
        if not self.prediction_history:
            print("📋 No predictions made yet")
            return

        print("\n📋 Recent Predictions:")
        recent = self.prediction_history[-10:]  # Last 10
        for i, (digit, conf) in enumerate(recent, 1):
            status = "✓" if conf >= self.confidence_threshold else "?"
            print(f"  {i:2d}. Digit: {digit}, Confidence: {conf:.3f} {status}")

    def _show_performance_stats(self):
        """Show performance statistics."""
        if not self.timing_history:
            print("📊 No timing data available")
            return

        pred_stats = self.predictor.get_performance_stats()
        times = np.array(self.timing_history)

        print("\n📊 Performance Statistics:")
        print(f"  Total predictions: {len(self.timing_history)}")
        print(f"  Avg total time: {np.mean(times):.2f}s")
        print(f"  Avg inference time: {pred_stats['avg_inference_time_ms']:.1f}ms")

        if len(self.prediction_history) > 0:
            confidences = [c for _, c in self.prediction_history]
            high_conf = sum(1 for c in confidences if c >= self.confidence_threshold)
            print(f"  High confidence predictions: {high_conf}/{len(confidences)}")

    def _change_settings(self):
        """Change predictor settings."""
        print("\n⚙️  Current Settings:")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Record duration: {self.record_seconds}s")

        try:
            new_threshold = input(
                f"New confidence threshold ({self.confidence_threshold}): "
            ).strip()
            if new_threshold:
                self.confidence_threshold = float(new_threshold)

            new_duration = input(
                f"New record duration ({self.record_seconds}s): "
            ).strip()
            if new_duration:
                self.record_seconds = float(new_duration)

            print("✅ Settings updated!")

        except ValueError:
            print("❌ Invalid input")


def main():
    """Main function for testing live predictor."""
    import argparse

    parser = argparse.ArgumentParser(description="Live digit prediction")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--device_index", type=int, help="Audio device index")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for valid predictions",
    )
    parser.add_argument(
        "--record_seconds",
        type=float,
        default=1.0,
        help="Recording duration per prediction",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive session"
    )

    args = parser.parse_args()

    try:
        with LiveDigitPredictor(
            args.model_path,
            confidence_threshold=args.confidence_threshold,
            record_seconds=args.record_seconds,
        ) as predictor:

            if args.interactive:
                predictor.interactive_session(args.device_index)
            else:
                # Single prediction test
                digit, confidence, _ = predictor.predict_from_microphone(
                    args.device_index, show_waveform=True
                )
                print(f"\nPredicted digit: {digit}")
                print(f"Confidence: {confidence:.3f}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
