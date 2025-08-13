#!/usr/bin/env python3
"""
Simple launcher for the Digit Classifier GUI
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Launch the GUI
if __name__ == "__main__":
    try:
        from gui_app import main

        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to activate your environment: conda activate audioLLM")
        print("And install required packages if missing.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
