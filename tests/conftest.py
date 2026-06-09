"""Make project root importable so tests can reach 01_handwritten_nn / 02_inference_principles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
