"""Make project root importable so tests can reach 01_handwritten_nn / 02_inference_principles."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 让 logging.info / logging.debug 在测试中可见
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
