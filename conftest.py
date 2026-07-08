"""Ensure `moodengine.*` is importable from source during tests.

With moodengine installed (editable), imports resolve directly; this keeps test
collection robust even when run without an install (adds src/ to sys.path).
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
