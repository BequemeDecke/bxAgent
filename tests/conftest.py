import sys
from pathlib import Path

# Add src directory to Python path so tests can import from src
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
