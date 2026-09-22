import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / 'src')]
