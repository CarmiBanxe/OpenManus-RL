"""Root conftest — makes openmanus_rl importable from tests."""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import openmanus_rl` works
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
