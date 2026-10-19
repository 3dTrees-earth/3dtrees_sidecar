import sys
from pathlib import Path

# The image ships src/ flat (/src/run.py, /src/sidecar.py); tests import it the same way.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
