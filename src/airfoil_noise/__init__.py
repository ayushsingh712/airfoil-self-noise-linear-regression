"""Linear regression on the UCI Airfoil Self-Noise dataset."""

from airfoil_noise.data import load_data
from airfoil_noise.pipeline import run

__version__ = "0.1.0"
__all__ = ["load_data", "run", "__version__"]
