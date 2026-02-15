"""Discrete Differential Geometry Playground"""

from .laplace_beltrami import LaplaceBeltrami
from .heat_diffusion import HeatDiffusion
from .spectral_geometry import SpectralGeometry
from .utils import MeshUtils

__all__ = [
    "LaplaceBeltrami",
    "HeatDiffusion",
    "SpectralGeometry",
    "MeshUtils",
]

__version__ = "0.1.0"
