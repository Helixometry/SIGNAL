"""SIGNAL: Graph-Augmented Instance Learning for Synthetic Speech Attribution"""

__version__ = "1.0.0"
__author__ = "Mohd Mujtaba Akhtar, Girish, Farhan Sheth, Muskaan Singh"

from .signal import SIGNAL
from .graph_encoder import GraphEncoder

__all__ = ["SIGNAL", "GraphEncoder"]
