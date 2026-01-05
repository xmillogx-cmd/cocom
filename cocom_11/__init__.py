"""
COCOM v11 - Modular Attention Architecture.

A modular implementation with MultiScaleEMA Tactical Expert (FFT convolution).
"""

from .model import COCOMv11
from .arbitrator import Arbitrator

__all__ = ['COCOMv11', 'Arbitrator']
__version__ = '11.0.0'
