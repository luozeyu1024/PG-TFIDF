"""
scNET inference utility module
Provides scNET training functionality

Module structure:
- scnet_trainer: scNET trainer (minimal version based on the official API)

Note: preprocessing and network construction are integrated into the official scNET API and need not be reimplemented.
"""

from .scnet_trainer import SCNetTrainer

__all__ = [
    'SCNetTrainer'
]
