#!/usr/bin/env python
# bilinear_head.py
# Provide a standalone bilinear-head constructor for reuse / testing / replacement

import torch


def build_head(dim: int) -> torch.nn.Bilinear:
    """
    Return a torch.nn.Bilinear layer with sigmoid activation *not* baked in.
    Match the original implementation: output dimension=1, with no extra wrapper layers.
    """
    return torch.nn.Bilinear(dim, dim, 1)
