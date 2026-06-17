"""Self-tracking head: predict per-pixel instance id + depth from DiT features.

Trains the model to bind each OBB to the right pixels (prevents identity bleed at many boxes).
Supervised by the free GT instance/depth maps from synthetic data (condition coarse, supervise rich).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SelfTrackingHead(nn.Module):
    def __init__(self, dim, n_instances):
        super().__init__()
        self.inst = nn.Linear(dim, n_instances)
        self.depth = nn.Linear(dim, 1)

    def forward(self, feat, fhw):
        B = feat.shape[0]
        F_, H_, W_ = fhw
        inst = self.inst(feat).view(B, F_, H_, W_, -1)   # (B,F,H,W,n_inst)
        depth = self.depth(feat).view(B, F_, H_, W_)     # (B,F,H,W)
        return inst, depth
