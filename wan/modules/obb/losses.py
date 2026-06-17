"""Losses: rectified-flow matching (main) + self-tracking auxiliary."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(v_hat, x0, x1):
    """Rectified flow: x_t=(1-t)x0+t x1, target velocity = x1-x0."""
    return F.mse_loss(v_hat, x1 - x0)


def self_tracking_loss(inst_logits, depth_pred, gt_inst, gt_depth, fg):
    """inst_logits:(B,F,H,W,N)  depth_pred:(B,F,H,W)  gt_inst:(B,F,H,W) long  fg:(B,F,H,W) bool."""
    if fg.sum() == 0:
        return inst_logits.sum() * 0.0
    ce = F.cross_entropy(inst_logits[fg], gt_inst[fg])
    de = F.mse_loss(depth_pred[fg], gt_depth[fg])
    return ce + de
