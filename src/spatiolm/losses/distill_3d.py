from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient(x):
    """Forward finite differences.
    Accepts tensors of shape (B, 1, H, W) or (B, H, W). Returns dx, dy.
    """
    # x: B x 1 x H x W
    dx = x[..., :, 1:] - x[..., :, :-1]  # B x 1 x H x (W-1)
    dy = x[..., 1:, :] - x[..., :-1, :]  # B x 1 x (H-1) x W
    return dx, dy


def regression_loss(
    pred: torch.Tensor,
    pred_conf: torch.Tensor,
    gt: torch.Tensor,
    gt_conf: torch.Tensor,
    gamma: float = 1.0,
    alpha: float = 0.2,
    grad: bool = False,
):
    # err_map = F.mse_loss(pred, gt, reduction="none")  # B x H x W
    err_map = (pred - gt).norm(dim=-1)

    # Apply confidence weights
    conf_weight = gt_conf.log()
    loss_reg = (err_map * conf_weight).sum() / conf_weight.sum().clamp(min=1e-5)
    # Apply confidence loss
    loss_conf = (gamma * err_map * pred_conf - alpha * pred_conf.log()).mean()

    losses = [loss_reg, loss_conf]
    if grad:
        pred_dx, pred_dy = gradient(pred.squeeze(-1))
        gt_dx, gt_dy = gradient(gt.squeeze(-1))
        loss_grad = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)
        losses.append(loss_grad)

    return tuple(losses)


class Distill3dLoss(nn.Module):
    def __init__(self, gamma: float = 1.0, alpha: float = 0.2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: Dict, ref: Dict):
        loss_depths = regression_loss(
            pred["depth"].unsqueeze(-1),
            pred["depth_conf"],
            ref["depth"].unsqueeze(-1),
            ref["depth_conf"],
            grad=True,
        )

        loss_raymap = regression_loss(
            pred["ray"],
            pred["ray_conf"],
            ref["ray"],
            ref["ray_conf"],
        )

        return dict(
            depth=sum(loss_depths),
            ray=sum(loss_raymap),
        )
