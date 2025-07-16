from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mewtwo.machine_learning.transformer.config.config_types import LossFunctionType
from mewtwo.parsers.parse_model_config import LossFunctionConfig


def weighted_mse_loss(preds, targets, weights):
    return (weights * (preds - targets) ** 2).mean()


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, weights):
        return weighted_mse_loss(preds, targets, weights)


class CombinedMSEPearsonLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: weight for MSE loss (0 <= alpha <= 1)
        (1 - alpha) will be used for the Pearson correlation loss
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure preds and targets have the same shape
        if preds.shape != targets.shape:
            raise ValueError("Shape mismatch between predictions and targets.")

        if weights is not None:
            if targets.shape != weights.shape:
                raise ValueError("Shape mismatch between weights and targets.")

        if weights is None:
            # MSE Loss
            mse_loss = F.mse_loss(preds, targets)
        else:
            mse_loss = weighted_mse_loss(preds, targets, weights)

        # Pearson Correlation
        preds_centered = preds - preds.mean()
        targets_centered = targets - targets.mean()

        numerator = torch.sum(preds_centered * targets_centered)
        denominator = torch.sqrt(torch.sum(preds_centered ** 2)) * torch.sqrt(torch.sum(targets_centered ** 2)) + 1e-8
        pearson_corr = numerator / denominator

        corr_loss = 1 - pearson_corr

        # Weighted combination
        return self.alpha * mse_loss + (1 - self.alpha) * corr_loss


def soft_rank(x, regularization_strength=1e-3):
    """
    Approximate the ranks of elements in x using a softmax-based method.
    Returns a tensor of the same shape with values approximating the rank of each element.
    """
    x = x.unsqueeze(-1)
    diff = x - x.transpose(0, 1)
    P = torch.sigmoid(-diff / regularization_strength)
    soft_ranks = P.sum(dim=-1) + 0.5  # Adding 0.5 to center ranks correctly
    return soft_ranks


class CombinedMSESpearmanLoss(nn.Module):
    def __init__(self, alpha=0.5, reg_strength=1e-3):
        """
        alpha: weight for MSE loss (0 <= alpha <= 1)
        reg_strength: smoothness factor for soft ranking
        """
        super().__init__()
        self.alpha = alpha
        self.reg_strength = reg_strength

    def forward(self, preds, targets):
        if preds.shape != targets.shape:
            raise ValueError("Shape mismatch between predictions and targets.")

        # MSE
        mse_loss = F.mse_loss(preds, targets)

        # Soft ranks
        preds_rank = soft_rank(preds.squeeze(), self.reg_strength)
        targets_rank = soft_rank(targets.squeeze(), self.reg_strength)

        # Centered ranks for Spearman
        preds_rank_centered = preds_rank - preds_rank.mean()
        targets_rank_centered = targets_rank - targets_rank.mean()

        # Spearman correlation (same as Pearson but on ranks)
        numerator = torch.sum(preds_rank_centered * targets_rank_centered)
        denominator = (
            torch.sqrt(torch.sum(preds_rank_centered**2)) * torch.sqrt(torch.sum(targets_rank_centered**2)) + 1e-8
        )
        spearman_corr = numerator / denominator
        spearman_loss = 1 - spearman_corr

        return self.alpha * mse_loss + (1 - self.alpha) * spearman_loss


TYPE_TO_LOSS_FN = {LossFunctionType.MSE: nn.MSELoss,
                   LossFunctionType.WEIGHTED_MSE: WeightedMSELoss,
                   LossFunctionType.MSE_PEARSON: CombinedMSEPearsonLoss,
                   LossFunctionType.MSE_SPEARMAN: CombinedMSESpearmanLoss,
                   LossFunctionType.WEIGHTED_MSE_PEARSON: CombinedMSEPearsonLoss,
                   LossFunctionType.WEIGHTED_MSE_SPEARMAN: CombinedMSESpearmanLoss,
                   LossFunctionType.PEARSON: CombinedMSEPearsonLoss,
                   LossFunctionType.SPEARMAN: CombinedMSESpearmanLoss}


def get_loss_function(config: LossFunctionConfig):
    loss_fn = TYPE_TO_LOSS_FN[config.type]
    if config.type in LossFunctionType.NEEDS_ALPHA:
        loss_fn_instance = loss_fn(alpha=config.alpha)
    elif config.type in LossFunctionType.CORRELATION_ONLY:
        loss_fn_instance = loss_fn(alpha=0.0)
    else:
        loss_fn_instance = loss_fn()

    return loss_fn_instance

