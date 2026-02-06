import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in CTR/CVR prediction.

    Focal loss down-weights easy examples and focuses on hard negatives,
    which is critical for imbalanced click/conversion data.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values increase focus on hard examples.
    alpha : float
        Weighting factor for the positive class.
    reduction : str
        Reduction method: 'mean', 'sum', or 'none'.

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities of shape (batch_size, 1) or (batch_size,).
        targets : torch.Tensor
            Binary targets of shape (batch_size, 1) or (batch_size,).

        Returns
        -------
        torch.Tensor
            Focal loss value.
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()

        predictions = torch.clamp(predictions, min=1e-7, max=1 - 1e-7)

        bce_loss = F.binary_cross_entropy(predictions, targets, reduction="none")

        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class IPSWeighter(nn.Module):
    """
    Inverse Propensity Score weighting for position bias correction.

    Uses a power-law model: w(pos) = 1 / pos^eta, normalized to mean 1.
    """

    def __init__(self, eta: float = 1.0, max_position: int = 40):
        super().__init__()
        self.eta = eta
        self.max_position = max_position

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.float().clamp(1, self.max_position)
        weights = 1.0 / (positions ** self.eta)
        return weights / weights.mean()


class ESMMLoss(nn.Module):
    """
    Combined loss for ESMM training.

    Computes weighted combination of CTR loss and CTCVR loss.
    Note: CVR is not trained directly; it's implicitly learned through
    the CTCVR = CTR * CVR formulation.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        ctr_weight: float = 1.0,
        ctcvr_weight: float = 5.0,
        ips_weighter: IPSWeighter | None = None,
    ):
        super().__init__()
        self.ctr_loss_fn = FocalLoss(gamma=gamma, alpha=alpha, reduction="none")
        self.ctcvr_loss_fn = FocalLoss(gamma=gamma, alpha=alpha, reduction="none")
        self.ctr_weight = ctr_weight
        self.ctcvr_weight = ctcvr_weight
        self.ips_weighter = ips_weighter

    def forward(
        self,
        p_ctr: torch.Tensor,
        p_ctcvr: torch.Tensor,
        click_labels: torch.Tensor,
        conversion_labels: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ctr_loss = self.ctr_loss_fn(p_ctr, click_labels)
        ctcvr_loss = self.ctcvr_loss_fn(p_ctcvr, conversion_labels)

        if self.ips_weighter is not None and positions is not None:
            w = self.ips_weighter(positions)
            ctr_loss = ctr_loss * w
            ctcvr_loss = ctcvr_loss * w

        ctr_loss = ctr_loss.mean()
        ctcvr_loss = ctcvr_loss.mean()
        total_loss = self.ctr_weight * ctr_loss + self.ctcvr_weight * ctcvr_loss

        return {
            "total": total_loss,
            "ctr": ctr_loss,
            "ctcvr": ctcvr_loss,
        }