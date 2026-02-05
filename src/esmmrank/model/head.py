import torch
import torch.nn as nn

from ..config import HeadConfig
from .layer import MLPBlock


class PredictionHead(nn.Module):
    """
    Prediction head for CTR or CVR estimation.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    head_config : HeadConfig
        Configuration for the prediction head.
    """

    def __init__(self, input_dim: int, head_config: HeadConfig):
        super().__init__()
        self.mlp = MLPBlock(
            input_dim=input_dim,
            hidden_dims=head_config.hidden_dims,
            output_dim=1,
            dropout=head_config.dropout,
            use_batch_norm=head_config.use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Probability logits of shape (batch_size, 1).
        """
        return self.mlp(x)


class ESMMHead(nn.Module):
    """
    Entire Space Multi-Task Model (ESMM) head.

    Implements the ESMM architecture for joint CTR and CVR prediction.
    The key insight is that P(CTCVR) = P(CTR) * P(CVR|CTR), which allows
    training on the entire impression space rather than just clicked items.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    ctr_config : HeadConfig
        Configuration for CTR prediction head.
    cvr_config : HeadConfig
        Configuration for CVR prediction head.
    """

    def __init__(
        self,
        input_dim: int,
        ctr_config: HeadConfig,
        cvr_config: HeadConfig,
    ):
        super().__init__()
        self.ctr_head = PredictionHead(input_dim, ctr_config)
        self.cvr_head = PredictionHead(input_dim, cvr_config)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shared representation of shape (batch_size, input_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - p_ctr: Click probability of shape (batch_size, 1)
            - p_cvr: Conversion probability given click of shape (batch_size, 1)
            - p_ctcvr: Click-through conversion probability of shape (batch_size, 1)
        """
        ctr_logits = self.ctr_head(x)
        cvr_logits = self.cvr_head(x)

        p_ctr = torch.sigmoid(ctr_logits)
        p_cvr = torch.sigmoid(cvr_logits)
        p_ctcvr = p_ctr * p_cvr

        return p_ctr, p_cvr, p_ctcvr