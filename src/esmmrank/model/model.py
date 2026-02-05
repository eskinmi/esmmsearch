import torch
import torch.nn as nn

from ..config import ModelConfig
from .head import ESMMHead
from .layer import FeatureInteraction
from .tower import ContextEncoder, HotelTower, UserTower


class TwoTowerESMM(nn.Module):
    """
    Two-Tower architecture with ESMM head for search ranking.

    This model implements a two-stage architecture:
    1. Two parallel towers encode user and hotel features into dense representations
    2. Representations are combined with context features through interaction layers
    3. ESMM head predicts CTR, CVR, and CTCVR jointly

    The architecture enables:
    - Efficient retrieval via precomputed hotel embeddings
    - Joint optimization of click and conversion objectives
    - Handling of selection bias through entire-space training

    Parameters
    ----------
    config : ModelConfig
        Complete model configuration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.user_tower = UserTower(config.feature, config.user_tower)
        self.hotel_tower = HotelTower(config.feature, config.hotel_tower)
        self.context_encoder = ContextEncoder(config.feature, output_dim=32)

        self.feature_interaction = FeatureInteraction(config.user_tower.output_dim)

        shared_input_dim = int(
            self.feature_interaction.output_dim + self.context_encoder.output_dim
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(shared_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.esmm_head = ESMMHead(
            input_dim=64,
            ctr_config=config.ctr_head,
            cvr_config=config.cvr_head,
        )

    def encode_user(
        self,
        categorical_features: dict[str, torch.Tensor],
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode user features into a dense representation.

        Can be precomputed for logged-in users to reduce inference latency.

        Parameters
        ----------
        categorical_features : dict[str, torch.Tensor]
            User categorical features.
        numerical_features : torch.Tensor
            User numerical features.

        Returns
        -------
        torch.Tensor
            User representation of shape (batch_size, output_dim).
        """
        return self.user_tower(categorical_features, numerical_features)

    def encode_hotel(
        self,
        categorical_features: dict[str, torch.Tensor],
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode hotel features into a dense representation.

        Can be precomputed for all hotels and indexed for fast retrieval.

        Parameters
        ----------
        categorical_features : dict[str, torch.Tensor]
            Hotel categorical features.
        numerical_features : torch.Tensor
            Hotel numerical features.

        Returns
        -------
        torch.Tensor
            Hotel representation of shape (batch_size, output_dim).
        """
        return self.hotel_tower(categorical_features, numerical_features)

    def forward(
        self,
        user_categorical: dict[str, torch.Tensor],
        user_numerical: torch.Tensor,
        hotel_categorical: dict[str, torch.Tensor],
        hotel_numerical: torch.Tensor,
        context_categorical: dict[str, torch.Tensor],
        context_numerical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass computing CTR, CVR, and CTCVR predictions.

        Parameters
        ----------
        user_categorical : dict[str, torch.Tensor]
            User categorical features.
        user_numerical : torch.Tensor
            User numerical features of shape (batch_size, user_numerical_dim).
        hotel_categorical : dict[str, torch.Tensor]
            Hotel categorical features.
        hotel_numerical : torch.Tensor
            Hotel numerical features of shape (batch_size, hotel_numerical_dim).
        context_categorical : dict[str, torch.Tensor]
            Context categorical features.
        context_numerical : torch.Tensor
            Context numerical features of shape (batch_size, context_numerical_dim).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - 'p_ctr': Click probability
            - 'p_cvr': Conversion probability given click
            - 'p_ctcvr': Click-through conversion probability
            - 'user_repr': User tower representation
            - 'hotel_repr': Hotel tower representation
        """
        user_repr = self.encode_user(user_categorical, user_numerical)
        hotel_repr = self.encode_hotel(hotel_categorical, hotel_numerical)
        context_repr = self.context_encoder(context_categorical, context_numerical)

        interaction = self.feature_interaction(user_repr, hotel_repr)
        combined = torch.cat([interaction, context_repr], dim=-1)

        shared_repr = self.shared_layer(combined)

        p_ctr, p_cvr, p_ctcvr = self.esmm_head(shared_repr)

        return {
            "p_ctr": p_ctr,
            "p_cvr": p_cvr,
            "p_ctcvr": p_ctcvr,
            "user_repr": user_repr,
            "hotel_repr": hotel_repr,
        }

    def predict_from_embeddings(
        self,
        user_repr: torch.Tensor,
        hotel_repr: torch.Tensor,
        context_categorical: dict[str, torch.Tensor],
        context_numerical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict from precomputed user and hotel embeddings.

        Useful for inference when embeddings are cached.

        Parameters
        ----------
        user_repr : torch.Tensor
            Precomputed user representation.
        hotel_repr : torch.Tensor
            Precomputed hotel representation.
        context_categorical : dict[str, torch.Tensor]
            Context categorical features.
        context_numerical : torch.Tensor
            Context numerical features.

        Returns
        -------
        dict[str, torch.Tensor]
            Prediction dictionary with p_ctr, p_cvr, p_ctcvr.
        """
        context_repr = self.context_encoder(context_categorical, context_numerical)

        interaction = self.feature_interaction(user_repr, hotel_repr)
        combined = torch.cat([interaction, context_repr], dim=-1)

        shared_repr = self.shared_layer(combined)
        p_ctr, p_cvr, p_ctcvr = self.esmm_head(shared_repr)

        return {
            "p_ctr": p_ctr,
            "p_cvr": p_cvr,
            "p_ctcvr": p_ctcvr,
        }