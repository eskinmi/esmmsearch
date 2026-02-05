import torch
import torch.nn as nn

from ..config import FeatureConfig, TowerConfig
from .layer import EmbeddingEncoder, MLPBlock


class Tower(nn.Module):
    """
    Base tower architecture for encoding entity features.

    Combines categorical embeddings with numerical features and processes
    through an MLP to produce dense representations.

    Parameters
    ----------
    categorical_dims : dict[str, int]
        Mapping of categorical feature names to vocabulary sizes.
    numerical_dim : int
        Number of numerical features.
    embedding_dim : int
        Embedding dimension for categorical features.
    tower_config : TowerConfig
        Configuration for the tower MLP.
    """

    def __init__(
        self,
        categorical_dims: dict[str, int],
        numerical_dim: int,
        embedding_dim: int,
        tower_config: TowerConfig,
    ):
        super().__init__()
        self.embedding_encoder = EmbeddingEncoder(categorical_dims, embedding_dim)

        total_input_dim = int(self.embedding_encoder.output_dim + numerical_dim)
        self.numerical_norm = nn.BatchNorm1d(numerical_dim)

        self.mlp = MLPBlock(
            input_dim=total_input_dim,
            hidden_dims=tower_config.hidden_dims,
            output_dim=tower_config.output_dim,
            dropout=tower_config.dropout,
            use_batch_norm=tower_config.use_batch_norm,
        )

        self.output_dim = tower_config.output_dim

    def forward(
        self,
        categorical_features: dict[str, torch.Tensor],
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        categorical_features : dict[str, torch.Tensor]
            Dictionary of categorical feature tensors.
        numerical_features : torch.Tensor
            Numerical features of shape (batch_size, numerical_dim).

        Returns
        -------
        torch.Tensor
            Entity representation of shape (batch_size, output_dim).
        """
        embedded = self.embedding_encoder(categorical_features)
        numerical_normed = self.numerical_norm(numerical_features)
        combined = torch.cat([embedded, numerical_normed], dim=-1)
        return self.mlp(combined)


class UserTower(Tower):
    """
    Tower for encoding user features.

    Parameters
    ----------
    feature_config : FeatureConfig
        Feature configuration containing user feature specs.
    tower_config : TowerConfig
        Configuration for the tower architecture.
    """

    def __init__(self, feature_config: FeatureConfig, tower_config: TowerConfig):
        super().__init__(
            categorical_dims=feature_config.user_categorical_dims,
            numerical_dim=feature_config.user_numerical_dim,
            embedding_dim=feature_config.embedding_dim,
            tower_config=tower_config,
        )


class HotelTower(Tower):
    """
    Tower for encoding hotel features.

    Parameters
    ----------
    feature_config : FeatureConfig
        Feature configuration containing hotel feature specs.
    tower_config : TowerConfig
        Configuration for the tower architecture.
    """

    def __init__(self, feature_config: FeatureConfig, tower_config: TowerConfig):
        super().__init__(
            categorical_dims=feature_config.hotel_categorical_dims,
            numerical_dim=feature_config.hotel_numerical_dim,
            embedding_dim=feature_config.embedding_dim,
            tower_config=tower_config,
        )


class ContextEncoder(nn.Module):
    """
    Encodes search context features (temporal, UI, etc.).

    Parameters
    ----------
    feature_config : FeatureConfig
        Feature configuration containing context feature specs.
    output_dim : int
        Output dimension of the context representation.
    """

    def __init__(self, feature_config: FeatureConfig, output_dim: int = 32):
        super().__init__()
        self.embedding_encoder = EmbeddingEncoder(
            feature_config.context_categorical_dims,
            feature_config.embedding_dim,
        )

        self.numerical_norm = nn.BatchNorm1d(feature_config.context_numerical_dim)

        total_input_dim = int(
            self.embedding_encoder.output_dim + feature_config.context_numerical_dim
        )

        self.projection = nn.Sequential(
            nn.Linear(total_input_dim, output_dim),
            nn.ReLU(),
        )

        self.output_dim = output_dim

    def forward(
        self,
        categorical_features: dict[str, torch.Tensor],
        numerical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        categorical_features : dict[str, torch.Tensor]
            Dictionary of context categorical feature tensors.
        numerical_features : torch.Tensor
            Context numerical features.

        Returns
        -------
        torch.Tensor
            Context representation of shape (batch_size, output_dim).
        """
        embedded = self.embedding_encoder(categorical_features)
        numerical_normed = self.numerical_norm(numerical_features)
        combined = torch.cat([embedded, numerical_normed], dim=-1)
        return self.projection(combined)