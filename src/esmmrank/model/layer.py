import torch
import torch.nn as nn


class EmbeddingEncoder(nn.Module):
    """
    Encodes categorical features into dense embeddings.

    Parameters
    ----------
    categorical_dims : dict[str, int]
        Mapping of feature names to their vocabulary sizes.
    embedding_dim : int
        Dimension of the output embeddings.
    """

    def __init__(self, categorical_dims: dict[str, int], embedding_dim: int):
        super().__init__()
        self.feature_names = list(categorical_dims.keys())
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, embedding_dim)
                for name, vocab_size in categorical_dims.items()
            }
        )
        self.output_dim = len(categorical_dims) * embedding_dim

    def forward(self, categorical_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        categorical_features : dict[str, torch.Tensor]
            Dictionary mapping feature names to tensors of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Concatenated embeddings of shape (batch_size, num_features * embedding_dim).
        """
        embedded = [
            self.embeddings[name](categorical_features[name])
            for name in self.feature_names
        ]
        return torch.cat(embedded, dim=-1)


class MLPBlock(nn.Module):
    """
    Multi-layer perceptron block with optional batch normalization and dropout.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list[int]
        List of hidden layer dimensions.
    output_dim : int
        Output dimension.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to apply batch normalization.
    activation : str
        Activation function name ('relu', 'gelu', 'leaky_relu').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        activation_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }[activation]

        layers = list()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeatureInteraction(nn.Module):
    """
    Computes feature interactions between user and hotel representations.

    Combines element-wise product, difference, and concatenation.

    Parameters
    ----------
    representation_dim : int
        Dimension of input representations.
    """

    def __init__(self, representation_dim: int):
        super().__init__()
        self.representation_dim = representation_dim
        self.output_dim = representation_dim * 4

    def forward(  # noqa
        self, user_repr: torch.Tensor, hotel_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        user_repr : torch.Tensor
            User representation of shape (batch_size, representation_dim).
        hotel_repr : torch.Tensor
            Hotel representation of shape (batch_size, representation_dim).

        Returns
        -------
        torch.Tensor
            Interaction features of shape (batch_size, representation_dim * 4).
        """
        cross_repr = user_repr * hotel_repr
        delta = user_repr - hotel_repr
        return torch.cat([user_repr, hotel_repr, cross_repr, delta], dim=-1)
