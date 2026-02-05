"""Pseudo (synthetic) data generation for development and testing."""
import numpy as np
from torch.utils.data import DataLoader

from ..config import FeatureConfig
from .base import BaseSearchDataset, collate_fn


class PseudoDataset(BaseSearchDataset):
    """Dataset created from synthetic pseudo data."""

    pass


class PseudoDataGenerator:
    """
    Generates synthetic search ranking data for development and testing.

    Simulates realistic click and conversion patterns:
    - CTR ~32% (based on presentation funnel)
    - CVR ~7.5% of clicks (2.4% overall conversion)
    - Position bias in click probability
    - User-hotel affinity patterns

    Parameters
    ----------
    feature_config : FeatureConfig
        Feature configuration for data generation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, feature_config: FeatureConfig, seed: int = 42):
        self.config = feature_config
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_sessions: int = 10000,
        hotels_per_session: int = 20,
    ) -> PseudoDataset:
        """
        Generate a synthetic dataset.

        Parameters
        ----------
        n_sessions : int
            Number of search sessions to generate.
        hotels_per_session : int
            Number of hotel impressions per session.

        Returns
        -------
        PseudoDataset
            Generated dataset.
        """
        n_samples = n_sessions * hotels_per_session

        user_categorical = self._generate_categorical(
            self.config.user_categorical_dims, n_sessions, hotels_per_session
        )
        user_numerical = self._generate_numerical(
            self.config.user_numerical_dim, n_sessions, hotels_per_session
        )

        hotel_categorical = self._generate_categorical(
            self.config.hotel_categorical_dims, n_samples
        )
        hotel_numerical = self._generate_numerical(
            self.config.hotel_numerical_dim, n_samples
        )

        context_categorical = self._generate_categorical(
            self.config.context_categorical_dims, n_sessions, hotels_per_session
        )
        context_numerical = self._generate_numerical(
            self.config.context_numerical_dim, n_sessions, hotels_per_session
        )

        session_ids = np.repeat(np.arange(n_sessions), hotels_per_session)
        positions = np.tile(np.arange(hotels_per_session), n_sessions)

        click_labels, conversion_labels = self._generate_labels(
            n_samples, positions, user_numerical, hotel_numerical
        )

        return PseudoDataset(
            user_categorical=user_categorical,
            user_numerical=user_numerical,
            hotel_categorical=hotel_categorical,
            hotel_numerical=hotel_numerical,
            context_categorical=context_categorical,
            context_numerical=context_numerical,
            click_labels=click_labels,
            conversion_labels=conversion_labels,
            session_ids=session_ids,
        )

    def _generate_categorical(
        self,
        dims: dict[str, int],
        n_samples: int,
        repeat: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate categorical features."""
        features = {}
        actual_samples = n_samples if repeat is None else n_samples

        for name, vocab_size in dims.items():
            values = self.rng.integers(0, vocab_size, size=actual_samples)
            if repeat is not None:
                values = np.repeat(values, repeat)
            features[name] = values

        return features

    def _generate_numerical(
        self,
        dim: int,
        n_samples: int,
        repeat: int | None = None,
    ) -> np.ndarray:
        """Generate numerical features."""
        values = self.rng.standard_normal((n_samples, dim))
        if repeat is not None:
            values = np.repeat(values, repeat, axis=0)
        return values

    def _generate_labels(
        self,
        n_samples: int,
        positions: np.ndarray,
        user_features: np.ndarray,
        hotel_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate click and conversion labels with realistic patterns.

        Simulates position bias, user-hotel affinity, and conversion after click.
        """
        position_bias = 1.0 / (1.0 + 0.1 * positions)

        user_signal = user_features[:, 0]
        hotel_signal = hotel_features[:, 0]
        affinity = 1.0 / (1.0 + np.abs(user_signal - hotel_signal))

        base_ctr = 0.32
        click_prob = base_ctr * position_bias * (0.5 + 0.5 * affinity)
        click_prob = np.clip(click_prob, 0.05, 0.8)

        clicks = self.rng.random(n_samples) < click_prob

        base_cvr = 0.075
        cvr_given_click = base_cvr * (0.5 + 0.5 * affinity)
        cvr_given_click = np.clip(cvr_given_click, 0.01, 0.3)

        conversions = clicks & (self.rng.random(n_samples) < cvr_given_click)

        return clicks.astype(np.float32), conversions.astype(np.float32)


def create_pseudo_loaders(
    feature_config: FeatureConfig,
    batch_size: int = 256,
    n_train_sessions: int = 8000,
    n_val_sessions: int = 1000,
    n_test_sessions: int = 1000,
    hotels_per_session: int = 20,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders with pseudo data.

    Parameters
    ----------
    feature_config : FeatureConfig
        Feature configuration.
    batch_size : int
        Batch size for data loaders.
    n_train_sessions : int
        Number of training sessions.
    n_val_sessions : int
        Number of validation sessions.
    n_test_sessions : int
        Number of test sessions.
    hotels_per_session : int
        Hotels per session.
    seed : int
        Random seed.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, and test data loaders.
    """
    generator = PseudoDataGenerator(feature_config, seed=seed)

    train_dataset = generator.generate(n_train_sessions, hotels_per_session)
    val_dataset = generator.generate(n_val_sessions, hotels_per_session)
    test_dataset = generator.generate(n_test_sessions, hotels_per_session)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
