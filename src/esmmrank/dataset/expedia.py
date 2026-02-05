"""Expedia Personalized Sort dataset loader."""
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader

from .base import collate_fn


class ExpediaDataset(Dataset):
    """
    Dataset for preprocessed Expedia search ranking data.

    Parameters
    ----------
    data_dir : Path
        Directory containing preprocessed .npy files.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

        self.user_categorical = self._load_categorical("user_categorical")
        self.user_numerical = np.load(self.data_dir / "user_numerical.npy")
        self.hotel_categorical = self._load_categorical("hotel_categorical")
        self.hotel_numerical = np.load(self.data_dir / "hotel_numerical.npy")
        self.context_categorical = self._load_categorical("context_categorical")
        self.context_numerical = np.load(self.data_dir / "context_numerical.npy")
        self.click_labels = np.load(self.data_dir / "click_labels.npy")
        self.booking_labels = np.load(self.data_dir / "booking_labels.npy")
        self.session_ids = np.load(self.data_dir / "session_ids.npy")

    def _load_categorical(self, prefix: str) -> dict[str, np.ndarray]:
        """Load all categorical feature arrays with given prefix."""
        features = {}
        for path in self.data_dir.glob(f"{prefix}_*.npy"):
            name = path.stem.replace(f"{prefix}_", "")
            features[name] = np.load(path)
        return features

    def __len__(self) -> int:
        return len(self.click_labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "user_categorical": {k: v[idx] for k, v in self.user_categorical.items()},
            "user_numerical": self.user_numerical[idx],
            "hotel_categorical": {k: v[idx] for k, v in self.hotel_categorical.items()},
            "hotel_numerical": self.hotel_numerical[idx],
            "context_categorical": {
                k: v[idx] for k, v in self.context_categorical.items()
            },
            "context_numerical": self.context_numerical[idx],
            "click_label": self.click_labels[idx],
            "conversion_label": self.booking_labels[idx],
            "session_id": self.session_ids[idx],
        }


def create_expedia_loaders(
    data_dir: Path,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for preprocessed Expedia data.

    Parameters
    ----------
    data_dir : Path
        Root directory containing train/, val/, test/ subdirectories.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data loading workers.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, and test data loaders.
    """
    data_dir = Path(data_dir)

    train_dataset = ExpediaDataset(data_dir / "train")
    val_dataset = ExpediaDataset(data_dir / "val")
    test_dataset = ExpediaDataset(data_dir / "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_expedia_feature_config() -> dict:
    """
    Get feature configuration matching preprocessed Expedia data.

    Returns
    -------
    dict
        Configuration dict for FeatureConfig.
    """
    return {
        "user_categorical_dims": {
            "visitor_location_country_id": 250,
            "site_id": 50,
        },
        "user_numerical_dim": 7,
        "hotel_categorical_dims": {
            "prop_id": 150000,
            "prop_country_id": 250,
            "prop_starrating": 6,
            "prop_brand_bool": 2,
            "promotion_flag": 2,
        },
        "hotel_numerical_dim": 7,
        "context_categorical_dims": {
            "srch_destination_id": 50000,
            "srch_saturday_night_bool": 2,
            "random_bool": 2,
        },
        "context_numerical_dim": 1,
        "embedding_dim": 32,
    }
