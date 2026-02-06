"""Base dataset utilities shared across dataset implementations."""
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseSearchDataset(Dataset):
    """
    Base dataset for search ranking with user-hotel impressions.

    Parameters
    ----------
    user_categorical : dict[str, np.ndarray]
        User categorical features.
    user_numerical : np.ndarray
        User numerical features.
    hotel_categorical : dict[str, np.ndarray]
        Hotel categorical features.
    hotel_numerical : np.ndarray
        Hotel numerical features.
    context_categorical : dict[str, np.ndarray]
        Context categorical features.
    context_numerical : np.ndarray
        Context numerical features.
    click_labels : np.ndarray
        Binary click labels.
    conversion_labels : np.ndarray
        Binary conversion labels.
    session_ids : np.ndarray
        Session/query group identifiers.
    """

    def __init__(
        self,
        user_categorical: dict[str, np.ndarray],
        user_numerical: np.ndarray,
        hotel_categorical: dict[str, np.ndarray],
        hotel_numerical: np.ndarray,
        context_categorical: dict[str, np.ndarray],
        context_numerical: np.ndarray,
        click_labels: np.ndarray,
        conversion_labels: np.ndarray,
        session_ids: np.ndarray,
    ):
        self.user_categorical = user_categorical
        self.user_numerical = user_numerical.astype(np.float32)
        self.hotel_categorical = hotel_categorical
        self.hotel_numerical = hotel_numerical.astype(np.float32)
        self.context_categorical = context_categorical
        self.context_numerical = context_numerical.astype(np.float32)
        self.click_labels = click_labels.astype(np.float32)
        self.conversion_labels = conversion_labels.astype(np.float32)
        self.session_ids = session_ids

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
            "conversion_label": self.conversion_labels[idx],
            "session_id": self.session_ids[idx],
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for batching search ranking samples.

    Parameters
    ----------
    batch : list[dict]
        List of sample dictionaries.

    Returns
    -------
    dict
        Batched tensors.
    """
    user_cat_keys = batch[0]["user_categorical"].keys()
    hotel_cat_keys = batch[0]["hotel_categorical"].keys()
    context_cat_keys = batch[0]["context_categorical"].keys()

    return {
        "user_categorical": {
            k: torch.tensor([s["user_categorical"][k] for s in batch], dtype=torch.long)
            for k in user_cat_keys
        },
        "user_numerical": torch.tensor(
            np.stack([s["user_numerical"] for s in batch]), dtype=torch.float32
        ),
        "hotel_categorical": {
            k: torch.tensor(
                [s["hotel_categorical"][k] for s in batch], dtype=torch.long
            )
            for k in hotel_cat_keys
        },
        "hotel_numerical": torch.tensor(
            np.stack([s["hotel_numerical"] for s in batch]), dtype=torch.float32
        ),
        "context_categorical": {
            k: torch.tensor(
                [s["context_categorical"][k] for s in batch], dtype=torch.long
            )
            for k in context_cat_keys
        },
        "context_numerical": torch.tensor(
            np.stack([s["context_numerical"] for s in batch]), dtype=torch.float32
        ),
        "click_labels": torch.tensor(
            [s["click_label"] for s in batch], dtype=torch.float32
        ),
        "conversion_labels": torch.tensor(
            [s["conversion_label"] for s in batch], dtype=torch.float32
        ),
        "session_ids": torch.tensor([s["session_id"] for s in batch], dtype=torch.long),
        "positions": torch.tensor(
            [s.get("position", 1) for s in batch], dtype=torch.long
        ),
    }
