import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEARCH_FEATURES = [
    "srch_id",
    "site_id",
    "visitor_location_country_id",
    "srch_destination_id",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
]

HOTEL_FEATURES = [
    "prop_id",
    "prop_country_id",
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
    "srch_query_affinity_score",
    "orig_destination_distance",
]

USER_FEATURES = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
]

CONTEXT_FEATURES = [
    "random_bool",
    "position",
]

TARGET_FEATURES = [
    "click_bool",
    "booking_bool",
]


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """
    Load raw Expedia training data.

    Parameters
    ----------
    input_path : Path
        Path to directory containing train.csv.

    Returns
    -------
    pd.DataFrame
        Raw training data.
    """
    train_path = input_path / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_path}. "
            "Download from: https://www.kaggle.com/c/expedia-personalized-sort/data"
        )

    print(f"Loading {train_path}...")
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df):,} rows")
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data.

    Returns
    -------
    pd.DataFrame
        Preprocessed data.
    """
    df = df.copy()

    df["prop_review_score"] = df["prop_review_score"].fillna(0)
    df["prop_location_score2"] = df["prop_location_score2"].fillna(
        df["prop_location_score2"].median()
    )
    df["visitor_hist_starrating"] = df["visitor_hist_starrating"].fillna(0)
    df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(0)
    df["orig_destination_distance"] = df["orig_destination_distance"].fillna(0)
    df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(0)

    df["price_usd"] = np.log1p(df["price_usd"].clip(lower=0))
    df["visitor_hist_adr_usd"] = np.log1p(df["visitor_hist_adr_usd"].clip(lower=0))
    df["orig_destination_distance"] = np.log1p(
        df["orig_destination_distance"].clip(lower=0)
    )

    return df


def create_feature_mappings(df: pd.DataFrame) -> dict[str, dict]:
    """
    Create categorical feature mappings.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data.

    Returns
    -------
    dict[str, dict]
        Feature name to value mapping dictionaries.
    """
    mappings = {}

    categorical_cols = [
        "site_id",
        "visitor_location_country_id",
        "srch_destination_id",
        "prop_country_id",
        "prop_id",
    ]

    for col in categorical_cols:
        unique_vals = df[col].unique()
        mappings[col] = {v: i for i, v in enumerate(unique_vals)}
        print(f"  {col}: {len(unique_vals):,} unique values")

    return mappings


def apply_mappings(df: pd.DataFrame, mappings: dict[str, dict]) -> pd.DataFrame:
    """Apply categorical mappings to convert IDs to indices."""
    df = df.copy()
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by search sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    train_ratio : float
        Fraction for training.
    val_ratio : float
        Fraction for validation.
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test dataframes.
    """
    rng = np.random.default_rng(seed)

    search_ids = df["srch_id"].unique()
    rng.shuffle(search_ids)

    n_train = int(len(search_ids) * train_ratio)
    n_val = int(len(search_ids) * val_ratio)

    train_ids = set(search_ids[:n_train])
    val_ids = set(search_ids[n_train : n_train + n_val])
    test_ids = set(search_ids[n_train + n_val :])

    train_df = df[df["srch_id"].isin(train_ids)]
    val_df = df[df["srch_id"].isin(val_ids)]
    test_df = df[df["srch_id"].isin(test_ids)]

    print(f"Train: {len(train_df):,} rows ({len(train_ids):,} sessions)")
    print(f"Val: {len(val_df):,} rows ({len(val_ids):,} sessions)")
    print(f"Test: {len(test_df):,} rows ({len(test_ids):,} sessions)")

    return train_df, val_df, test_df


def extract_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Extract feature arrays for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.

    Returns
    -------
    dict[str, np.ndarray]
        Feature arrays organized by category.
    """
    user_categorical = {
        "visitor_location_country_id": df["visitor_location_country_id"].values,
        "site_id": df["site_id"].values,
    }

    user_numerical = np.column_stack(
        [
            df["visitor_hist_starrating"].values,
            df["visitor_hist_adr_usd"].values,
            df["srch_adults_count"].values,
            df["srch_children_count"].values,
            df["srch_room_count"].values,
            df["srch_length_of_stay"].values,
            df["srch_booking_window"].values,
        ]
    ).astype(np.float32)

    hotel_categorical = {
        "prop_id": df["prop_id"].values,
        "prop_country_id": df["prop_country_id"].values,
        "prop_starrating": df["prop_starrating"].values.clip(0, 5),
        "prop_brand_bool": df["prop_brand_bool"].values,
        "promotion_flag": df["promotion_flag"].values,
    }

    hotel_numerical = np.column_stack(
        [
            df["prop_review_score"].values,
            df["prop_location_score1"].values,
            df["prop_location_score2"].values,
            df["prop_log_historical_price"].values,
            df["price_usd"].values,
            df["orig_destination_distance"].values,
            df["srch_query_affinity_score"].values,
        ]
    ).astype(np.float32)

    context_categorical = {
        "srch_destination_id": df["srch_destination_id"].values,
        "srch_saturday_night_bool": df["srch_saturday_night_bool"].values,
        "random_bool": df["random_bool"].values,
    }

    context_numerical = np.column_stack(
        [
            df["position"].values / 40.0,
        ]
    ).astype(np.float32)

    return {
        "user_categorical": user_categorical,
        "user_numerical": user_numerical,
        "hotel_categorical": hotel_categorical,
        "hotel_numerical": hotel_numerical,
        "context_categorical": context_categorical,
        "context_numerical": context_numerical,
        "click_labels": df["click_bool"].values.astype(np.float32),
        "booking_labels": df["booking_bool"].values.astype(np.float32),
        "session_ids": df["srch_id"].values,
    }


def save_processed(data: dict[str, np.ndarray], output_path: Path, split: str):
    """Save processed features to disk."""
    split_path = output_path / split
    split_path.mkdir(parents=True, exist_ok=True)

    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                np.save(split_path / f"{key}_{subkey}.npy", subvalue)
        else:
            np.save(split_path / f"{key}.npy", value)

    print(f"Saved {split} data to {split_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Expedia dataset for ESMM training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory containing train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Sample fraction for testing (e.g., 0.1 for 10%%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading raw data...")
    df = load_raw_data(args.input)

    if args.sample:
        print(f"Sampling {args.sample * 100:.0f}%% of sessions...")
        search_ids = df["srch_id"].unique()
        rng = np.random.default_rng(args.seed)
        sampled_ids = rng.choice(
            search_ids, size=int(len(search_ids) * args.sample), replace=False
        )
        df = df[df["srch_id"].isin(sampled_ids)]
        print(f"Sampled to {len(df):,} rows")

    print("\nPreprocessing features...")
    df = preprocess_features(df)

    print("\nCreating feature mappings...")
    mappings = create_feature_mappings(df)
    df = apply_mappings(df, mappings)

    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(df, seed=args.seed)

    print("\nExtracting features...")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        features = extract_features(split_df)
        save_processed(features, args.output, split_name)

    np.save(args.output / "mappings.npy", mappings, allow_pickle=True)

    print(f"\nPreprocessing complete! Data saved to {args.output}")
    print("\nLabel distribution:")
    print(f"  Click rate: {df['click_bool'].mean() * 100:.2f}%")
    print(f"  Booking rate: {df['booking_bool'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()