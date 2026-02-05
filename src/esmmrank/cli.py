import argparse
from pathlib import Path

import torch

from .config import FeatureConfig, ModelConfig, TrainingConfig
from .dataset import create_pseudo_loaders
from .model import TwoTowerESMM
from .trainer import Trainer


def get_device(force_cpu: bool = False) -> str:
    """
    Determine the best available device.

    Parameters
    ----------
    force_cpu : bool
        If True, always return 'cpu'.

    Returns
    -------
    str
        Device string ('cuda', 'mps', or 'cpu').
    """
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args: argparse.Namespace) -> None:
    """
    Run model training.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """
    device = get_device(force_cpu=args.cpu)
    print(f"Using device: {device}")

    if args.data == "expedia":
        from .dataset import create_expedia_loaders, get_expedia_feature_config

        if not args.data_dir:
            raise ValueError("--data-dir required for Expedia dataset")

        print(f"Loading Expedia data from {args.data_dir}...")
        feature_config = FeatureConfig(**get_expedia_feature_config())
        train_loader, val_loader, test_loader = create_expedia_loaders(
            data_dir=Path(args.data_dir),
            batch_size=args.batch_size,
        )
    else:
        print("Generating pseudo data...")
        feature_config = FeatureConfig()
        train_loader, val_loader, test_loader = create_pseudo_loaders(
            feature_config=feature_config,
            batch_size=args.batch_size,
            n_train_sessions=args.train_sessions,
            n_val_sessions=args.val_sessions,
            n_test_sessions=args.test_sessions,
            hotels_per_session=args.hotels_per_session,
            seed=args.seed,
        )

    model_config = ModelConfig(feature=feature_config)

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=device,
    )

    print(f"Train samples: {len(train_loader.dataset):,}")  # noqa
    print(f"Val samples: {len(val_loader.dataset):,}")  # noqa
    print(f"Test samples: {len(test_loader.dataset):,}")  # noqa

    print("Initializing model...")
    model = TwoTowerESMM(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    trainer = Trainer(model, training_config)

    print("\nStarting training...")
    print("-" * 60)
    trainer.fit(
        train_loader,
        val_loader,
        verbose=True,
        log_path=args.log_path,
        tensorboard_dir=args.tensorboard_dir,
    )
    print("-" * 60)

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader, prefix="test")
    print(f"  NDCG@38 (competition): {test_metrics['test_ctcvr_ndcg@38']:.4f}")
    print(f"  NDCG@10:               {test_metrics['test_ctcvr_ndcg@10']:.4f}")
    print(f"  CTR PR-AUC:            {test_metrics['test_ctr_prauc']:.4f}")
    print(f"  CTCVR PR-AUC:          {test_metrics['test_ctcvr_prauc']:.4f}")

    if args.save_path:
        trainer.save_checkpoint(args.save_path)
        print(f"\nModel saved to {args.save_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ESMM Search Ranking Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        choices=["pseudo", "expedia"],
        default="pseudo",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to processed data directory (required for expedia)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--train-sessions",
        type=int,
        default=8000,
        help="Number of training sessions (pseudo data only)",
    )
    parser.add_argument(
        "--val-sessions",
        type=int,
        default=1000,
        help="Number of validation sessions (pseudo data only)",
    )
    parser.add_argument(
        "--test-sessions",
        type=int,
        default=1000,
        help="Number of test sessions (pseudo data only)",
    )
    parser.add_argument(
        "--hotels-per-session",
        type=int,
        default=20,
        help="Number of hotels per search session (pseudo data only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to save training metrics (JSON)",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()