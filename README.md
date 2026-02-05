# ESMMSearch

Two-Tower architecture with ESMM (Entire Space Multi-Task Model) head for hotel search ranking.

## Architecture

The model implements a two-stage ranking architecture:

1. **Two-Tower Encoding**: Separate towers encode user and hotel features into dense representations
2. **Feature Interaction**: User and hotel representations are combined with context features
3. **ESMM Head**: Joint prediction of CTR (Click-Through Rate) and CVR (Conversion Rate)

Key insight: `P(CTCVR) = P(CTR) × P(CVR|CTR)` enables training on the entire impression space, addressing selection bias.

## Installation

```bash
uv sync
```

## Quick Start

### Training with Pseudo Data

```bash
uv run esmmrank --epochs 10 --batch-size 256
```

### Expedia Benchmark

```bash
# Setup (download & preprocess)
make benchmark-setup

# Train
make benchmark-train

# Clean data
make benchmark-clean
```

See [benchmark/README.md](benchmark/README.md) for details.

## CLI Options

```
--data                Dataset: "pseudo" or "expedia" (default: pseudo)
--data-dir            Path to processed data (required for expedia)
--epochs              Number of training epochs (default: 10)
--batch-size          Training batch size (default: 256)
--lr                  Learning rate (default: 0.001)
--train-sessions      Number of training sessions (default: 8000)
--val-sessions        Number of validation sessions (default: 1000)
--test-sessions       Number of test sessions (default: 1000)
--hotels-per-session  Hotels per search session (default: 20)
--seed                Random seed (default: 42)
--cpu                 Force CPU usage
--save-path           Path to save model checkpoint
```

## Python API

```python
from esmmrank import (
    ModelConfig,
    TrainingConfig,
    TwoTowerESMM,
    Trainer,
    create_pseudo_loaders,
)

# Configure model
model_config = ModelConfig()
model = TwoTowerESMM(model_config)

# Create data loaders
train_loader, val_loader, test_loader = create_pseudo_loaders(
    feature_config=model_config.feature,
    batch_size=256,
)

# Train
training_config = TrainingConfig(num_epochs=10, device="mps")
trainer = Trainer(model, training_config)
history = trainer.fit(train_loader, val_loader)
```

## Configuration via Environment Variables

Training config can be set via environment variables with `ESMM_` prefix:

```bash
ESMM_BATCH_SIZE=512 ESMM_LEARNING_RATE=0.0001 uv run esmmrank
```

See available variables in: `src/esmmrank/config.py` 
