from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureConfig(BaseModel):
    """Configuration for feature dimensions and vocabulary sizes."""

    user_categorical_dims: dict[str, int] = Field(
        default_factory=lambda: {
            "device_type": 4,
            "user_segment": 10,
            "country": 200,
        }
    )
    user_numerical_dim: int = 16

    hotel_categorical_dims: dict[str, int] = Field(
        default_factory=lambda: {
            "star_rating": 6,
            "property_type": 20,
            "city": 500,
            "amenities": 50,
        }
    )
    hotel_numerical_dim: int = 24

    context_categorical_dims: dict[str, int] = Field(
        default_factory=lambda: {
            "day_of_week": 7,
            "month": 12,
            "is_weekend": 2,
            "sort_option": 5,
        }
    )
    context_numerical_dim: int = 8

    embedding_dim: int = 32


class TowerConfig(BaseModel):
    """Configuration for tower architecture."""

    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128])
    output_dim: int = 64
    dropout: float = 0.2
    use_batch_norm: bool = True


class HeadConfig(BaseModel):
    """Configuration for ESMM prediction heads."""

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    use_batch_norm: bool = True


class ModelConfig(BaseModel):
    """Complete model configuration."""

    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    user_tower: TowerConfig = Field(default_factory=TowerConfig)
    hotel_tower: TowerConfig = Field(default_factory=TowerConfig)
    ctr_head: HeadConfig = Field(default_factory=HeadConfig)
    cvr_head: HeadConfig = Field(default_factory=HeadConfig)


class TrainingConfig(BaseSettings):
    """
    Training hyperparameters.

    Can be configured via environment variables with ESMM_ prefix.
    """

    model_config = SettingsConfigDict(env_prefix="ESMM_", env_nested_delimiter="__")

    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 10
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    ctr_loss_weight: float = 1.0
    ctcvr_loss_weight: float = 1.0
    ipsw_enabled: bool = False
    ipsw_eta: float = 1.0
    gradient_clip_norm: float = 1.0
    device: str = "cpu"

    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6