"""
Configuration management using Pydantic for validation and type safety.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "pinn_finance"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "pinn_user"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class APIConfig(BaseModel):
    """API configuration for data sources"""
    alpha_vantage_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY")
    )

    @validator("alpha_vantage_key")
    def validate_alpha_vantage(cls, v):
        if v and v == "your_alpha_vantage_key_here":
            return None
        return v


class DataConfig(BaseModel):
    """Data processing configuration"""
    start_date: str = Field(default_factory=lambda: os.getenv("START_DATE", "2014-01-01"))
    end_date: str = Field(default_factory=lambda: os.getenv("END_DATE", "2024-01-01"))
    train_ratio: float = Field(default_factory=lambda: float(os.getenv("TRAIN_RATIO", "0.7")))
    val_ratio: float = Field(default_factory=lambda: float(os.getenv("VAL_RATIO", "0.15")))
    test_ratio: float = Field(default_factory=lambda: float(os.getenv("TEST_RATIO", "0.15")))

    sequence_length: int = 60  # Number of time steps to look back
    forecast_horizon: int = 1  # Number of steps to predict ahead

    # S&P 500 constituents (top 50 for MVP, can expand)
    tickers: list[str] = Field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'MRK',
        'COST', 'PEP', 'KO', 'AVGO', 'ADBE', 'MCD', 'TMO', 'CSCO', 'ACN', 'LLY',
        'NKE', 'DHR', 'ABT', 'VZ', 'TXN', 'NEE', 'CRM', 'WFC', 'ORCL', 'DIS',
        'CMCSA', 'PM', 'BMY', 'INTC', 'QCOM', 'UPS', 'HON', 'MS', 'RTX', 'AMGN'
    ])

    @validator("train_ratio", "val_ratio", "test_ratio")
    def validate_ratio(cls, v):
        if not 0 < v < 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v


class ModelConfig(BaseModel):
    """Model architecture configuration"""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    bidirectional: bool = True

    # Transformer specific
    num_heads: int = 8
    feedforward_dim: int = 512

    # PINN specific
    physics_hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])


class TrainingConfig(BaseModel):
    """Training configuration"""
    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))
    learning_rate: float = Field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.001")))
    epochs: int = Field(default_factory=lambda: int(os.getenv("EPOCHS", "100")))
    early_stopping_patience: int = Field(
        default_factory=lambda: int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
    )

    # Physics loss weights
    lambda_physics: float = Field(default_factory=lambda: float(os.getenv("LAMBDA_PHYSICS", "0.1")))
    lambda_gbm: float = Field(default_factory=lambda: float(os.getenv("LAMBDA_GBM", "0.1")))
    lambda_bs: float = Field(default_factory=lambda: float(os.getenv("LAMBDA_BS", "0.1")))
    lambda_ou: float = Field(default_factory=lambda: float(os.getenv("LAMBDA_OU", "0.1")))
    lambda_langevin: float = Field(default_factory=lambda: float(os.getenv("LAMBDA_LANGEVIN", "0.1")))

    # Device
    device: Literal["cuda", "cpu"] = Field(
        default_factory=lambda: os.getenv("DEVICE", "cuda")
    )

    # Reproducibility
    random_seed: int = Field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))


class TradingConfig(BaseModel):
    """Trading agent configuration"""
    initial_capital: float = Field(
        default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "100000"))
    )
    transaction_cost: float = Field(
        default_factory=lambda: float(os.getenv("TRANSACTION_COST", "0.001"))
    )
    max_position_size: float = Field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "0.2"))
    )
    stop_loss: float = Field(default_factory=lambda: float(os.getenv("STOP_LOSS", "0.02")))
    take_profit: float = Field(default_factory=lambda: float(os.getenv("TAKE_PROFIT", "0.05")))

    # Risk management
    max_drawdown_limit: float = 0.2  # Max 20% drawdown before stopping
    confidence_threshold: float = 0.6  # Min confidence for trades


class Config(BaseModel):
    """Main configuration object"""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    checkpoint_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "checkpoints")
    log_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "parquet").mkdir(exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance (singleton pattern)"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config():
    """Reset global configuration (useful for testing)"""
    global _config
    _config = None
