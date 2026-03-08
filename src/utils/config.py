"""
Configuration management using Pydantic for validation and type safety.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from dotenv import load_dotenv


def get_dynamic_date_range(years: int = 10) -> tuple[str, str]:
    """
    Calculate dynamic date range for the last N years of data.

    Returns:
        tuple: (start_date, end_date) as ISO format strings
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

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
    model_config = ConfigDict(extra="ignore")
    """API configuration for data sources"""
    alpha_vantage_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY")
    )

    @field_validator("alpha_vantage_key")
    def validate_alpha_vantage(cls, v):
        if v and v == "your_alpha_vantage_key_here":
            return None
        return v


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """Data processing configuration"""
    start_date: str = Field(default_factory=lambda: os.getenv("START_DATE") or get_dynamic_date_range(10)[0])
    end_date: str = Field(default_factory=lambda: os.getenv("END_DATE") or get_dynamic_date_range(10)[1])
    interval: str = Field(default_factory=lambda: os.getenv("INTERVAL", "1d"))

    train_ratio: float = Field(default_factory=lambda: float(os.getenv("TRAIN_RATIO", "0.7")))
    val_ratio: float = Field(default_factory=lambda: float(os.getenv("VAL_RATIO", "0.15")))
    test_ratio: float = Field(default_factory=lambda: float(os.getenv("TEST_RATIO", "0.15")))

    sequence_length: int = 60  # Number of time steps to look back
    forecast_horizon: int = 1  # Number of steps to predict ahead

    universe_name: str = Field(default_factory=lambda: os.getenv("UNIVERSE_NAME", "core_multi_asset"))
    target_symbol: str = Field(default_factory=lambda: os.getenv("TARGET_SYMBOL", "SPY"))
    target_type: str = Field(default_factory=lambda: os.getenv("TARGET_TYPE", "next_day_log_return"))
    target_column: str = Field(default_factory=lambda: os.getenv("TARGET_COLUMN", "target"))
    target_vol_window: int = Field(default_factory=lambda: int(os.getenv("TARGET_VOL_WINDOW", "5")))
    price_column: str = Field(default_factory=lambda: os.getenv("PRICE_COLUMN", "adjusted_close"))
    calendar: str = Field(default_factory=lambda: os.getenv("CALENDAR", "NYSE"))
    master_calendar_holidays: list[str] = Field(default_factory=list)
    cache_dir: str = Field(default_factory=lambda: os.getenv("CACHE_DIR", "cache"))
    cache_ttl_days: int = Field(default_factory=lambda: int(os.getenv("CACHE_TTL_DAYS", "3")))
    force_refresh: bool = Field(default_factory=lambda: os.getenv("FORCE_REFRESH", "false").lower() == "true")

    # Calendar/fill policy
    default_forward_fill_limit: int = Field(default_factory=lambda: int(os.getenv("DEFAULT_FFILL_LIMIT", "1")))
    per_symbol_forward_fill_limit: dict[str, int] = Field(default_factory=lambda: {
        "^VIX": 0,
        "^TNX": 0,
    })
    disable_qa: bool = Field(default_factory=lambda: os.getenv("DISABLE_QA", "false").lower() == "true")
    missing_policy: str = Field(default_factory=lambda: os.getenv("MISSING_POLICY", "leakage_safe"))
    scaling_policy: str = Field(default_factory=lambda: os.getenv("SCALING_POLICY", "train_only"))

    # Multi-asset baseline universe (stable core + optional overlays)
    base_universe: list[str] = Field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLE', '^VIX', '^TNX'
    ])
    optional_universe: list[str] = Field(default_factory=lambda: ['GC=F', 'CL=F'])
    include_optional_assets: bool = Field(default_factory=lambda: os.getenv("INCLUDE_OPTIONAL_ASSETS", "true").lower() == "true")
    tickers: list[str] = Field(default_factory=list)

    base_feature_columns: list[str] = Field(default_factory=lambda: [
        "adj_return_1d", "adj_return_5d", "rolling_vol_10", "rolling_vol_20",
        "momentum_20_z", "momentum_60_z", "vix_level", "vix_change",
        "tnx_yield", "commodity_gc_ret", "commodity_cl_ret",
        "cross_spy_qqq_spread", "cross_spy_iwm_spread"
    ])

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    def validate_ratio(cls, v):
        if not 0 < v < 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def validate_splits(self):
        # Ensure stable universe defaults if user did not provide a custom list
        if not self.tickers:
            combined = list(dict.fromkeys(self.base_universe + (self.optional_universe if self.include_optional_assets else [])))
            self.tickers = combined

        # Normalize common aliases to caret-based symbols
        alias_map = {
            "VIX": "^VIX",
            "TNX": "^TNX",
            "^TNX": "^TNX",
            "^VIX": "^VIX",
        }
        normalized: list[str] = []
        for t in self.tickers:
            normalized.append(alias_map.get(t, t))
        self.tickers = list(dict.fromkeys(normalized))

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test ratios must sum to 1.0 (got {total:.4f})")
        # ensure date order
        try:
            if datetime.fromisoformat(self.end_date) < datetime.fromisoformat(self.start_date):
                raise ValueError("end_date must be after start_date")
        except ValueError:
            # datetime parsing errors will surface with clearer context
            raise
        return self


class ModelConfig(BaseModel):
    """Model architecture configuration"""
    hidden_dim: int = 128
    num_layers: int = 2  # Matches actual model instantiation in model_registry.py
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
    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "cuda"))

    # Reproducibility
    random_seed: int = Field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))


class ResearchConfig(BaseModel):
    """
    Research-compliant training configuration for fair model comparison.

    All parameters are locked to ensure reproducible and comparable results
    across different model architectures. This follows research best practices
    for ablation studies and model comparisons.
    """
    # LOCKED TRAINING PARAMETERS - Same for all models
    epochs: int = Field(default=100, description="Fixed number of epochs for all models")
    batch_size: int = Field(default=16, description="Small batch size for better gradient estimates")
    learning_rate: float = Field(default=0.0005, description="Lower LR for stability with deep models")

    # Early stopping disabled for research mode - all models train for full epochs
    use_early_stopping: bool = Field(default=False, description="Disable early stopping for fair comparison")

    # Regularization - CRITICAL for preventing overfitting
    weight_decay: float = Field(default=1e-4, description="L2 regularization weight decay")

    # Learning rate scheduler parameters (same for all)
    scheduler_patience: int = Field(default=10, description="LR scheduler patience (more patience for research)")
    scheduler_factor: float = Field(default=0.5, description="LR scheduler reduction factor")

    # Gradient clipping (same for all)
    gradient_clip_norm: float = Field(default=1.0, description="Gradient clipping norm")

    # Reproducibility
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # Data split ratios (same for all)
    train_ratio: float = Field(default=0.7, description="Training set ratio")
    val_ratio: float = Field(default=0.15, description="Validation set ratio")
    test_ratio: float = Field(default=0.15, description="Test set ratio")

    # Sequence parameters (same for all)
    sequence_length: int = Field(default=180, description="6 months lookback for research-grade training")
    forecast_horizon: int = Field(default=1, description="Prediction horizon")

    # Model architecture (same for all) - deep models for complex patterns
    hidden_dim: int = Field(default=512, description="Deep hidden dimension for 10-year data")
    num_layers: int = Field(default=4, description="4 layers for hierarchical features")
    dropout: float = Field(default=0.15, description="Moderate dropout for deep models")

    # Financial metrics parameters (same for all)
    transaction_cost: float = Field(default=0.003, description="Transaction cost (0.3%)")
    risk_free_rate: float = Field(default=0.02, description="Annual risk-free rate (2%)")
    periods_per_year: int = Field(default=252, description="Trading days per year")

    def get_training_params(self) -> Dict:
        """Get locked training parameters as dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'use_early_stopping': self.use_early_stopping,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_factor': self.scheduler_factor,
            'gradient_clip_norm': self.gradient_clip_norm,
            'random_seed': self.random_seed,
        }

    def get_data_params(self) -> Dict:
        """Get locked data parameters as dictionary."""
        return {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
        }

    def get_model_params(self) -> Dict:
        """Get locked model architecture parameters."""
        return {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
        }

    def get_financial_params(self) -> Dict:
        """Get locked financial evaluation parameters."""
        return {
            'transaction_cost': self.transaction_cost,
            'risk_free_rate': self.risk_free_rate,
            'periods_per_year': self.periods_per_year,
        }


# Global research config instance
_research_config: Optional[ResearchConfig] = None


def get_research_config() -> ResearchConfig:
    """Get global research configuration instance (singleton pattern)"""
    global _research_config
    if _research_config is None:
        _research_config = ResearchConfig()
    return _research_config


def reset_research_config():
    """Reset research configuration (useful for testing)"""
    global _research_config
    _research_config = None


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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
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

    def model_post_init(self, __context: Any) -> None:
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "raw_cache").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "parquet").mkdir(exist_ok=True)
        (self.data_dir / "splits").mkdir(exist_ok=True)
        (self.data_dir / "artifacts").mkdir(exist_ok=True)
        (self.data_dir / self.data.cache_dir).mkdir(parents=True, exist_ok=True)
        (self.project_root / self.data.cache_dir).mkdir(parents=True, exist_ok=True)


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
