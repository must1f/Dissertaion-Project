"""
Experiment Configuration System

Pydantic-based configuration for reproducible experiments.
Single YAML/JSON config per experiment with validation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib


class ModelType(str, Enum):
    """Available model types"""
    # Baseline models
    LSTM = "lstm"
    GRU = "gru"
    BILSTM = "bilstm"
    ATTENTION_LSTM = "attention_lstm"
    TRANSFORMER = "transformer"

    # PINN variants
    BASELINE_PINN = "baseline_pinn"
    GBM = "gbm"
    OU = "ou"
    BLACK_SCHOLES = "black_scholes"
    GBM_OU = "gbm_ou"
    GLOBAL = "global"

    # Advanced architectures
    STACKED = "stacked"
    RESIDUAL = "residual"


class WindowStrategy(str, Enum):
    """Walk-forward window strategies"""
    EXPANDING = "expanding"  # Anchored - train on all prior data
    ROLLING = "rolling"      # Fixed lookback window


class AdaptiveWeighting(str, Enum):
    """Adaptive loss weighting methods"""
    NONE = "none"
    GRADNORM = "gradnorm"
    UNCERTAINTY = "uncertainty"
    RESIDUAL = "residual"


class CurriculumSchedule(str, Enum):
    """Curriculum learning schedules"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "lstm"
    input_dim: int = 5
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Transformer specific
    n_heads: int = 4
    d_model: int = 64
    dim_feedforward: int = 256

    # PINN specific
    lambda_gbm: float = 0.1
    lambda_ou: float = 0.1
    lambda_bs: float = 0.1
    mu: float = 0.05
    sigma: float = 0.2
    kappa: float = 0.5
    theta: float = 0.0
    r: float = 0.02

    # Advanced PINN
    physics_hidden_size: int = 64
    n_physics_layers: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "none"  # none, cosine, step, plateau

    # Scheduler params
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6

    # Gradient clipping
    gradient_clip_norm: float = 1.0

    # PINN specific
    adaptive_weighting: str = "none"
    curriculum_enabled: bool = False
    curriculum_warmup_epochs: int = 10
    curriculum_ramp_epochs: int = 20
    curriculum_schedule: str = "linear"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DataConfig:
    """Data pipeline configuration"""
    tickers: List[str] = field(default_factory=lambda: ["AAPL"])
    start_date: str = "2015-01-01"
    end_date: str = "2024-01-01"

    # Sequence parameters
    sequence_length: int = 30
    forecast_horizon: int = 1

    # Features
    feature_columns: List[str] = field(default_factory=lambda: [
        "close", "volume", "log_return", "rolling_volatility_20", "rsi_14"
    ])
    target_column: str = "close"

    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Normalization
    normalization: str = "standard"  # standard, minmax, none

    # Walk-forward
    window_strategy: str = "expanding"
    window_train_size: int = 252  # 1 year
    window_test_size: int = 21    # 1 month
    window_step_size: int = 21    # Monthly retraining
    n_folds: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvaluationConfig:
    """Evaluation settings"""
    compute_rolling_metrics: bool = True
    rolling_window_size: int = 63  # 3 months

    # Transaction costs
    transaction_cost: float = 0.003
    risk_free_rate: float = 0.02

    # Statistical tests
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    n_seeds: int = 5

    # Regime analysis
    analyze_regimes: bool = True
    volatility_thresholds: List[float] = field(default_factory=lambda: [0.15, 0.25])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Metadata
    name: str = "experiment"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Output
    output_dir: str = "outputs"
    save_checkpoints: bool = True
    save_predictions: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'created_at': self.created_at,
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'data': self.data.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'seed': self.seed,
            'deterministic': self.deterministic,
            'output_dir': self.output_dir,
            'save_checkpoints': self.save_checkpoints,
            'save_predictions': self.save_predictions,
            'log_level': self.log_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(
            name=data.get('name', 'experiment'),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            author=data.get('author', ''),
            created_at=data.get('created_at', datetime.now().isoformat()),
            model=ModelConfig.from_dict(data.get('model', {})),
            training=TrainingConfig.from_dict(data.get('training', {})),
            data=DataConfig.from_dict(data.get('data', {})),
            evaluation=EvaluationConfig.from_dict(data.get('evaluation', {})),
            seed=data.get('seed', 42),
            deterministic=data.get('deterministic', True),
            output_dir=data.get('output_dir', 'outputs'),
            save_checkpoints=data.get('save_checkpoints', True),
            save_predictions=data.get('save_predictions', True),
            log_level=data.get('log_level', 'INFO'),
        )

    def compute_hash(self) -> str:
        """Compute hash of config for versioning"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def get_output_path(self) -> Path:
        """Get output directory path"""
        return Path(self.output_dir) / f"{self.name}_{self.compute_hash()}"

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Data validation
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            issues.append(f"Data ratios must sum to 1.0, got {total_ratio}")

        if self.data.sequence_length < 1:
            issues.append("Sequence length must be >= 1")

        if self.data.forecast_horizon < 1:
            issues.append("Forecast horizon must be >= 1")

        # Training validation
        if self.training.epochs < 1:
            issues.append("Epochs must be >= 1")

        if self.training.batch_size < 1:
            issues.append("Batch size must be >= 1")

        if self.training.learning_rate <= 0:
            issues.append("Learning rate must be > 0")

        # Model validation
        if self.model.hidden_size < 1:
            issues.append("Hidden size must be >= 1")

        if self.model.num_layers < 1:
            issues.append("Number of layers must be >= 1")

        if self.model.dropout < 0 or self.model.dropout >= 1:
            issues.append("Dropout must be in [0, 1)")

        return issues


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    """
    Load experiment configuration from YAML or JSON file.

    Args:
        path: Path to config file

    Returns:
        ExperimentConfig instance
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    config = ExperimentConfig.from_dict(data)

    # Validate
    issues = config.validate()
    if issues:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {i}" for i in issues))

    return config


def save_experiment_config(config: ExperimentConfig, path: Union[str, Path]):
    """
    Save experiment configuration to YAML or JSON file.

    Args:
        config: ExperimentConfig instance
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """Create default experiment configs for each model type"""
    configs = {}

    # LSTM baseline
    configs['lstm_baseline'] = ExperimentConfig(
        name='lstm_baseline',
        description='LSTM baseline model',
        model=ModelConfig(
            model_type='lstm',
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
    )

    # PINN GBM
    configs['pinn_gbm'] = ExperimentConfig(
        name='pinn_gbm',
        description='PINN with GBM physics constraint',
        model=ModelConfig(
            model_type='gbm',
            hidden_size=128,
            num_layers=2,
            lambda_gbm=0.1
        ),
        training=TrainingConfig(
            curriculum_enabled=True,
            curriculum_warmup_epochs=10,
            curriculum_ramp_epochs=20
        )
    )

    # PINN Global
    configs['pinn_global'] = ExperimentConfig(
        name='pinn_global',
        description='PINN with all physics constraints',
        model=ModelConfig(
            model_type='global',
            hidden_size=128,
            num_layers=2,
            lambda_gbm=0.05,
            lambda_ou=0.05,
            lambda_bs=0.03
        ),
        training=TrainingConfig(
            curriculum_enabled=True,
            adaptive_weighting='gradnorm'
        )
    )

    # Transformer
    configs['transformer'] = ExperimentConfig(
        name='transformer',
        description='Transformer baseline',
        model=ModelConfig(
            model_type='transformer',
            d_model=64,
            n_heads=4,
            num_layers=2,
            dim_feedforward=256
        )
    )

    return configs


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Experiment Configuration Demo")
    print("=" * 60)

    # Create default config
    config = ExperimentConfig(
        name='test_experiment',
        description='Test configuration',
        model=ModelConfig(model_type='lstm', hidden_size=128),
        training=TrainingConfig(epochs=50, learning_rate=0.001),
        data=DataConfig(tickers=['AAPL', 'MSFT'], sequence_length=30)
    )

    print(f"\nConfig name: {config.name}")
    print(f"Config hash: {config.compute_hash()}")
    print(f"Output path: {config.get_output_path()}")

    # Validate
    issues = config.validate()
    print(f"\nValidation: {'PASSED' if not issues else 'FAILED'}")
    for issue in issues:
        print(f"  - {issue}")

    # Save/load roundtrip
    test_path = Path("/tmp/test_config.yaml")
    save_experiment_config(config, test_path)
    loaded = load_experiment_config(test_path)

    print(f"\nSave/load roundtrip: {'PASSED' if config.compute_hash() == loaded.compute_hash() else 'FAILED'}")

    print("\n" + "=" * 60)
