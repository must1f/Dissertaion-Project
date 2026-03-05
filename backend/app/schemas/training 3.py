"""Pydantic schemas for training endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from backend.app.core.config import ALLOWED_TICKERS, DEFAULT_TICKER, validate_ticker


class TrainingStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingRequest(BaseModel):
    """Request to start training. Only S&P 500 data is supported."""

    model_type: str = Field(..., description="Type of model to train")
    ticker: str = Field(
        default=DEFAULT_TICKER,
        description=f"Stock ticker for training data (only {', '.join(ALLOWED_TICKERS)} supported)"
    )

    # Training parameters
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=8, le=256)
    learning_rate: float = Field(default=0.001, ge=1e-6, le=0.1)
    sequence_length: int = Field(default=120, ge=10, le=300)  # Increased for better patterns

    # Data date range
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for training data (YYYY-MM-DD). Defaults to 10 years ago."
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date for training data (YYYY-MM-DD). Defaults to today."
    )

    # Multi-ticker training (matches terminal training approach)
    use_multi_ticker: bool = Field(
        default=False,
        description="Train on top 10 S&P 500 stocks instead of single ticker"
    )

    # Model architecture (increased for research-grade training)
    hidden_dim: int = Field(default=256, ge=32, le=1024)  # Increased from 128
    num_layers: int = Field(default=3, ge=1, le=8)  # Increased from 2
    dropout: float = Field(default=0.15, ge=0.0, le=0.5)  # Reduced slightly

    # Physics settings (for PINN models)
    enable_physics: bool = Field(default=True)
    physics_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    # Training settings
    early_stopping_patience: int = Field(default=30, ge=5, le=100)  # Increased from 10
    use_curriculum: bool = Field(default=False)
    save_checkpoints: bool = Field(default=True)

    # Research mode - disables early stopping for fair comparison
    research_mode: bool = Field(
        default=True,
        description="Enable research mode: disables early stopping, trains full epochs"
    )

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        """Validate that only allowed tickers are used for training."""
        return validate_ticker(v)


class ResearchMetrics(BaseModel):
    """Research-grade metrics for model evaluation."""

    # Prediction metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    mse: Optional[float] = None
    r2: Optional[float] = None
    directional_accuracy: Optional[float] = None

    # Financial metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    annualized_return: Optional[float] = None
    total_return: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None


class TrainingJobInfo(BaseModel):
    """Information about a training job."""

    job_id: str
    model_type: str
    ticker: str
    status: TrainingStatus

    # Progress
    current_epoch: int = 0
    total_epochs: int
    progress_percent: float = 0.0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None

    # Current metrics
    current_train_loss: Optional[float] = None
    current_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None

    # Research-grade metrics (computed on test set at end of training)
    test_metrics: Optional[ResearchMetrics] = None

    # Saved model info
    saved_model_name: Optional[str] = None

    # Configuration
    config: Dict[str, Any] = {}

    # Optional post-training evaluation metrics (test set)
    final_metrics: Optional[Dict[str, Any]] = None  # currently computed on demand, not during training


class TrainingStartResponse(BaseModel):
    """Response when training is started."""

    success: bool
    job_id: str
    message: str
    websocket_url: str


class TrainingStopResponse(BaseModel):
    """Response when training is stopped."""

    success: bool
    job_id: str
    message: str
    final_status: TrainingStatus


class TrainingStatusResponse(BaseModel):
    """Response for training status query."""

    job: TrainingJobInfo
    history: Optional[Dict[str, List[float]]] = None


class EpochMetrics(BaseModel):
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float

    # Detailed losses
    data_loss: Optional[float] = None
    physics_loss: Optional[float] = None

    # Physics parameters (for PINN)
    theta: Optional[float] = None
    gamma: Optional[float] = None
    temperature: Optional[float] = None

    # Timing
    epoch_time_seconds: float


class TrainingHistoryResponse(BaseModel):
    """Response with training history."""

    job_id: str
    model_type: str
    epochs: List[EpochMetrics]
    best_epoch: int
    best_val_loss: float
    final_metrics: Dict[str, float]


class TrainingRunListResponse(BaseModel):
    """Response listing training runs."""

    runs: List[TrainingJobInfo]
    total: int
    page: int
    page_size: int


# WebSocket message schemas
class WSTrainingUpdate(BaseModel):
    """WebSocket message for training updates."""

    type: str = "training_update"
    job_id: str
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    best_val_loss: float
    learning_rate: float
    epoch_time: float

    # Physics parameters (optional)
    physics_params: Optional[Dict[str, float]] = None


class WSTrainingComplete(BaseModel):
    """WebSocket message when training completes."""

    type: str = "training_complete"
    job_id: str
    status: TrainingStatus
    final_metrics: Dict[str, float]
    checkpoint_path: Optional[str] = None


class WSTrainingError(BaseModel):
    """WebSocket message for training errors."""

    type: str = "training_error"
    job_id: str
    error: str
    traceback: Optional[str] = None


# ============== Batch Training Schemas ==============

class ModelTrainingConfig(BaseModel):
    """Configuration for a single model in batch training."""

    model_key: str = Field(..., description="Unique model identifier (e.g., 'lstm', 'pinn_gbm')")
    enabled: bool = Field(default=True, description="Whether to train this model")

    # Optional per-model overrides (uses global if not set)
    epochs: Optional[int] = Field(None, ge=1, le=1000)
    learning_rate: Optional[float] = Field(None, ge=1e-6, le=0.1)
    batch_size: Optional[int] = Field(None, ge=8, le=256)
    hidden_dim: Optional[int] = Field(None, ge=32, le=512)
    num_layers: Optional[int] = Field(None, ge=1, le=6)
    dropout: Optional[float] = Field(None, ge=0.0, le=0.5)


class BatchTrainingRequest(BaseModel):
    """Request to start batch training of multiple models.

    Research-grade defaults:
    - 10 years of data (fetched fresh)
    - Deep models (hidden_dim=512, num_layers=4)
    - Small batch size (16) for better gradient estimates
    - No early stopping (research_mode=True)
    - 100 epochs minimum
    """

    # Models to train
    models: List[ModelTrainingConfig] = Field(..., min_length=1, max_length=20)

    # Global training parameters (research-grade defaults)
    ticker: str = Field(default=DEFAULT_TICKER)
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=8, le=256)  # Smaller for better gradients
    learning_rate: float = Field(default=0.001, ge=1e-6, le=0.1)
    sequence_length: int = Field(default=180, ge=10, le=500)  # 180 days (~6 months lookback)

    # Data date range (10 years by default for research-grade training)
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for training data (YYYY-MM-DD). Defaults to 10 years ago."
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date for training data (YYYY-MM-DD). Defaults to today."
    )

    # Multi-ticker training (matches terminal training approach)
    use_multi_ticker: bool = Field(
        default=False,
        description="Train on top 10 S&P 500 stocks instead of single ticker"
    )

    # Force data refresh (ensure fresh 10-year data)
    force_refresh: bool = Field(
        default=True,
        description="Force re-fetch data even if cached (ensures fresh 10-year coverage)"
    )

    # Model architecture defaults (deep models for decade-long data)
    hidden_dim: int = Field(default=512, ge=32, le=1024)  # Deep: 512
    num_layers: int = Field(default=4, ge=1, le=8)  # Deep: 4 layers
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)  # Lower dropout for deep models

    # Training settings
    gradient_clip_norm: float = Field(default=1.0, ge=0.1, le=10.0)
    scheduler_patience: int = Field(default=10, ge=3, le=30)  # More patience
    early_stopping_patience: int = Field(default=50, ge=0, le=100)  # Much higher if used

    # Research mode - disables early stopping for fair model comparison
    research_mode: bool = Field(
        default=True,
        description="Enable research mode: disables early stopping, all models train for full epochs"
    )

    # Physics settings
    enable_physics: bool = Field(default=True)
    physics_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    # Checkpoint settings
    save_checkpoints: bool = Field(default=True)

    @field_validator('ticker')
    @classmethod
    def validate_allowed_ticker(cls, v: str) -> str:
        return validate_ticker(v)


class BatchJobStatus(BaseModel):
    """Status of a single model within a batch training job."""

    model_key: str
    model_name: str
    model_type: str  # 'baseline', 'pinn', 'advanced'
    status: TrainingStatus
    current_epoch: int = 0
    total_epochs: int = 0
    # Batch-level progress for real-time updates within each epoch
    current_batch: int = 0
    total_batches: int = 0
    batch_loss: Optional[float] = None  # Loss of most recent batch
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    val_rmse: Optional[float] = None
    val_mae: Optional[float] = None
    val_mape: Optional[float] = None
    val_r2: Optional[float] = None
    val_directional_accuracy: Optional[float] = None
    data_loss: Optional[float] = None
    physics_loss: Optional[float] = None
    learning_rate: float = 0.001
    progress_percent: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None

    # Research-grade metrics (computed on test set at end of training)
    test_metrics: Optional[ResearchMetrics] = None


class BatchTrainingJobInfo(BaseModel):
    """Information about a batch training job."""

    batch_id: str
    status: TrainingStatus
    total_models: int
    completed_models: int
    failed_models: int
    current_model: Optional[str] = None
    models: List[BatchJobStatus]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    overall_progress: float = 0.0

    # Global config used
    config: Dict[str, Any] = {}


class BatchTrainingStartResponse(BaseModel):
    """Response when batch training is started."""

    success: bool
    batch_id: str
    message: str
    total_models: int
    model_keys: List[str]
    websocket_url: str


class BatchTrainingStatusResponse(BaseModel):
    """Response for batch training status query."""

    batch: BatchTrainingJobInfo
    history: Dict[str, List[Dict[str, float]]] = {}  # model_key -> epoch history


class WSBatchTrainingUpdate(BaseModel):
    """WebSocket message for batch training updates."""

    type: str = "batch_training_update"
    batch_id: str
    model_key: str
    model_name: str
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    best_val_loss: float
    learning_rate: float
    data_loss: Optional[float] = None
    physics_loss: Optional[float] = None
    overall_progress: float
    completed_models: int
    total_models: int


class WSBatchModelComplete(BaseModel):
    """WebSocket message when a model in batch completes."""

    type: str = "batch_model_complete"
    batch_id: str
    model_key: str
    model_name: str
    status: TrainingStatus
    final_metrics: Dict[str, float]
    remaining_models: int


class WSBatchTrainingComplete(BaseModel):
    """WebSocket message when entire batch training completes."""

    type: str = "batch_training_complete"
    batch_id: str
    status: TrainingStatus
    total_models: int
    completed_models: int
    failed_models: int
    summary: List[Dict[str, Any]]


# Available model definitions with default physics constraints
AVAILABLE_MODELS = {
    # Baseline models
    "lstm": {
        "name": "LSTM",
        "type": "baseline",
        "description": "Long Short-Term Memory network",
        "physics_constraints": None
    },
    "gru": {
        "name": "GRU",
        "type": "baseline",
        "description": "Gated Recurrent Unit network",
        "physics_constraints": None
    },
    "bilstm": {
        "name": "BiLSTM",
        "type": "baseline",
        "description": "Bidirectional LSTM",
        "physics_constraints": None
    },
    "transformer": {
        "name": "Transformer",
        "type": "baseline",
        "description": "Multi-head attention transformer",
        "physics_constraints": None
    },
    # PINN variants
    "pinn_baseline": {
        "name": "PINN Baseline",
        "type": "pinn",
        "description": "Pure data-driven (no physics)",
        "physics_constraints": {"lambda_gbm": 0.0, "lambda_bs": 0.0, "lambda_ou": 0.0}
    },
    "pinn_gbm": {
        "name": "PINN GBM",
        "type": "pinn",
        "description": "Geometric Brownian Motion constraint",
        "physics_constraints": {"lambda_gbm": 0.1, "lambda_bs": 0.0, "lambda_ou": 0.0}
    },
    "pinn_ou": {
        "name": "PINN OU",
        "type": "pinn",
        "description": "Ornstein-Uhlenbeck mean-reversion",
        "physics_constraints": {"lambda_gbm": 0.0, "lambda_bs": 0.0, "lambda_ou": 0.1}
    },
    "pinn_black_scholes": {
        "name": "PINN Black-Scholes",
        "type": "pinn",
        "description": "No-arbitrage PDE constraint",
        "physics_constraints": {"lambda_gbm": 0.0, "lambda_bs": 0.1, "lambda_ou": 0.0}
    },
    "pinn_gbm_ou": {
        "name": "PINN GBM+OU",
        "type": "pinn",
        "description": "Combined trend + mean-reversion",
        "physics_constraints": {"lambda_gbm": 0.05, "lambda_bs": 0.0, "lambda_ou": 0.05}
    },
    "pinn_global": {
        "name": "PINN Global",
        "type": "pinn",
        "description": "All physics constraints combined",
        "physics_constraints": {"lambda_gbm": 0.05, "lambda_bs": 0.03, "lambda_ou": 0.05, "lambda_langevin": 0.02}
    },
    # Advanced baseline models
    "attention_lstm": {
        "name": "Attention LSTM",
        "type": "baseline",
        "description": "LSTM with attention mechanism for long-term dependencies",
        "physics_constraints": None
    },
    # Advanced PINN architectures
    "stacked_pinn": {
        "name": "StackedPINN",
        "type": "advanced",
        "description": "Physics encoder + parallel LSTM/GRU + attention fusion",
        "physics_constraints": {"lambda_gbm": 0.1, "lambda_ou": 0.1}
    },
    "residual_pinn": {
        "name": "ResidualPINN",
        "type": "advanced",
        "description": "Base LSTM + physics-informed correction network",
        "physics_constraints": {"lambda_gbm": 0.1, "lambda_ou": 0.1}
    },
}


class AvailableModelsResponse(BaseModel):
    """Response listing all available models for batch training."""

    models: Dict[str, Dict[str, Any]]
    total: int
    by_type: Dict[str, int]
