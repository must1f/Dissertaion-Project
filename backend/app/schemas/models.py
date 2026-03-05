"""Pydantic schemas for model endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Types of available models."""

    # Baseline models
    LSTM = "lstm"
    GRU = "gru"
    BILSTM = "bilstm"
    ATTENTION_LSTM = "attention_lstm"
    TRANSFORMER = "transformer"

    # PINN variants
    PINN_BASELINE = "pinn_baseline"
    PINN_GBM = "pinn_gbm"
    PINN_OU = "pinn_ou"
    PINN_BLACK_SCHOLES = "pinn_black_scholes"
    PINN_GBM_OU = "pinn_gbm_ou"
    PINN_GLOBAL = "pinn_global"

    # Advanced models
    STACKED_PINN = "stacked_pinn"
    RESIDUAL_PINN = "residual_pinn"


class ModelStatus(str, Enum):
    """Model training status."""

    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class PhysicsParameters(BaseModel):
    """Learned physics parameters from PINN models."""

    theta: Optional[float] = Field(
        None,
        description="OU mean reversion speed"
    )
    gamma: Optional[float] = Field(
        None,
        description="Langevin friction coefficient"
    )
    temperature: Optional[float] = Field(
        None,
        description="Langevin temperature"
    )
    mu: Optional[float] = Field(
        None,
        description="GBM drift parameter"
    )
    sigma: Optional[float] = Field(
        None,
        description="GBM/OU volatility parameter"
    )


class ModelInfo(BaseModel):
    """Information about a model."""

    model_key: str
    model_type: str
    display_name: str
    description: Optional[str] = None
    status: ModelStatus
    is_pinn: bool = False

    # Architecture info
    architecture: Optional[Dict[str, Any]] = None
    input_dim: int = 5
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None

    # Training info
    trained_at: Optional[datetime] = None
    training_epochs: Optional[int] = None
    best_val_loss: Optional[float] = None

    # Physics parameters (for PINN models)
    physics_parameters: Optional[PhysicsParameters] = None

    # File info
    checkpoint_path: Optional[str] = None
    file_size_mb: Optional[float] = None


class ModelListResponse(BaseModel):
    """Response for listing models."""

    models: List[ModelInfo]
    total: int
    trained_count: int
    pinn_count: int


class ModelDetailResponse(BaseModel):
    """Detailed model information."""

    model: ModelInfo
    training_history: Optional[Dict[str, List[float]]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None


class ModelWeightsInfo(BaseModel):
    """Information about model weights."""

    model_key: str
    total_parameters: int
    trainable_parameters: int
    layer_info: List[Dict[str, Any]]
    checkpoint_info: Optional[Dict[str, Any]] = None


class ModelComparisonItem(BaseModel):
    """Single model in comparison."""

    model_key: str
    display_name: str
    is_pinn: bool
    metrics: Dict[str, float]
    physics_parameters: Optional[PhysicsParameters] = None


class ModelComparisonResponse(BaseModel):
    """Response for model comparison."""

    models: List[ModelComparisonItem]
    metric_names: List[str]
    best_by_metric: Dict[str, str]  # metric_name -> model_key
