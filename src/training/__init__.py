"""Training modules"""

from .trainer import Trainer
from .train import main as train_main
from .batch_trainer import BatchTrainer, TrainingConfig, get_default_training_configs
from .loss_diagnostics import (
    LossDiagnostics,
    GradientSnapshot,
    ResidualSnapshot,
    DiagnosticsReport,
    ResidualTracker,
    create_diagnostics_callback
)
from .adaptive_loss import (
    AdaptiveLossWeighter,
    GradNormWeighter,
    UncertaintyWeighter,
    ResidualWeighter,
    SoftAdaptWeighter,
    WeightingMethod,
    create_adaptive_weighter
)
from .curriculum_scheduler import (
    CurriculumScheduler,
    CurriculumConfig,
    CurriculumState,
    ConstraintScheduler,
    DynamicCurriculumScheduler,
    ScheduleType,
    create_curriculum_scheduler
)
from .dp_pinn_trainer import (
    DPPINNTrainer,
    TrainingConfig as DPPINNTrainingConfig,
    TrainingHistory,
    train_standard_pinn,
    train_dual_phase_pinn,
)

__all__ = [
    # Core training
    "Trainer",
    "train_main",
    "BatchTrainer",
    "TrainingConfig",
    "get_default_training_configs",
    # Loss diagnostics
    "LossDiagnostics",
    "GradientSnapshot",
    "ResidualSnapshot",
    "DiagnosticsReport",
    "ResidualTracker",
    "create_diagnostics_callback",
    # Adaptive loss weighting
    "AdaptiveLossWeighter",
    "GradNormWeighter",
    "UncertaintyWeighter",
    "ResidualWeighter",
    "SoftAdaptWeighter",
    "WeightingMethod",
    "create_adaptive_weighter",
    # Curriculum learning
    "CurriculumScheduler",
    "CurriculumConfig",
    "CurriculumState",
    "ConstraintScheduler",
    "DynamicCurriculumScheduler",
    "ScheduleType",
    "create_curriculum_scheduler",
    # DP-PINN trainer
    "DPPINNTrainer",
    "DPPINNTrainingConfig",
    "TrainingHistory",
    "train_standard_pinn",
    "train_dual_phase_pinn",
]
