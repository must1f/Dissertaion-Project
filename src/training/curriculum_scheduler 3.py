"""
Curriculum Learning Scheduler for PINN Training

Implements gradual introduction of physics constraints:
1. Warmup phase: Train on data loss only
2. Ramp phase: Gradually increase physics constraint weights
3. Full phase: Train with full physics constraints

This stabilizes training by allowing the network to first learn
the basic prediction task before adding complex physics constraints.

Schedules:
- Linear: Constant rate increase
- Exponential: Slow start, fast finish
- Step: Discrete jumps
- Cosine: Smooth cosine annealing
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ScheduleType(Enum):
    """Available schedule types"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    COSINE = "cosine"
    SIGMOID = "sigmoid"


@dataclass
class CurriculumState:
    """Current state of curriculum"""
    epoch: int
    phase: str  # 'warmup', 'ramp', 'full'
    progress: float  # 0.0 to 1.0
    physics_weight: float  # Current multiplier for physics losses
    schedule_type: str


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    enabled: bool = True
    warmup_epochs: int = 10
    ramp_epochs: int = 20
    schedule: str = "linear"
    final_physics_weight: float = 1.0
    min_physics_weight: float = 0.0

    # Per-constraint weights (optional fine-grained control)
    gbm_weight_schedule: Optional[List[float]] = None
    ou_weight_schedule: Optional[List[float]] = None
    bs_weight_schedule: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'warmup_epochs': self.warmup_epochs,
            'ramp_epochs': self.ramp_epochs,
            'schedule': self.schedule,
            'final_physics_weight': self.final_physics_weight,
            'min_physics_weight': self.min_physics_weight
        }


class CurriculumScheduler:
    """
    Curriculum learning scheduler for PINN training.

    Gradually introduces physics constraints over training to stabilize
    optimization and improve final performance.

    Usage:
        scheduler = CurriculumScheduler(
            warmup_epochs=10,
            ramp_epochs=20,
            schedule='linear'
        )

        for epoch in range(total_epochs):
            physics_weight = scheduler.get_physics_weight(epoch)
            loss = data_loss + physics_weight * physics_loss
            scheduler.step()
    """

    def __init__(
        self,
        warmup_epochs: int = 10,
        ramp_epochs: int = 20,
        schedule: str = "linear",
        final_physics_weight: float = 1.0,
        min_physics_weight: float = 0.0,
        total_epochs: Optional[int] = None
    ):
        """
        Initialize curriculum scheduler.

        Args:
            warmup_epochs: Epochs with physics weight = 0
            ramp_epochs: Epochs to ramp from 0 to final weight
            schedule: Schedule type ('linear', 'exponential', 'step', 'cosine', 'sigmoid')
            final_physics_weight: Target physics weight
            min_physics_weight: Minimum physics weight (usually 0)
            total_epochs: Total training epochs (for logging)
        """
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.schedule_type = ScheduleType(schedule.lower())
        self.final_physics_weight = final_physics_weight
        self.min_physics_weight = min_physics_weight
        self.total_epochs = total_epochs or (warmup_epochs + ramp_epochs + 50)

        self.current_epoch = 0
        self.history: List[CurriculumState] = []

        logger.info(f"CurriculumScheduler initialized:")
        logger.info(f"  Warmup: {warmup_epochs} epochs (physics_weight=0)")
        logger.info(f"  Ramp: {ramp_epochs} epochs ({schedule})")
        logger.info(f"  Final weight: {final_physics_weight}")

    @classmethod
    def from_config(cls, config: CurriculumConfig) -> 'CurriculumScheduler':
        """Create scheduler from config"""
        return cls(
            warmup_epochs=config.warmup_epochs,
            ramp_epochs=config.ramp_epochs,
            schedule=config.schedule,
            final_physics_weight=config.final_physics_weight,
            min_physics_weight=config.min_physics_weight
        )

    def get_phase(self, epoch: int) -> str:
        """Get current training phase"""
        if epoch < self.warmup_epochs:
            return 'warmup'
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            return 'ramp'
        else:
            return 'full'

    def get_ramp_progress(self, epoch: int) -> float:
        """Get progress through ramp phase (0.0 to 1.0)"""
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch >= self.warmup_epochs + self.ramp_epochs:
            return 1.0
        else:
            ramp_epoch = epoch - self.warmup_epochs
            return ramp_epoch / self.ramp_epochs

    def _linear_schedule(self, progress: float) -> float:
        """Linear schedule: constant rate"""
        return progress

    def _exponential_schedule(self, progress: float, base: float = 2.0) -> float:
        """Exponential schedule: slow start, fast finish"""
        return (base ** progress - 1) / (base - 1)

    def _step_schedule(self, progress: float, n_steps: int = 4) -> float:
        """Step schedule: discrete jumps"""
        step = int(progress * n_steps)
        return step / n_steps

    def _cosine_schedule(self, progress: float) -> float:
        """Cosine schedule: smooth S-curve"""
        return 0.5 * (1 - np.cos(np.pi * progress))

    def _sigmoid_schedule(self, progress: float, steepness: float = 10.0) -> float:
        """Sigmoid schedule: sharp transition in middle"""
        return 1 / (1 + np.exp(-steepness * (progress - 0.5)))

    def get_schedule_value(self, progress: float) -> float:
        """Get scheduled value for given progress"""
        if self.schedule_type == ScheduleType.LINEAR:
            return self._linear_schedule(progress)
        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            return self._exponential_schedule(progress)
        elif self.schedule_type == ScheduleType.STEP:
            return self._step_schedule(progress)
        elif self.schedule_type == ScheduleType.COSINE:
            return self._cosine_schedule(progress)
        elif self.schedule_type == ScheduleType.SIGMOID:
            return self._sigmoid_schedule(progress)
        else:
            return progress

    def get_physics_weight(self, epoch: Optional[int] = None) -> float:
        """
        Get current physics weight multiplier.

        Args:
            epoch: Epoch number (uses current_epoch if None)

        Returns:
            Physics weight (0.0 to final_physics_weight)
        """
        if epoch is None:
            epoch = self.current_epoch

        phase = self.get_phase(epoch)

        if phase == 'warmup':
            return self.min_physics_weight
        elif phase == 'full':
            return self.final_physics_weight
        else:
            # Ramp phase
            progress = self.get_ramp_progress(epoch)
            schedule_value = self.get_schedule_value(progress)
            weight = (self.min_physics_weight +
                     (self.final_physics_weight - self.min_physics_weight) * schedule_value)
            return weight

    def get_constraint_weights(
        self,
        epoch: Optional[int] = None,
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get per-constraint weights.

        Args:
            epoch: Epoch number
            base_weights: Base lambda values for each constraint

        Returns:
            Dict of scaled constraint weights
        """
        physics_multiplier = self.get_physics_weight(epoch)

        if base_weights is None:
            base_weights = {
                'lambda_gbm': 0.1,
                'lambda_ou': 0.1,
                'lambda_bs': 0.1,
                'lambda_langevin': 0.1
            }

        return {name: value * physics_multiplier for name, value in base_weights.items()}

    def step(self):
        """Advance to next epoch"""
        state = self.get_state()
        self.history.append(state)
        self.current_epoch += 1

        # Log phase transitions
        if self.current_epoch == self.warmup_epochs:
            logger.info(f"Curriculum: Starting ramp phase (epoch {self.current_epoch})")
        elif self.current_epoch == self.warmup_epochs + self.ramp_epochs:
            logger.info(f"Curriculum: Entering full physics phase (epoch {self.current_epoch})")

    def get_state(self) -> CurriculumState:
        """Get current curriculum state"""
        return CurriculumState(
            epoch=self.current_epoch,
            phase=self.get_phase(self.current_epoch),
            progress=self.get_ramp_progress(self.current_epoch),
            physics_weight=self.get_physics_weight(self.current_epoch),
            schedule_type=self.schedule_type.value
        )

    def get_schedule_preview(self, total_epochs: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Preview the full schedule.

        Args:
            total_epochs: Number of epochs to preview

        Returns:
            List of (epoch, physics_weight) tuples
        """
        total = total_epochs or self.total_epochs
        return [(e, self.get_physics_weight(e)) for e in range(total)]

    def reset(self):
        """Reset scheduler to initial state"""
        self.current_epoch = 0
        self.history.clear()


class ConstraintScheduler:
    """
    Fine-grained scheduler for individual physics constraints.

    Allows different constraints to be introduced at different rates.
    """

    def __init__(
        self,
        constraint_schedules: Dict[str, CurriculumScheduler]
    ):
        """
        Initialize with per-constraint schedulers.

        Args:
            constraint_schedules: Dict mapping constraint name to its scheduler
        """
        self.schedules = constraint_schedules
        self.current_epoch = 0

    @classmethod
    def create_staggered(
        cls,
        constraints: List[str],
        base_warmup: int = 5,
        stagger: int = 5,
        ramp_epochs: int = 15,
        schedule: str = 'linear'
    ) -> 'ConstraintScheduler':
        """
        Create staggered scheduler where constraints are introduced one at a time.

        Args:
            constraints: List of constraint names
            base_warmup: Warmup for first constraint
            stagger: Additional warmup per subsequent constraint
            ramp_epochs: Ramp duration for each constraint
            schedule: Schedule type

        Returns:
            ConstraintScheduler with staggered schedules
        """
        schedules = {}
        for i, name in enumerate(constraints):
            warmup = base_warmup + i * stagger
            schedules[name] = CurriculumScheduler(
                warmup_epochs=warmup,
                ramp_epochs=ramp_epochs,
                schedule=schedule
            )
        return cls(schedules)

    def get_weights(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """Get all constraint weights"""
        if epoch is None:
            epoch = self.current_epoch
        return {name: sched.get_physics_weight(epoch)
                for name, sched in self.schedules.items()}

    def step(self):
        """Advance all schedulers"""
        for sched in self.schedules.values():
            sched.step()
        self.current_epoch += 1

    def reset(self):
        """Reset all schedulers"""
        for sched in self.schedules.values():
            sched.reset()
        self.current_epoch = 0


class DynamicCurriculumScheduler(CurriculumScheduler):
    """
    Dynamic curriculum scheduler that adjusts based on training metrics.

    Can slow down or speed up curriculum based on:
    - Loss convergence
    - Gradient stability
    - Residual magnitudes
    """

    def __init__(
        self,
        warmup_epochs: int = 10,
        ramp_epochs: int = 20,
        schedule: str = "linear",
        final_physics_weight: float = 1.0,
        loss_threshold: float = 0.1,
        stability_threshold: float = 0.5
    ):
        super().__init__(
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
            schedule=schedule,
            final_physics_weight=final_physics_weight
        )
        self.loss_threshold = loss_threshold
        self.stability_threshold = stability_threshold
        self.adjustment_factor = 1.0  # Speed multiplier

    def update_from_metrics(
        self,
        loss: float,
        stability_score: float,
        prev_loss: Optional[float] = None
    ):
        """
        Update curriculum speed based on training metrics.

        Args:
            loss: Current loss value
            stability_score: Training stability (0-1)
            prev_loss: Previous loss for comparison
        """
        # If training is unstable, slow down curriculum
        if stability_score < self.stability_threshold:
            self.adjustment_factor = max(0.5, self.adjustment_factor * 0.9)
            logger.warning(f"Curriculum slowed down (stability={stability_score:.2f})")
        # If loss is improving well, can speed up
        elif prev_loss is not None and loss < prev_loss * 0.95:
            self.adjustment_factor = min(2.0, self.adjustment_factor * 1.1)

    def get_ramp_progress(self, epoch: int) -> float:
        """Get adjusted ramp progress"""
        base_progress = super().get_ramp_progress(epoch)
        # Apply adjustment factor
        adjusted = base_progress * self.adjustment_factor
        return min(1.0, adjusted)


# Convenience functions

def create_curriculum_scheduler(
    config: Dict,
    total_epochs: int
) -> CurriculumScheduler:
    """
    Create curriculum scheduler from config dict.

    Args:
        config: Configuration dict with curriculum settings
        total_epochs: Total training epochs

    Returns:
        CurriculumScheduler instance
    """
    return CurriculumScheduler(
        warmup_epochs=config.get('warmup_epochs', 10),
        ramp_epochs=config.get('ramp_epochs', 20),
        schedule=config.get('schedule', 'linear'),
        final_physics_weight=config.get('final_physics_weight', 1.0),
        min_physics_weight=config.get('min_physics_weight', 0.0),
        total_epochs=total_epochs
    )


def visualize_curriculum(
    scheduler: CurriculumScheduler,
    total_epochs: int,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize curriculum schedule (requires matplotlib).

    Args:
        scheduler: CurriculumScheduler to visualize
        total_epochs: Number of epochs
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        epochs = list(range(total_epochs))
        weights = [scheduler.get_physics_weight(e) for e in epochs]
        phases = [scheduler.get_phase(e) for e in epochs]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Color by phase
        colors = {'warmup': 'lightblue', 'ramp': 'orange', 'full': 'green'}
        for i, (e, w) in enumerate(zip(epochs[:-1], weights[:-1])):
            ax.fill_between([e, e+1], [0, 0], [w, weights[i+1]],
                          color=colors[phases[i]], alpha=0.3)

        ax.plot(epochs, weights, 'b-', linewidth=2, label='Physics Weight')

        ax.axvline(scheduler.warmup_epochs, color='gray', linestyle='--',
                  label='End Warmup')
        ax.axvline(scheduler.warmup_epochs + scheduler.ramp_epochs,
                  color='gray', linestyle=':', label='End Ramp')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics Weight')
        ax.set_title(f'Curriculum Schedule ({scheduler.schedule_type.value})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    except ImportError:
        logger.warning("matplotlib not available for visualization")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Curriculum Scheduler Demo")
    print("=" * 60)

    # Create scheduler
    scheduler = CurriculumScheduler(
        warmup_epochs=10,
        ramp_epochs=20,
        schedule='cosine',
        final_physics_weight=1.0
    )

    # Preview schedule
    print("\nSchedule Preview:")
    print("-" * 40)
    for epoch, weight in scheduler.get_schedule_preview(50):
        phase = scheduler.get_phase(epoch)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: phase={phase:8s}, weight={weight:.4f}")

    # Simulate training
    print("\n" + "-" * 40)
    print("Simulating training...")
    for epoch in range(40):
        weight = scheduler.get_physics_weight()
        scheduler.step()

    print(f"Final state: {scheduler.get_state()}")
    print("\n" + "=" * 60)
