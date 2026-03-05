"""
Dual-Phase PINN Trainer

Training orchestration for Burgers' equation PINNs:
- Standard PINN training with Adam + L-BFGS
- Two-phase training for DualPhasePINN
- Learning rate scheduling
- Training history tracking
- Gradient clipping for stability

Training Protocol:
1. Phase 1: Adam (lr=1e-3, 50k iter) + L-BFGS on t ∈ [0, 0.4]
2. Phase 2: Freeze phase1, train phase2 on t ∈ [0.4, 1] with intermediate constraint
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.logger import get_logger
from ..utils.reproducibility import set_seed

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PINN training."""

    # Adam optimizer
    adam_lr: float = 1e-3
    adam_iterations: int = 50000
    adam_betas: Tuple[float, float] = (0.9, 0.999)

    # L-BFGS optimizer
    lbfgs_lr: float = 1.0
    lbfgs_iterations: int = 10000
    lbfgs_max_iter: int = 50  # L-BFGS iterations per step
    lbfgs_history_size: int = 50

    # Gradient clipping
    gradient_clip_norm: float = 1.0

    # Learning rate scheduler
    scheduler_patience: int = 1000
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000

    # Device
    device: str = "cuda"

    # Random seed
    seed: int = 42


@dataclass
class TrainingHistory:
    """Container for training history."""

    losses: List[float] = field(default_factory=list)
    pde_losses: List[float] = field(default_factory=list)
    ic_losses: List[float] = field(default_factory=list)
    bc_losses: List[float] = field(default_factory=list)
    intermediate_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    l2_errors: List[float] = field(default_factory=list)

    # Phase tracking for dual-phase
    phase: int = 1
    phase_start_iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "losses": self.losses,
            "pde_losses": self.pde_losses,
            "ic_losses": self.ic_losses,
            "bc_losses": self.bc_losses,
            "intermediate_losses": self.intermediate_losses,
            "learning_rates": self.learning_rates,
            "iterations": self.iterations,
            "times": self.times,
            "l2_errors": self.l2_errors,
            "phase": self.phase,
        }

    def save(self, path: Union[str, Path]):
        """Save history to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingHistory":
        """Load history from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        history = cls()
        for key, value in data.items():
            if hasattr(history, key):
                setattr(history, key, value)
        return history


class DPPINNTrainer:
    """
    Trainer for Burgers' equation PINNs.

    Supports:
    - Standard PINN training (single network)
    - Dual-phase training (two networks with intermediate constraint)
    - Adam + L-BFGS optimization
    - Learning rate scheduling
    - Training history tracking
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: BurgersPINN or DualPhasePINN model
            config: Training configuration
            device: PyTorch device
        """
        self.config = config or TrainingConfig()
        self.device = device or torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.history = TrainingHistory()

        # Will be set during training
        self.optimizer = None
        self.scheduler = None

        logger.info(f"DPPINNTrainer initialized on {self.device}")

    def _create_adam_optimizer(
        self,
        params,
        lr: Optional[float] = None,
    ) -> optim.Adam:
        """Create Adam optimizer."""
        lr = lr or self.config.adam_lr
        return optim.Adam(
            params,
            lr=lr,
            betas=self.config.adam_betas,
        )

    def _create_lbfgs_optimizer(
        self,
        params,
        lr: Optional[float] = None,
    ) -> optim.LBFGS:
        """Create L-BFGS optimizer."""
        lr = lr or self.config.lbfgs_lr
        return optim.LBFGS(
            params,
            lr=lr,
            max_iter=self.config.lbfgs_max_iter,
            history_size=self.config.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
    ) -> optim.lr_scheduler.ReduceLROnPlateau:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_lr,
        )

    def _move_data_to_device(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Move training data to device."""
        return {k: v.to(self.device) for k, v in data.items()}

    def train_standard_pinn(
        self,
        data: Dict[str, torch.Tensor],
        n_adam: Optional[int] = None,
        n_lbfgs: Optional[int] = None,
        eval_fn: Optional[callable] = None,
    ) -> TrainingHistory:
        """
        Train a standard BurgersPINN.

        Training protocol:
        1. Adam optimization for n_adam iterations
        2. L-BFGS optimization for n_lbfgs iterations

        Args:
            data: Training data dictionary with:
                - x_collocation, t_collocation
                - x_ic
                - t_bc
            n_adam: Adam iterations (default from config)
            n_lbfgs: L-BFGS iterations (default from config)
            eval_fn: Optional function to compute L2 error during training

        Returns:
            TrainingHistory object
        """
        n_adam = n_adam or self.config.adam_iterations
        n_lbfgs = n_lbfgs or self.config.lbfgs_iterations

        data = self._move_data_to_device(data)
        set_seed(self.config.seed)

        logger.info(f"Training standard PINN: {n_adam} Adam + {n_lbfgs} L-BFGS")

        # Adam phase
        self.history = TrainingHistory()
        self._train_adam(
            params=self.model.parameters(),
            data=data,
            n_iterations=n_adam,
            loss_fn=self._standard_loss_fn,
            eval_fn=eval_fn,
        )

        # L-BFGS phase
        self._train_lbfgs(
            params=self.model.parameters(),
            data=data,
            n_iterations=n_lbfgs,
            loss_fn=self._standard_loss_fn,
            eval_fn=eval_fn,
        )

        return self.history

    def train_phase1(
        self,
        data: Dict[str, torch.Tensor],
        n_adam: Optional[int] = None,
        n_lbfgs: Optional[int] = None,
        eval_fn: Optional[callable] = None,
    ) -> TrainingHistory:
        """
        Train phase 1 of DualPhasePINN.

        Phase 1 trains on t ∈ [0, t_switch] with IC constraint.

        Args:
            data: Training data with phase 1 points
            n_adam: Adam iterations
            n_lbfgs: L-BFGS iterations
            eval_fn: Optional evaluation function

        Returns:
            TrainingHistory for phase 1
        """
        if not hasattr(self.model, "phase1_net"):
            raise ValueError("Model must be DualPhasePINN for phase training")

        n_adam = n_adam or self.config.adam_iterations
        n_lbfgs = n_lbfgs or self.config.lbfgs_iterations

        data = self._move_data_to_device(data)
        set_seed(self.config.seed)

        logger.info(f"Training Phase 1: {n_adam} Adam + {n_lbfgs} L-BFGS")

        self.history = TrainingHistory()
        self.history.phase = 1

        # Use phase 1 collocation points
        phase1_data = {
            "x_collocation": data["x_collocation_p1"],
            "t_collocation": data["t_collocation_p1"],
            "x_ic": data["x_ic"],
            "t_bc": data["t_bc_p1"],
        }

        # Train phase 1 network
        params = self.model.phase1_net.parameters()

        self._train_adam(
            params=params,
            data=phase1_data,
            n_iterations=n_adam,
            loss_fn=self._phase1_loss_fn,
            eval_fn=eval_fn,
        )

        self._train_lbfgs(
            params=params,
            data=phase1_data,
            n_iterations=n_lbfgs,
            loss_fn=self._phase1_loss_fn,
            eval_fn=eval_fn,
        )

        return self.history

    def train_phase2(
        self,
        data: Dict[str, torch.Tensor],
        n_adam: Optional[int] = None,
        n_lbfgs: Optional[int] = None,
        eval_fn: Optional[callable] = None,
    ) -> TrainingHistory:
        """
        Train phase 2 of DualPhasePINN.

        Phase 2 trains on t ∈ [t_switch, 1] with intermediate constraint.
        Phase 1 network is frozen.

        Args:
            data: Training data with phase 2 points
            n_adam: Adam iterations
            n_lbfgs: L-BFGS iterations
            eval_fn: Optional evaluation function

        Returns:
            TrainingHistory for phase 2
        """
        if not hasattr(self.model, "phase2_net"):
            raise ValueError("Model must be DualPhasePINN for phase training")

        n_adam = n_adam or self.config.adam_iterations
        n_lbfgs = n_lbfgs or self.config.lbfgs_iterations

        data = self._move_data_to_device(data)
        set_seed(self.config.seed + 1000)

        logger.info(f"Training Phase 2: {n_adam} Adam + {n_lbfgs} L-BFGS")

        # Freeze phase 1
        self.model.freeze_phase1()

        self.history = TrainingHistory()
        self.history.phase = 2
        self.history.phase_start_iteration = len(self.history.iterations)

        # Use phase 2 collocation points
        phase2_data = {
            "x_collocation": data["x_collocation_p2"],
            "t_collocation": data["t_collocation_p2"],
            "x_intermediate": data["x_intermediate"],
            "t_bc": data["t_bc_p2"],
        }

        # Train phase 2 network
        params = self.model.phase2_net.parameters()

        self._train_adam(
            params=params,
            data=phase2_data,
            n_iterations=n_adam,
            loss_fn=self._phase2_loss_fn,
            eval_fn=eval_fn,
        )

        self._train_lbfgs(
            params=params,
            data=phase2_data,
            n_iterations=n_lbfgs,
            loss_fn=self._phase2_loss_fn,
            eval_fn=eval_fn,
        )

        return self.history

    def _train_adam(
        self,
        params,
        data: Dict[str, torch.Tensor],
        n_iterations: int,
        loss_fn: callable,
        eval_fn: Optional[callable] = None,
    ):
        """
        Adam optimization phase.

        Args:
            params: Model parameters to optimize
            data: Training data
            n_iterations: Number of iterations
            loss_fn: Loss function
            eval_fn: Optional evaluation function
        """
        self.optimizer = self._create_adam_optimizer(list(params))
        self.scheduler = self._create_scheduler(self.optimizer)

        start_time = time.time()
        start_iter = len(self.history.iterations)

        for iteration in range(n_iterations):
            self.optimizer.zero_grad()

            loss, loss_dict = loss_fn(data)
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()
            self.scheduler.step(loss)

            # Record history
            current_iter = start_iter + iteration
            self._record_history(current_iter, loss_dict, time.time() - start_time)

            # Logging
            if iteration % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Adam [{iteration}/{n_iterations}] "
                    f"Loss: {loss.item():.6f} LR: {lr:.2e}"
                )

            # Evaluation
            if eval_fn is not None and iteration % self.config.eval_interval == 0:
                l2_error = eval_fn(self.model)
                self.history.l2_errors.append(l2_error)
                logger.info(f"  L2 Error: {l2_error:.6f}")

    def _train_lbfgs(
        self,
        params,
        data: Dict[str, torch.Tensor],
        n_iterations: int,
        loss_fn: callable,
        eval_fn: Optional[callable] = None,
    ):
        """
        L-BFGS optimization phase.

        Args:
            params: Model parameters to optimize
            data: Training data
            n_iterations: Number of iterations
            loss_fn: Loss function
            eval_fn: Optional evaluation function
        """
        self.optimizer = self._create_lbfgs_optimizer(list(params))

        start_time = time.time()
        start_iter = len(self.history.iterations)

        def closure():
            self.optimizer.zero_grad()
            loss, _ = loss_fn(data)
            loss.backward()
            return loss

        for iteration in range(n_iterations):
            loss = self.optimizer.step(closure)

            # Record history (L-BFGS doesn't have per-component losses readily)
            loss_dict = {"total_loss": loss.item()}
            current_iter = start_iter + iteration
            self._record_history(current_iter, loss_dict, time.time() - start_time)

            # Logging
            if iteration % (self.config.log_interval // 10) == 0:
                logger.info(f"L-BFGS [{iteration}/{n_iterations}] Loss: {loss.item():.6f}")

            # Evaluation
            if eval_fn is not None and iteration % (self.config.eval_interval // 10) == 0:
                l2_error = eval_fn(self.model)
                self.history.l2_errors.append(l2_error)
                logger.info(f"  L2 Error: {l2_error:.6f}")

    def _standard_loss_fn(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for standard PINN."""
        return self.model.compute_loss(
            x_collocation=data["x_collocation"],
            t_collocation=data["t_collocation"],
            x_ic=data["x_ic"],
            t_bc=data["t_bc"],
        )

    def _phase1_loss_fn(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for phase 1."""
        return self.model.compute_phase1_loss(
            x_collocation=data["x_collocation"],
            t_collocation=data["t_collocation"],
            x_ic=data["x_ic"],
            t_bc=data["t_bc"],
        )

    def _phase2_loss_fn(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for phase 2."""
        return self.model.compute_phase2_loss(
            x_collocation=data["x_collocation"],
            t_collocation=data["t_collocation"],
            x_intermediate=data["x_intermediate"],
            t_bc=data["t_bc"],
        )

    def _record_history(
        self,
        iteration: int,
        loss_dict: Dict[str, float],
        elapsed_time: float,
    ):
        """Record training metrics to history."""
        self.history.iterations.append(iteration)
        self.history.times.append(elapsed_time)
        self.history.losses.append(loss_dict.get("total_loss", 0.0))
        self.history.pde_losses.append(loss_dict.get("pde_loss", 0.0))
        self.history.ic_losses.append(loss_dict.get("ic_loss", 0.0))
        self.history.bc_losses.append(loss_dict.get("bc_loss", 0.0))
        self.history.intermediate_losses.append(
            loss_dict.get("intermediate_loss", 0.0)
        )

        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(lr)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        extra_info: Optional[Dict] = None,
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            extra_info: Additional info to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "history": self.history.to_dict(),
        }

        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if extra_info is not None:
            checkpoint.update(extra_info)

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(
        self,
        path: Union[str, Path],
    ):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "history" in checkpoint:
            for key, value in checkpoint["history"].items():
                if hasattr(self.history, key):
                    setattr(self.history, key, value)

        logger.info(f"Checkpoint loaded from {path}")


def train_standard_pinn(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    eval_fn: Optional[callable] = None,
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Convenience function to train a standard PINN.

    Args:
        model: BurgersPINN model
        data: Training data
        config: Training configuration
        device: PyTorch device
        eval_fn: Optional evaluation function

    Returns:
        Tuple of (trained_model, history)
    """
    trainer = DPPINNTrainer(model, config, device)
    history = trainer.train_standard_pinn(data, eval_fn=eval_fn)
    return model, history


def train_dual_phase_pinn(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    eval_fn: Optional[callable] = None,
) -> Tuple[nn.Module, TrainingHistory, TrainingHistory]:
    """
    Convenience function to train a DualPhasePINN.

    Args:
        model: DualPhasePINN model
        data: Training data
        config: Training configuration
        device: PyTorch device
        eval_fn: Optional evaluation function

    Returns:
        Tuple of (trained_model, phase1_history, phase2_history)
    """
    trainer = DPPINNTrainer(model, config, device)

    # Phase 1
    history1 = trainer.train_phase1(data, eval_fn=eval_fn)

    # Phase 2
    history2 = trainer.train_phase2(data, eval_fn=eval_fn)

    return model, history1, history2
