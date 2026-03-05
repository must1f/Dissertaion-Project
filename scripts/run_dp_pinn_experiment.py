#!/usr/bin/env python3
"""
Dual-Phase PINN Experiment for Burgers' Equation

This script trains and evaluates both standard PINN and Dual-Phase PINN
on the viscous Burgers' equation benchmark.

Usage:
    python scripts/run_dp_pinn_experiment.py
    python scripts/run_dp_pinn_experiment.py --config configs/dp_pinn_config.yaml
    python scripts/run_dp_pinn_experiment.py --model dual_phase --seed 42

The script:
1. Generates training data using Latin Hypercube Sampling
2. Trains a standard PINN (baseline)
3. Trains a Dual-Phase PINN (two-phase approach)
4. Evaluates both models against the exact solution
5. Generates comprehensive visualizations
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dp_pinn import BurgersPINN, DualPhasePINN, create_burgers_pinn
from src.training.dp_pinn_trainer import (
    DPPINNTrainer,
    TrainingConfig,
    train_standard_pinn,
    train_dual_phase_pinn,
)
from src.evaluation.pde_evaluator import (
    PDEEvaluator,
    create_burgers_evaluator,
    compare_models,
    burgers_exact_solution_hopf_cole,
)
from src.reporting.pde_visualization import (
    BurgersVisualization,
    create_comparison_visualization,
)
from src.utils.sampling import generate_burgers_training_data
from src.utils.reproducibility import set_seed, get_environment_info, init_experiment
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Dual-Phase PINN experiment for Burgers' equation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dp_pinn_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["standard", "dual_phase", "both"],
        default="both",
        help="Which model(s) to train",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Device to use (overrides config)",
    )

    parser.add_argument(
        "--adam-iter",
        type=int,
        default=None,
        help="Number of Adam iterations (overrides config)",
    )

    parser.add_argument(
        "--lbfgs-iter",
        type=int,
        default=None,
        help="Number of L-BFGS iterations (overrides config)",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and load from checkpoints",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with reduced iterations for testing",
    )

    return parser.parse_args()


def create_training_config(config: dict, args) -> TrainingConfig:
    """Create TrainingConfig from config dict and args."""
    training = config.get("training", {})
    adam = training.get("adam", {})
    lbfgs = training.get("lbfgs", {})
    scheduler = training.get("scheduler", {})

    # Override with command line args
    adam_iter = args.adam_iter or adam.get("iterations", 50000)
    lbfgs_iter = args.lbfgs_iter or lbfgs.get("iterations", 10000)

    if args.quick:
        adam_iter = min(adam_iter, 1000)
        lbfgs_iter = min(lbfgs_iter, 100)

    device = args.device or training.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    return TrainingConfig(
        adam_lr=adam.get("learning_rate", 1e-3),
        adam_iterations=adam_iter,
        adam_betas=(adam.get("beta1", 0.9), adam.get("beta2", 0.999)),
        lbfgs_lr=lbfgs.get("learning_rate", 1.0),
        lbfgs_iterations=lbfgs_iter,
        lbfgs_max_iter=lbfgs.get("max_iter_per_step", 50),
        lbfgs_history_size=lbfgs.get("history_size", 50),
        gradient_clip_norm=training.get("gradient_clip_norm", 1.0),
        scheduler_patience=scheduler.get("patience", 1000),
        scheduler_factor=scheduler.get("factor", 0.5),
        min_lr=scheduler.get("min_lr", 1e-6),
        log_interval=training.get("log_interval", 100),
        eval_interval=training.get("eval_interval", 1000),
        device=device,
        seed=args.seed,
    )


def main():
    """Main experiment entry point."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    # Initialize experiment
    seed = args.seed
    env_info = init_experiment(seed=seed)

    # Setup output directory
    output_dir = Path(
        args.output_dir
        or config.get("output", {}).get("base_dir", "results/dp_pinn")
    )
    output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create training config
    training_config = create_training_config(config, args)
    device = torch.device(training_config.device)

    # Get problem parameters
    problem = config.get("problem", {})
    viscosity = problem.get("viscosity", 0.01 / 3.14159265359)
    x_range = (problem.get("x_min", -1.0), problem.get("x_max", 1.0))
    t_range = (problem.get("t_min", 0.0), problem.get("t_max", 1.0))

    # Get model parameters
    model_config = config.get("model", {})
    num_layers = model_config.get("num_layers", 8)
    hidden_dim = model_config.get("hidden_dim", 50)
    activation = model_config.get("activation", "tanh")
    lambda_pde = model_config.get("lambda_pde", 1.0)
    lambda_ic = model_config.get("lambda_ic", 100.0)
    lambda_bc = model_config.get("lambda_bc", 100.0)

    # Get dual-phase parameters
    dp_config = config.get("dual_phase", {})
    t_switch = dp_config.get("t_switch", 0.4)
    lambda_intermediate = dp_config.get("lambda_intermediate", 100.0)

    # Get data parameters
    data_config = config.get("data", {})
    n_collocation = data_config.get("n_collocation", 20000)
    n_boundary = data_config.get("n_boundary", 2000)
    n_initial = data_config.get("n_initial", 2000)
    n_intermediate = data_config.get("n_intermediate", 1000)

    if args.quick:
        n_collocation = min(n_collocation, 2000)
        n_boundary = min(n_boundary, 200)
        n_initial = min(n_initial, 200)
        n_intermediate = min(n_intermediate, 100)

    # Generate training data
    logger.info("Generating training data...")
    data = generate_burgers_training_data(
        n_collocation=n_collocation,
        n_boundary=n_boundary,
        n_initial=n_initial,
        n_intermediate=n_intermediate,
        t_switch=t_switch,
        x_range=x_range,
        t_range=t_range,
        seed=seed,
        device=device,
    )

    # Create evaluator
    eval_config = config.get("evaluation", {})
    n_x_eval = eval_config.get("n_x", 256)
    n_t_eval = eval_config.get("n_t", 100)

    evaluator = create_burgers_evaluator(
        viscosity=viscosity,
        n_x=n_x_eval,
        n_t=n_t_eval,
        device=device,
    )

    # Create evaluation function for training
    def eval_fn(model):
        return evaluator.relative_l2_error(model)

    # Results storage
    models = {}
    histories = {}
    metrics = {}

    # =========================================================================
    # Train Standard PINN
    # =========================================================================
    if args.model in ["standard", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING STANDARD PINN")
        logger.info("=" * 60)

        pinn = BurgersPINN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            viscosity=viscosity,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc,
            lambda_pde=lambda_pde,
        )

        if not args.skip_training:
            trainer = DPPINNTrainer(pinn, training_config, device)
            pinn_history = trainer.train_standard_pinn(
                data=data,
                eval_fn=eval_fn,
            )

            # Save checkpoint
            checkpoint_path = output_dir / "checkpoints" / "standard_pinn.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path)

            # Save history
            history_path = output_dir / "checkpoints" / "standard_pinn_history.json"
            pinn_history.save(history_path)

            histories["Standard PINN"] = pinn_history.to_dict()
        else:
            checkpoint_path = output_dir / "checkpoints" / "standard_pinn.pt"
            if checkpoint_path.exists():
                trainer = DPPINNTrainer(pinn, training_config, device)
                trainer.load_checkpoint(checkpoint_path)
                histories["Standard PINN"] = trainer.history.to_dict()

        models["Standard PINN"] = pinn

        # Evaluate
        pinn_metrics = evaluator.evaluate_all(pinn)
        metrics["Standard PINN"] = pinn_metrics

        logger.info(f"Standard PINN - L2 Error: {pinn_metrics.relative_l2_error:.6f}")

    # =========================================================================
    # Train Dual-Phase PINN
    # =========================================================================
    if args.model in ["dual_phase", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING DUAL-PHASE PINN")
        logger.info("=" * 60)

        dp_pinn = DualPhasePINN(
            t_switch=t_switch,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            viscosity=viscosity,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc,
            lambda_pde=lambda_pde,
            lambda_intermediate=lambda_intermediate,
        )

        if not args.skip_training:
            trainer = DPPINNTrainer(dp_pinn, training_config, device)

            # Phase 1
            logger.info("\n--- Phase 1 Training ---")
            phase1_history = trainer.train_phase1(data=data, eval_fn=eval_fn)

            # Phase 2
            logger.info("\n--- Phase 2 Training ---")
            phase2_history = trainer.train_phase2(data=data, eval_fn=eval_fn)

            # Save checkpoint
            checkpoint_path = output_dir / "checkpoints" / "dual_phase_pinn.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path)

            # Combine histories
            combined_history = {
                "phase1": phase1_history.to_dict(),
                "phase2": phase2_history.to_dict(),
                "losses": phase1_history.losses + phase2_history.losses,
                "pde_losses": phase1_history.pde_losses + phase2_history.pde_losses,
                "ic_losses": phase1_history.ic_losses + phase2_history.ic_losses,
                "bc_losses": phase1_history.bc_losses + phase2_history.bc_losses,
                "intermediate_losses": (
                    phase1_history.intermediate_losses + phase2_history.intermediate_losses
                ),
            }
            histories["DP-PINN"] = combined_history

            # Save history
            with open(output_dir / "checkpoints" / "dp_pinn_history.json", "w") as f:
                json.dump(combined_history, f, indent=2)
        else:
            checkpoint_path = output_dir / "checkpoints" / "dual_phase_pinn.pt"
            if checkpoint_path.exists():
                trainer = DPPINNTrainer(dp_pinn, training_config, device)
                trainer.load_checkpoint(checkpoint_path)
                histories["DP-PINN"] = trainer.history.to_dict()

        models["DP-PINN"] = dp_pinn

        # Evaluate
        dp_metrics = evaluator.evaluate_all(dp_pinn)
        metrics["DP-PINN"] = dp_metrics

        logger.info(f"DP-PINN - L2 Error: {dp_metrics.relative_l2_error:.6f}")

    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)

    plots_dir = output_dir / "plots"

    if len(models) > 1:
        # Comparison visualizations
        saved_plots = create_comparison_visualization(
            models=models,
            evaluator=evaluator,
            histories=histories,
            output_dir=plots_dir,
        )
    else:
        # Single model visualization
        viz = BurgersVisualization()
        for name, model in models.items():
            grids = evaluator.get_prediction_grid(model)
            saved_plots = viz.generate_all(
                grids=grids,
                history=histories.get(name, {"losses": []}),
                metrics=metrics[name],
                output_dir=plots_dir / name.replace(" ", "_").lower(),
                model_name=name.replace(" ", "_").lower(),
            )

    logger.info(f"Saved {len(saved_plots)} plots to {plots_dir}")

    # =========================================================================
    # Summary Report
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    summary = {
        "experiment": {
            "date": datetime.now().isoformat(),
            "seed": seed,
            "device": str(device),
            "config_path": str(config_path),
        },
        "problem": {
            "viscosity": viscosity,
            "domain": {"x": x_range, "t": t_range},
        },
        "training": {
            "adam_iterations": training_config.adam_iterations,
            "lbfgs_iterations": training_config.lbfgs_iterations,
        },
        "results": {},
    }

    for name, m in metrics.items():
        summary["results"][name] = m.to_dict()
        logger.info(f"\n{name}:")
        logger.info(f"  Relative L2 Error: {m.relative_l2_error:.6e}")
        logger.info(f"  Absolute L2 Error: {m.absolute_l2_error:.6e}")
        logger.info(f"  Max Error: {m.max_error:.6f}")
        logger.info(f"  Mean Error: {m.mean_error:.6f}")

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Experiment complete!")

    return summary


if __name__ == "__main__":
    main()
