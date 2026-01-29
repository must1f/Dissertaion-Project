"""
Model Registry - Central registry for all neural network models

Tracks all available models, their training status, and metadata
"""

from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a model"""
    model_key: str
    model_name: str
    model_type: str  # 'baseline', 'pinn', 'advanced'
    architecture: str  # 'LSTM', 'GRU', 'Transformer', 'PINN', etc.
    description: str
    physics_constraints: Optional[Dict[str, float]] = None
    trained: bool = False
    model_path: Optional[Path] = None
    results_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    training_date: Optional[str] = None
    epochs_trained: Optional[int] = None


class ModelRegistry:
    """
    Central registry for all neural network models in the system
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.models_dir = project_root / 'models'
        self.results_dir = project_root / 'results'

        # Define all available models
        self.models = self._define_all_models()

    def _define_all_models(self) -> Dict[str, ModelInfo]:
        """Define all available models in the system"""

        models = {}

        # ========== BASELINE MODELS ==========
        models['lstm'] = ModelInfo(
            model_key='lstm',
            model_name='LSTM',
            model_type='baseline',
            architecture='LSTM',
            description='Long Short-Term Memory network for sequence modeling'
        )

        models['gru'] = ModelInfo(
            model_key='gru',
            model_name='GRU',
            model_type='baseline',
            architecture='GRU',
            description='Gated Recurrent Unit for efficient sequence processing'
        )

        models['bilstm'] = ModelInfo(
            model_key='bilstm',
            model_name='Bidirectional LSTM',
            model_type='baseline',
            architecture='BiLSTM',
            description='Bidirectional LSTM for forward and backward context'
        )

        models['attention_lstm'] = ModelInfo(
            model_key='attention_lstm',
            model_name='Attention LSTM',
            model_type='baseline',
            architecture='AttentionLSTM',
            description='LSTM with attention mechanism for long-term dependencies'
        )

        models['transformer'] = ModelInfo(
            model_key='transformer',
            model_name='Transformer',
            model_type='baseline',
            architecture='Transformer',
            description='Transformer model with multi-head self-attention'
        )

        # ========== BASIC PINN VARIANTS ==========
        models['baseline_pinn'] = ModelInfo(
            model_key='baseline',
            model_name='Baseline (Data-only)',
            model_type='pinn',
            architecture='PINN',
            description='Pure data-driven learning without physics constraints',
            physics_constraints={'lambda_gbm': 0.0, 'lambda_bs': 0.0, 'lambda_ou': 0.0, 'lambda_langevin': 0.0}
        )

        models['gbm'] = ModelInfo(
            model_key='gbm',
            model_name='Pure GBM (Trend)',
            model_type='pinn',
            architecture='PINN',
            description='Geometric Brownian Motion - trend-following dynamics',
            physics_constraints={'lambda_gbm': 0.1, 'lambda_bs': 0.0, 'lambda_ou': 0.0, 'lambda_langevin': 0.0}
        )

        models['ou'] = ModelInfo(
            model_key='ou',
            model_name='Pure OU (Mean-Reversion)',
            model_type='pinn',
            architecture='PINN',
            description='Ornstein-Uhlenbeck process - mean-reverting dynamics',
            physics_constraints={'lambda_gbm': 0.0, 'lambda_bs': 0.0, 'lambda_ou': 0.1, 'lambda_langevin': 0.0}
        )

        models['black_scholes'] = ModelInfo(
            model_key='black_scholes',
            model_name='Pure Black-Scholes',
            model_type='pinn',
            architecture='PINN',
            description='No-arbitrage PDE constraint',
            physics_constraints={'lambda_gbm': 0.0, 'lambda_bs': 0.1, 'lambda_ou': 0.0, 'lambda_langevin': 0.0}
        )

        models['gbm_ou'] = ModelInfo(
            model_key='gbm_ou',
            model_name='GBM+OU Hybrid',
            model_type='pinn',
            architecture='PINN',
            description='Combined trend and mean-reversion dynamics',
            physics_constraints={'lambda_gbm': 0.05, 'lambda_bs': 0.0, 'lambda_ou': 0.05, 'lambda_langevin': 0.0}
        )

        models['global'] = ModelInfo(
            model_key='global',
            model_name='Global Constraint',
            model_type='pinn',
            architecture='PINN',
            description='All physics equations combined',
            physics_constraints={'lambda_gbm': 0.05, 'lambda_bs': 0.03, 'lambda_ou': 0.05, 'lambda_langevin': 0.02}
        )

        # ========== ADVANCED PINN ARCHITECTURES ==========
        models['stacked'] = ModelInfo(
            model_key='stacked',
            model_name='StackedPINN',
            model_type='advanced',
            architecture='StackedPINN',
            description='Physics encoder + parallel LSTM/GRU + curriculum learning',
            physics_constraints={'lambda_gbm': 0.1, 'lambda_ou': 0.1}
        )

        models['residual'] = ModelInfo(
            model_key='residual',
            model_name='ResidualPINN',
            model_type='advanced',
            architecture='ResidualPINN',
            description='Base model + physics-informed correction',
            physics_constraints={'lambda_gbm': 0.1, 'lambda_ou': 0.1}
        )

        # Check training status for all models
        self._update_training_status(models)

        return models

    def _update_training_status(self, models: Dict[str, ModelInfo]):
        """Check which models have been trained"""

        for model_key, model_info in models.items():
            # Check for model checkpoints
            possible_paths = [
                self.models_dir / f'{model_key}_best.pt',
                self.models_dir / f'{model_key}_best.pth',
                self.models_dir / f'pinn_{model_key}_best.pt',
                self.models_dir / 'stacked_pinn' / f'{model_key}_pinn_best.pt',
            ]

            for path in possible_paths:
                if path.exists():
                    model_info.trained = True
                    model_info.checkpoint_path = path
                    model_info.training_date = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')

                    # Try to read epochs from results
                    results_path = self.results_dir / f'{model_key}_results.json'
                    if not results_path.exists():
                        results_path = self.results_dir / 'pinn_comparison' / f'{model_key}_results.json'
                    if not results_path.exists():
                        results_path = self.models_dir / 'stacked_pinn' / f'{model_key}_pinn_results.json'

                    # Try individual result file first
                    if results_path.exists():
                        try:
                            with open(results_path, 'r') as f:
                                results = json.load(f)
                                if 'history' in results and 'train_loss' in results['history']:
                                    model_info.epochs_trained = len(results['history']['train_loss'])
                                model_info.results_path = results_path
                        except:
                            pass
                    else:
                        # Try loading from detailed_results.json for PINN models
                        detailed_path = self.results_dir / 'pinn_comparison' / 'detailed_results.json'
                        if detailed_path.exists() and model_info.model_type == 'pinn':
                            try:
                                with open(detailed_path, 'r') as f:
                                    detailed_results = json.load(f)
                                    for variant_result in detailed_results:
                                        if variant_result.get('variant_key') == model_key:
                                            if 'history' in variant_result and 'train_loss' in variant_result['history']:
                                                model_info.epochs_trained = len(variant_result['history']['train_loss'])
                                            model_info.results_path = detailed_path
                                            break
                            except:
                                pass
                    break

    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get all models"""
        return self.models

    def get_trained_models(self) -> Dict[str, ModelInfo]:
        """Get only trained models"""
        return {k: v for k, v in self.models.items() if v.trained}

    def get_untrained_models(self) -> Dict[str, ModelInfo]:
        """Get untrained models"""
        return {k: v for k, v in self.models.items() if not v.trained}

    def get_models_by_type(self, model_type: str) -> Dict[str, ModelInfo]:
        """Get models by type (baseline, pinn, advanced)"""
        return {k: v for k, v in self.models.items() if v.model_type == model_type}

    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get info for a specific model"""
        return self.models.get(model_key)

    def refresh_status(self):
        """Refresh training status for all models"""
        self._update_training_status(self.models)

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        total = len(self.models)
        trained = len([m for m in self.models.values() if m.trained])
        untrained = total - trained

        by_type = {}
        for model_type in ['baseline', 'pinn', 'advanced']:
            type_models = self.get_models_by_type(model_type)
            by_type[model_type] = {
                'total': len(type_models),
                'trained': len([m for m in type_models.values() if m.trained]),
                'untrained': len([m for m in type_models.values() if not m.trained])
            }

        return {
            'total_models': total,
            'trained': trained,
            'untrained': untrained,
            'by_type': by_type
        }

    def export_registry(self, output_path: Path):
        """Export registry to JSON"""
        registry_data = {
            'generated': datetime.now().isoformat(),
            'total_models': len(self.models),
            'models': {}
        }

        for key, model in self.models.items():
            registry_data['models'][key] = {
                'model_name': model.model_name,
                'model_type': model.model_type,
                'architecture': model.architecture,
                'description': model.description,
                'physics_constraints': model.physics_constraints,
                'trained': model.trained,
                'checkpoint_path': str(model.checkpoint_path) if model.checkpoint_path else None,
                'results_path': str(model.results_path) if model.results_path else None,
                'training_date': model.training_date,
                'epochs_trained': model.epochs_trained
            }

        with open(output_path, 'w') as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Model registry exported to {output_path}")


def get_model_registry(project_root: Path) -> ModelRegistry:
    """Get the model registry instance"""
    return ModelRegistry(project_root)
