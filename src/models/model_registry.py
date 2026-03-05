"""
Model Registry - Central registry for all neural network models

Tracks all available models, their training status, and metadata.
Provides model loading functionality from checkpoints.

Performance optimizations:
- Uses glob patterns instead of multiple path.exists() calls
- Caches checkpoint locations in memory
- Provides Streamlit-compatible caching decorator
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime
import time

import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache for checkpoint locations (avoids repeated filesystem scans)
_checkpoint_cache: Dict[str, Dict] = {}
_cache_timestamp: float = 0
_CACHE_TTL_SECONDS = 60  # Cache checkpoint locations for 60 seconds


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

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models with metadata for UI/tests."""
        output: List[Dict[str, Any]] = []
        for key, info in self.models.items():
            output.append({
                'model_key': info.model_key,
                'model_name': info.model_name,
                'model_type': info.model_type,
                'architecture': info.architecture,
                'description': info.description,
                'physics_constraints': info.physics_constraints,
                'trained': info.trained,
                'checkpoint_path': str(info.checkpoint_path) if info.checkpoint_path else None,
                'results_path': str(info.results_path) if info.results_path else None,
                'training_date': info.training_date,
                'epochs_trained': info.epochs_trained,
            })
        return output

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
            model_key='baseline_pinn',
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

        models['spectral_pinn'] = ModelInfo(
            model_key='spectral_pinn',
            model_name='Spectral Regime PINN',
            model_type='advanced',
            architecture='SpectralRegimePINN',
            description='Spectral encoder + regime conditioning + physics constraints',
            physics_constraints={
                'lambda_gbm': 0.1,
                'lambda_ou': 0.1,
                'lambda_autocorr': 0.05,
                'lambda_spectral': 0.05
            }
        )

        # ========== ALIASES for API compatibility ==========
        # Add pinn_ prefixed aliases for PINN models
        models['pinn_baseline'] = models['baseline_pinn']
        models['baseline'] = models['baseline_pinn']
        models['pinn_gbm'] = models['gbm']
        models['pinn_ou'] = models['ou']
        models['pinn_black_scholes'] = models['black_scholes']
        models['pinn_gbm_ou'] = models['gbm_ou']
        models['pinn_global'] = models['global']
        models['stacked_pinn'] = models['stacked']
        models['residual_pinn'] = models['residual']

        # ========== VOLATILITY FORECASTING MODELS ==========
        models['vol_lstm'] = ModelInfo(
            model_key='vol_lstm',
            model_name='Volatility LSTM',
            model_type='volatility',
            architecture='VolatilityLSTM',
            description='LSTM for volatility forecasting with Softplus output'
        )

        models['vol_gru'] = ModelInfo(
            model_key='vol_gru',
            model_name='Volatility GRU',
            model_type='volatility',
            architecture='VolatilityGRU',
            description='GRU for volatility forecasting'
        )

        models['vol_transformer'] = ModelInfo(
            model_key='vol_transformer',
            model_name='Volatility Transformer',
            model_type='volatility',
            architecture='VolatilityTransformer',
            description='Transformer for volatility forecasting'
        )

        models['vol_pinn'] = ModelInfo(
            model_key='vol_pinn',
            model_name='Volatility PINN',
            model_type='volatility',
            architecture='VolatilityPINN',
            description='PINN with variance mean-reversion and GARCH constraints',
            physics_constraints={'lambda_ou': 0.1, 'lambda_garch': 0.1, 'lambda_feller': 0.05, 'lambda_leverage': 0.05}
        )

        models['heston_pinn'] = ModelInfo(
            model_key='heston_pinn',
            model_name='Heston PINN',
            model_type='volatility',
            architecture='HestonPINN',
            description='PINN based on Heston stochastic volatility model',
            physics_constraints={'lambda_heston': 0.1, 'lambda_feller': 0.05, 'lambda_leverage': 0.05}
        )

        models['stacked_vol_pinn'] = ModelInfo(
            model_key='stacked_vol_pinn',
            model_name='Stacked Volatility PINN',
            model_type='volatility',
            architecture='StackedVolatilityPINN',
            description='Advanced stacked architecture for volatility forecasting',
            physics_constraints={'lambda_ou': 0.1, 'lambda_garch': 0.1, 'lambda_feller': 0.05, 'lambda_leverage': 0.05}
        )

        # Volatility model aliases
        models['volatility_lstm'] = models['vol_lstm']
        models['volatility_gru'] = models['vol_gru']
        models['volatility_transformer'] = models['vol_transformer']
        models['volatility_pinn'] = models['vol_pinn']
        models['heston'] = models['heston_pinn']
        models['stacked_volatility_pinn'] = models['stacked_vol_pinn']

        # Check training status for all models
        self._update_training_status(models)

        return models

    def _update_training_status(self, models: Dict[str, ModelInfo]):
        """
        Check which models have been trained.

        OPTIMIZED: Uses glob patterns to scan all checkpoints at once,
        then matches them to models. This is much faster than checking
        individual paths for each model.
        """
        global _checkpoint_cache, _cache_timestamp

        current_time = time.time()

        # Use cached checkpoint locations if available and not expired
        if _checkpoint_cache and (current_time - _cache_timestamp) < _CACHE_TTL_SECONDS:
            checkpoint_map = _checkpoint_cache
            logger.debug("Using cached checkpoint locations")
        else:
            # Build checkpoint map using glob (single filesystem scan)
            checkpoint_map = self._scan_all_checkpoints()
            _checkpoint_cache = checkpoint_map
            _cache_timestamp = current_time
            logger.debug(f"Scanned {len(checkpoint_map)} checkpoint files")

        # Load detailed_results.json once for all PINN models
        detailed_results_data = None
        detailed_path = self.results_dir / 'pinn_comparison' / 'detailed_results.json'
        if detailed_path.exists():
            try:
                with open(detailed_path, 'r') as f:
                    detailed_results_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Could not load detailed results: {e}")

        # Match checkpoints to models
        for model_key, model_info in models.items():
            # Check if this model has a checkpoint
            checkpoint_path = checkpoint_map.get(model_key)

            if checkpoint_path:
                model_info.trained = True
                model_info.checkpoint_path = checkpoint_path
                try:
                    model_info.training_date = datetime.fromtimestamp(
                        checkpoint_path.stat().st_mtime
                    ).strftime('%Y-%m-%d %H:%M')
                except OSError:
                    model_info.training_date = None

                # Try to load epochs from results
                self._load_model_epochs(model_key, model_info, detailed_results_data)

    def _scan_all_checkpoints(self) -> Dict[str, Path]:
        """
        Scan all checkpoint files at once using glob patterns.
        Returns a mapping of model_key -> checkpoint_path.

        This is much faster than checking individual paths because:
        1. Single filesystem traversal instead of multiple exists() calls
        2. Results cached for subsequent calls
        """
        checkpoint_map = {}

        # Scan main models directory
        if self.models_dir.exists():
            for pt_file in self.models_dir.glob('*_best.pt'):
                # Extract model key from filename
                # e.g., "lstm_best.pt" -> "lstm", "pinn_gbm_best.pt" -> "pinn_gbm" or "gbm"
                name = pt_file.stem.replace('_best', '')
                checkpoint_map[name] = pt_file

                # Also map without pinn_ prefix for easier lookup
                if name.startswith('pinn_'):
                    short_name = name[5:]  # Remove 'pinn_' prefix
                    if short_name not in checkpoint_map:
                        checkpoint_map[short_name] = pt_file
                    if 'baseline' in short_name and 'baseline_pinn' not in checkpoint_map:
                        checkpoint_map['baseline_pinn'] = pt_file

            # Also check .pth files
            for pth_file in self.models_dir.glob('*_best.pth'):
                name = pth_file.stem.replace('_best', '')
                if name not in checkpoint_map:
                    checkpoint_map[name] = pth_file
                if name.startswith('pinn_'):
                    short_name = name[5:]
                    if short_name not in checkpoint_map:
                        checkpoint_map[short_name] = pth_file
                    if 'baseline' in short_name and 'baseline_pinn' not in checkpoint_map:
                        checkpoint_map['baseline_pinn'] = pth_file

        # Scan stacked_pinn subdirectory
        stacked_dir = self.models_dir / 'stacked_pinn'
        if stacked_dir.exists():
            for pt_file in stacked_dir.glob('*_best.pt'):
                name = pt_file.stem.replace('_best', '').replace('_pinn', '')
                # Map both full and short names
                checkpoint_map[name] = pt_file

                # Special handling for stacked/residual
                if 'stacked_pinn' in pt_file.stem:
                    checkpoint_map['stacked'] = pt_file
                    checkpoint_map['stacked_pinn'] = pt_file
                elif 'residual_pinn' in pt_file.stem:
                    checkpoint_map['residual'] = pt_file
                    checkpoint_map['residual_pinn'] = pt_file

        return checkpoint_map

    def _load_model_epochs(
        self,
        model_key: str,
        model_info: ModelInfo,
        detailed_results_data: Optional[List] = None
    ):
        """Load epochs trained from results files."""
        # Try multiple result file patterns
        result_patterns = [
            self.results_dir / f'{model_key}_results.json',
            self.results_dir / f'pinn_{model_key}_results.json',
            self.results_dir / 'pinn_comparison' / f'{model_key}_results.json',
            self.models_dir / 'stacked_pinn' / f'{model_key}_pinn_results.json',
        ]

        for results_path in result_patterns:
            if results_path.exists():
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                        if 'history' in results and 'train_loss' in results['history']:
                            model_info.epochs_trained = len(results['history']['train_loss'])
                        model_info.results_path = results_path
                        return
                except (json.JSONDecodeError, IOError, KeyError) as e:
                    logger.debug(f"Could not load results from {results_path}: {e}")

        # Try detailed_results.json for PINN models
        if detailed_results_data and model_info.model_type == 'pinn':
            for variant_result in detailed_results_data:
                if variant_result.get('variant_key') == model_key:
                    if 'history' in variant_result and 'train_loss' in variant_result['history']:
                        model_info.epochs_trained = len(variant_result['history']['train_loss'])
                    model_info.results_path = self.results_dir / 'pinn_comparison' / 'detailed_results.json'
                    return

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

    def load_model(
        self,
        model_key: str,
        device: Optional[torch.device] = None,
        input_dim: int = 5
    ) -> Optional[nn.Module]:
        """
        Load a trained model from checkpoint

        Args:
            model_key: The model key (e.g., 'lstm', 'pinn_global', 'stacked')
            device: Device to load model on (default: CPU)
            input_dim: Input feature dimension (default: 5)

        Returns:
            Loaded model or None if not found/trained
        """
        model_info = self.get_model_info(model_key)

        if model_info is None:
            logger.error(f"Model '{model_key}' not found in registry")
            return None

        if not model_info.trained or model_info.checkpoint_path is None:
            logger.error(f"Model '{model_key}' is not trained or checkpoint not found")
            return None

        device = device or torch.device('cpu')

        try:
            # Load checkpoint to CPU first to avoid MPS/CUDA device compatibility issues
            # Then move to target device after loading
            # weights_only=False is safe here as these are our own trained model checkpoints
            checkpoint = torch.load(model_info.checkpoint_path, map_location='cpu', weights_only=False)

            # Extract training-time hyperparameters from checkpoint if present
            cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
            model_cfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}
            research_cfg = checkpoint.get('research_config', {}) if isinstance(checkpoint, dict) else {}
            data_cfg = cfg.get('data', {}) if isinstance(cfg, dict) else {}

            # Prefer saved hyperparameters; fall back to defaults
            input_dim_ckpt = model_cfg.get('input_dim') or data_cfg.get('input_dim') or len(data_cfg.get('feature_cols', [])) or input_dim
            hidden_dim_ckpt = model_cfg.get('hidden_dim') or research_cfg.get('hidden_dim') or 128
            num_layers_ckpt = model_cfg.get('num_layers') or research_cfg.get('num_layers') or 2
            dropout_ckpt = model_cfg.get('dropout', 0.2)
            base_model_ckpt = model_cfg.get('base_model', 'lstm')

            # Infer dimensions from state_dict if mismatch (handles research-mode checkpoints)
            state_dict = checkpoint.get('model_state_dict', {})
            try:
                ih_keys = [k for k in state_dict.keys() if 'weight_ih_l0' in k]
                if ih_keys:
                    ih0 = state_dict[ih_keys[0]]
                    gates = ih0.shape[0] // ih0.shape[1]  # rough; adjust below
                    input_dim_ckpt = ih0.shape[1]
                    # Determine gate factor (LSTM=4, GRU=3)
                    gate_factor = 4 if 'lstm' in ih_keys[0] else 3
                    hidden_dim_ckpt = ih0.shape[0] // gate_factor
                    # Count layers
                    layer_ids = set()
                    for k in state_dict:
                        if 'weight_ih_l' in k:
                            try:
                                layer_ids.add(int(k.split('weight_ih_l')[-1]))
                            except ValueError:
                                pass
                    if layer_ids:
                        num_layers_ckpt = max(layer_ids) + 1
            except Exception:
                pass

            # Instantiate model based on architecture with checkpoint hyperparams
            model = self._instantiate_model(
                model_info,
                input_dim=input_dim_ckpt,
                hidden_dim=hidden_dim_ckpt,
                num_layers=num_layers_ckpt,
                dropout=dropout_ckpt,
                base_model=base_model_ckpt,
            )

            if model is None:
                return None

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            logger.info(f"Loaded model '{model_key}' from {model_info.checkpoint_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model '{model_key}': {e}")
            return None

    def _instantiate_model(
        self,
        model_info: ModelInfo,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        base_model: str = 'lstm',
    ) -> Optional[nn.Module]:
        """
        Instantiate a model based on its architecture

        Args:
            model_info: Model information
            input_dim: Input feature dimension

        Returns:
            Instantiated model (not loaded with weights yet)
        """
        architecture = model_info.architecture
        physics = model_info.physics_constraints or {}

        try:
            if architecture == 'LSTM':
                from .baseline import LSTMModel
                return LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=1,
                    dropout=dropout
                )

            elif architecture == 'GRU':
                from .baseline import GRUModel
                return GRUModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=1,
                    dropout=dropout
                )

            elif architecture == 'BiLSTM':
                from .baseline import LSTMModel
                return LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=1,
                    dropout=dropout,
                    bidirectional=True
                )

            elif architecture == 'AttentionLSTM':
                from .baseline import LSTMModel
                # Attention LSTM uses same base with attention layer
                return LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=1,
                    dropout=dropout
                )

            elif architecture == 'Transformer':
                from .transformer import TransformerModel
                return TransformerModel(
                    input_dim=input_dim,
                    d_model=hidden_dim,
                    nhead=4,
                    num_encoder_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'PINN':
                from .pinn import PINNModel
                return PINNModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=1,
                    dropout=dropout,
                    base_model=base_model,
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_bs=physics.get('lambda_bs', 0.0),
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_langevin=physics.get('lambda_langevin', 0.0)
                )

            elif architecture == 'StackedPINN':
                from .stacked_pinn import StackedPINN
                return StackedPINN(
                    input_dim=input_dim,
                    encoder_dim=hidden_dim,
                    lstm_hidden_dim=hidden_dim,
                    num_encoder_layers=num_layers,
                    num_rnn_layers=num_layers,
                    prediction_hidden_dim=max(hidden_dim // 2, 32),
                    dropout=dropout,
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_ou=physics.get('lambda_ou', 0.1)
                )

            elif architecture == 'ResidualPINN':
                from .stacked_pinn import ResidualPINN
                return ResidualPINN(
                    input_dim=input_dim,
                    base_model_type=base_model,
                    base_hidden_dim=hidden_dim,
                    correction_hidden_dim=max(hidden_dim // 2, 32),
                    num_base_layers=num_layers,
                    num_correction_layers=num_layers,
                    dropout=dropout,
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_ou=physics.get('lambda_ou', 0.1)
                )

            elif architecture == 'SpectralRegimePINN':
                from .spectral_pinn import SpectralRegimePINN
                return SpectralRegimePINN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    n_regimes=3,
                    num_layers=num_layers,
                    dropout=dropout,
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_autocorr=physics.get('lambda_autocorr', 0.05),
                    lambda_spectral=physics.get('lambda_spectral', 0.05)
                )

            # ========== VOLATILITY FORECASTING ARCHITECTURES ==========
            elif architecture == 'VolatilityLSTM':
                from .volatility import VolatilityLSTM
                return VolatilityLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'VolatilityGRU':
                from .volatility import VolatilityGRU
                return VolatilityGRU(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'VolatilityTransformer':
                from .volatility import VolatilityTransformer
                return VolatilityTransformer(
                    input_dim=input_dim,
                    d_model=hidden_dim,
                    nhead=4,
                    num_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'VolatilityPINN':
                from .volatility import VolatilityPINN
                return VolatilityPINN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    base_model='lstm',
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_garch=physics.get('lambda_garch', 0.1),
                    lambda_feller=physics.get('lambda_feller', 0.05),
                    lambda_leverage=physics.get('lambda_leverage', 0.05)
                )

            elif architecture == 'HestonPINN':
                from .volatility import HestonPINN
                return HestonPINN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    lambda_heston=physics.get('lambda_heston', 0.1),
                    lambda_feller=physics.get('lambda_feller', 0.05),
                    lambda_leverage=physics.get('lambda_leverage', 0.05)
                )

            elif architecture == 'StackedVolatilityPINN':
                from .volatility import StackedVolatilityPINN
                return StackedVolatilityPINN(
                    input_dim=input_dim,
                    encoder_dim=hidden_dim // 2,
                    rnn_hidden_dim=hidden_dim,
                    num_encoder_layers=num_layers,
                    num_rnn_layers=num_layers,
                    dropout=dropout,
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_garch=physics.get('lambda_garch', 0.1),
                    lambda_feller=physics.get('lambda_feller', 0.05),
                    lambda_leverage=physics.get('lambda_leverage', 0.05)
                )

            else:
                logger.error(f"Unknown architecture: {architecture}")
                return None

        except Exception as e:
            logger.error(f"Failed to instantiate model: {e}")
            return None

    def get_available_models_for_inference(self) -> List[str]:
        """
        Get list of model keys that are available for inference (trained)

        Returns:
            List of model keys
        """
        return [k for k, v in self.models.items() if v.trained]

    def create_model(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> Optional[nn.Module]:
        """
        Create a fresh model instance for training.

        Args:
            model_type: The model key (e.g., 'lstm', 'pinn_gbm', 'transformer')
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout rate
            output_dim: Output dimension

        Returns:
            Fresh model instance ready for training
        """
        model_info = self.get_model_info(model_type)

        if model_info is None:
            logger.error(f"Model type '{model_type}' not found in registry")
            return None

        architecture = model_info.architecture
        physics = model_info.physics_constraints or {}

        try:
            if architecture == 'LSTM':
                from .baseline import LSTMModel
                return LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    dropout=dropout
                )

            elif architecture == 'GRU':
                from .baseline import GRUModel
                return GRUModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    dropout=dropout
                )

            elif architecture == 'BiLSTM':
                from .baseline import LSTMModel
                return LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    dropout=dropout,
                    bidirectional=True
                )

            elif architecture == 'AttentionLSTM':
                from .baseline import LSTMModel
                return LSTMModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    dropout=dropout
                )

            elif architecture == 'Transformer':
                from .transformer import TransformerModel
                return TransformerModel(
                    input_dim=input_dim,
                    d_model=hidden_dim // 2,  # Transformer uses smaller d_model
                    nhead=4,
                    num_encoder_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'PINN':
                from .pinn import PINNModel
                return PINNModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    output_dim=output_dim,
                    dropout=dropout,
                    base_model='lstm',
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_bs=physics.get('lambda_bs', 0.0),
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_langevin=physics.get('lambda_langevin', 0.0)
                )

            elif architecture == 'StackedPINN':
                from .stacked_pinn import StackedPINN
                return StackedPINN(
                    input_dim=input_dim,
                    encoder_dim=hidden_dim,
                    lstm_hidden_dim=hidden_dim,
                    num_encoder_layers=num_layers,
                    num_rnn_layers=num_layers,
                    prediction_hidden_dim=hidden_dim // 2,
                    dropout=dropout,
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_ou=physics.get('lambda_ou', 0.1)
                )

            elif architecture == 'ResidualPINN':
                from .stacked_pinn import ResidualPINN
                return ResidualPINN(
                    input_dim=input_dim,
                    base_model_type='lstm',
                    base_hidden_dim=hidden_dim,
                    correction_hidden_dim=hidden_dim // 2,
                    num_base_layers=num_layers,
                    num_correction_layers=num_layers,
                    dropout=dropout,
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_ou=physics.get('lambda_ou', 0.1)
                )

            elif architecture == 'SpectralRegimePINN':
                from .spectral_pinn import SpectralRegimePINN
                return SpectralRegimePINN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    n_regimes=3,
                    num_layers=num_layers,
                    dropout=dropout,
                    lambda_gbm=physics.get('lambda_gbm', 0.1),
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_autocorr=physics.get('lambda_autocorr', 0.05),
                    lambda_spectral=physics.get('lambda_spectral', 0.05)
                )

            # ========== VOLATILITY FORECASTING ARCHITECTURES ==========
            elif architecture == 'VolatilityLSTM':
                from .volatility import VolatilityLSTM
                return VolatilityLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'VolatilityGRU':
                from .volatility import VolatilityGRU
                return VolatilityGRU(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'VolatilityTransformer':
                from .volatility import VolatilityTransformer
                return VolatilityTransformer(
                    input_dim=input_dim,
                    d_model=hidden_dim,
                    nhead=4,
                    num_layers=num_layers,
                    dropout=dropout
                )

            elif architecture == 'VolatilityPINN':
                from .volatility import VolatilityPINN
                return VolatilityPINN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    base_model='lstm',
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_garch=physics.get('lambda_garch', 0.1),
                    lambda_feller=physics.get('lambda_feller', 0.05),
                    lambda_leverage=physics.get('lambda_leverage', 0.05)
                )

            elif architecture == 'HestonPINN':
                from .volatility import HestonPINN
                return HestonPINN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    lambda_heston=physics.get('lambda_heston', 0.1),
                    lambda_feller=physics.get('lambda_feller', 0.05),
                    lambda_leverage=physics.get('lambda_leverage', 0.05)
                )

            elif architecture == 'StackedVolatilityPINN':
                from .volatility import StackedVolatilityPINN
                return StackedVolatilityPINN(
                    input_dim=input_dim,
                    encoder_dim=hidden_dim // 2,
                    rnn_hidden_dim=hidden_dim,
                    num_encoder_layers=num_layers,
                    num_rnn_layers=num_layers,
                    dropout=dropout,
                    lambda_ou=physics.get('lambda_ou', 0.1),
                    lambda_garch=physics.get('lambda_garch', 0.1),
                    lambda_feller=physics.get('lambda_feller', 0.05),
                    lambda_leverage=physics.get('lambda_leverage', 0.05)
                )

            else:
                logger.error(f"Unknown architecture: {architecture}")
                return None

        except Exception as e:
            logger.error(f"Failed to create model '{model_type}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def refresh_status(self):
        """Refresh training status for all models"""
        self._update_training_status(self.models)

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        total = len(self.models)
        trained = len([m for m in self.models.values() if m.trained])
        untrained = total - trained

        by_type = {}
        for model_type in ['baseline', 'pinn', 'advanced', 'volatility']:
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


def clear_checkpoint_cache():
    """Clear the checkpoint cache to force a rescan."""
    global _checkpoint_cache, _cache_timestamp
    _checkpoint_cache = {}
    _cache_timestamp = 0
    logger.debug("Checkpoint cache cleared")


# Streamlit-compatible cached registry getter
# This function can be decorated with @st.cache_resource when imported in Streamlit apps
def get_cached_model_registry(project_root: Path) -> ModelRegistry:
    """
    Get a cached model registry instance.

    When used with Streamlit's @st.cache_resource decorator, this prevents
    repeated filesystem scans on every page interaction.

    Usage in Streamlit apps:
        @st.cache_resource
        def get_registry():
            return get_cached_model_registry(config.project_root)
    """
    return ModelRegistry(project_root)
