"""
Reproducibility utilities - ensuring deterministic behavior across runs

Provides:
- Full seed control (Python, NumPy, PyTorch, CUDA)
- Environment logging for experiment tracking
- Reproducibility context manager
- Determinism verification
"""

import os
import sys
import random
import platform
import subprocess
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnvironmentInfo:
    """Complete environment information for reproducibility"""
    # System
    platform: str
    platform_version: str
    processor: str
    python_version: str

    # PyTorch
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    gpu_name: Optional[str]
    gpu_count: int

    # NumPy
    numpy_version: str

    # Git
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: bool

    # Timestamp
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Path):
        """Save environment info to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Path) -> 'EnvironmentInfo':
        """Load environment info from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def get_git_info() -> Dict[str, Any]:
    """Get git repository information"""
    try:
        # Get commit hash
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        git_commit = commit.stdout.strip() if commit.returncode == 0 else None

        # Get branch name
        branch = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        git_branch = branch.stdout.strip() if branch.returncode == 0 else None

        # Check if dirty
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        git_dirty = bool(status.stdout.strip()) if status.returncode == 0 else False

        return {
            'commit': git_commit,
            'branch': git_branch,
            'dirty': git_dirty
        }
    except Exception:
        return {'commit': None, 'branch': None, 'dirty': False}


def get_environment_info() -> EnvironmentInfo:
    """
    Collect complete environment information for reproducibility.

    Returns:
        EnvironmentInfo with all system, library, and git details
    """
    git_info = get_git_info()

    # CUDA info
    cuda_available = torch.cuda.is_available()
    cuda_version = None
    cudnn_version = None
    gpu_name = None
    gpu_count = 0

    if cuda_available:
        cuda_version = torch.version.cuda
        cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)

    return EnvironmentInfo(
        platform=platform.system(),
        platform_version=platform.version(),
        processor=platform.processor(),
        python_version=sys.version,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        numpy_version=np.__version__,
        git_commit=git_info['commit'],
        git_branch=git_info['branch'],
        git_dirty=git_info['dirty'],
        timestamp=datetime.now().isoformat()
    )


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of configuration"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries

    Args:
        seed: Random seed value
    """
    logger.info(f"Setting random seed to {seed} for reproducibility")

    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Random seeds set successfully")


def log_system_info():
    """Log system information for reproducibility"""
    import platform
    import torch

    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)

    # Python
    logger.info(f"Python version: {platform.python_version()}")

    # PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # NumPy
    logger.info(f"NumPy version: {np.__version__}")

    # System
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")

    logger.info("=" * 80)


def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """
    Get the appropriate device for PyTorch

    Args:
        prefer_cuda: Whether to prefer CUDA if available
        prefer_mps: Whether to prefer MPS (Apple Silicon) if available

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif prefer_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


class ReproducibilityContext:
    """Context manager for reproducible code blocks"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random_state = None
        self.np_state = None
        self.torch_state = None

    def __enter__(self):
        # Save current states
        self.random_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()

        # Set seed
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore states
        random.setstate(self.random_state)
        np.random.set_state(self.np_state)
        torch.random.set_rng_state(self.torch_state)


def set_deterministic_mode(enabled: bool = True):
    """
    Enable or disable deterministic mode for PyTorch.

    Args:
        enabled: Whether to enable deterministic mode
    """
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                logger.warning("Could not enable full deterministic algorithms")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)


def verify_reproducibility(seed: int = 42, n_samples: int = 100) -> bool:
    """
    Verify that seed setting produces reproducible results.

    Args:
        seed: Seed to test
        n_samples: Number of samples to generate

    Returns:
        True if reproducibility verified
    """
    # First run
    set_seed(seed)
    random_vals1 = [random.random() for _ in range(n_samples)]
    np_vals1 = np.random.randn(n_samples).tolist()
    torch_vals1 = torch.randn(n_samples).tolist()

    # Second run with same seed
    set_seed(seed)
    random_vals2 = [random.random() for _ in range(n_samples)]
    np_vals2 = np.random.randn(n_samples).tolist()
    torch_vals2 = torch.randn(n_samples).tolist()

    # Compare
    random_match = random_vals1 == random_vals2
    np_match = np_vals1 == np_vals2
    torch_match = torch_vals1 == torch_vals2

    if random_match and np_match and torch_match:
        logger.info("Reproducibility verification: PASSED")
        return True
    else:
        logger.warning("Reproducibility verification: FAILED")
        logger.warning(f"  random match: {random_match}")
        logger.warning(f"  numpy match: {np_match}")
        logger.warning(f"  torch match: {torch_match}")
        return False


@dataclass
class ExperimentState:
    """Complete state for experiment reproducibility"""
    seed: int
    environment: EnvironmentInfo
    config_hash: str

    def save(self, path: Path):
        """Save experiment state"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'seed': self.seed,
            'environment': self.environment.to_dict(),
            'config_hash': self.config_hash
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentState':
        """Load experiment state"""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            seed=data['seed'],
            environment=EnvironmentInfo(**data['environment']),
            config_hash=data['config_hash']
        )


def init_experiment(
    seed: int = 42,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    deterministic: bool = True
) -> ExperimentState:
    """
    Initialize experiment with full reproducibility setup.

    Args:
        seed: Random seed
        config: Experiment configuration dict
        output_dir: Directory to save experiment state
        deterministic: Whether to enable deterministic mode

    Returns:
        ExperimentState with all reproducibility info
    """
    logger.info("=" * 60)
    logger.info("INITIALIZING EXPERIMENT")
    logger.info("=" * 60)

    # Set seed
    set_seed(seed)

    # Enable deterministic mode
    if deterministic:
        set_deterministic_mode(True)

    # Get environment info
    env_info = get_environment_info()

    # Compute config hash
    config_hash = compute_config_hash(config) if config else "no_config"

    # Create state
    state = ExperimentState(
        seed=seed,
        environment=env_info,
        config_hash=config_hash
    )

    # Log key info
    logger.info(f"Seed: {seed}")
    logger.info(f"Deterministic: {deterministic}")
    logger.info(f"Config hash: {config_hash}")
    logger.info(f"Git commit: {env_info.git_commit}")
    logger.info(f"Git branch: {env_info.git_branch}")
    logger.info(f"Git dirty: {env_info.git_dirty}")
    logger.info(f"PyTorch: {env_info.torch_version}")
    logger.info(f"CUDA: {env_info.cuda_version}")
    logger.info(f"GPU: {env_info.gpu_name}")

    # Save state if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        state.save(output_dir / "experiment_state.json")
        env_info.save(output_dir / "environment.json")
        logger.info(f"Experiment state saved to {output_dir}")

    logger.info("=" * 60)

    return state


@dataclass
class ScalerParams:
    """
    Scaler parameters for data normalisation.

    CRITICAL FOR REPRODUCIBILITY:
    These values are needed to de-standardise predictions before computing
    financial metrics. Without them, metrics like Sharpe ratio are meaningless.
    """
    close_mean: float
    close_std: float
    ticker: Optional[str] = None
    date_range: Optional[str] = None  # e.g., "2020-01-01 to 2023-12-31"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionAssumptions:
    """
    Trading execution assumptions for backtesting.

    These should be stored with results for research transparency.
    """
    execution_model: str = "close_to_close"  # When trades execute
    transaction_cost: float = 0.001  # 10 basis points
    slippage: float = 0.0  # Slippage model (0 = none)
    position_lag: int = 1  # Signal at t executes at t+lag
    max_leverage: float = 1.0  # Maximum position size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentMetadata:
    """
    Complete metadata for dissertation-ready experiment reproducibility.

    This class captures ALL information needed to reproduce an experiment:
    - Configuration hash for parameter tracking
    - Scaler parameters for de-normalisation
    - Execution assumptions for backtesting
    - All random seeds used
    - Environment information
    - Git commit and dirty status
    """
    # Experiment identification
    experiment_id: str
    experiment_name: str
    timestamp: str

    # Configuration
    config_hash: str  # SHA256 hash of full config
    config: Dict[str, Any]

    # Model information
    model_key: str
    model_type: str  # 'baseline', 'pinn', 'advanced'
    is_causal: bool = True  # Causal vs oracle

    # Scaler parameters (CRITICAL for de-standardisation)
    scaler_params: Dict[str, ScalerParams] = None  # Keyed by ticker

    # Execution assumptions
    execution: ExecutionAssumptions = None

    # Seeds
    seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42
    python_seed: int = 42

    # Environment
    environment: EnvironmentInfo = None

    # Dataset version
    dataset_version: str = None
    data_date_range: str = None
    n_samples: int = 0
    n_features: int = 0

    def __post_init__(self):
        if self.scaler_params is None:
            self.scaler_params = {}
        if self.execution is None:
            self.execution = ExecutionAssumptions()

    def add_scaler_params(self, ticker: str, close_mean: float, close_std: float,
                          date_range: str = None):
        """Add scaler parameters for a ticker"""
        self.scaler_params[ticker] = ScalerParams(
            close_mean=close_mean,
            close_std=close_std,
            ticker=ticker,
            date_range=date_range
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'config_hash': self.config_hash,
            'config': self.config,
            'model_key': self.model_key,
            'model_type': self.model_type,
            'is_causal': self.is_causal,
            'scaler_params': {k: v.to_dict() for k, v in self.scaler_params.items()},
            'execution': self.execution.to_dict() if self.execution else None,
            'seeds': {
                'seed': self.seed,
                'torch_seed': self.torch_seed,
                'numpy_seed': self.numpy_seed,
                'python_seed': self.python_seed
            },
            'environment': self.environment.to_dict() if self.environment else None,
            'dataset_version': self.dataset_version,
            'data_date_range': self.data_date_range,
            'n_samples': self.n_samples,
            'n_features': self.n_features
        }

    def save(self, path: Path):
        """Save experiment metadata to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved experiment metadata to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ExperimentMetadata':
        """Load experiment metadata from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct nested objects
        scaler_params = {}
        for k, v in data.get('scaler_params', {}).items():
            scaler_params[k] = ScalerParams(**v)

        execution = ExecutionAssumptions(**data['execution']) if data.get('execution') else None
        environment = EnvironmentInfo(**data['environment']) if data.get('environment') else None

        seeds = data.get('seeds', {})

        return cls(
            experiment_id=data['experiment_id'],
            experiment_name=data['experiment_name'],
            timestamp=data['timestamp'],
            config_hash=data['config_hash'],
            config=data['config'],
            model_key=data['model_key'],
            model_type=data['model_type'],
            is_causal=data.get('is_causal', True),
            scaler_params=scaler_params,
            execution=execution,
            seed=seeds.get('seed', 42),
            torch_seed=seeds.get('torch_seed', 42),
            numpy_seed=seeds.get('numpy_seed', 42),
            python_seed=seeds.get('python_seed', 42),
            environment=environment,
            dataset_version=data.get('dataset_version'),
            data_date_range=data.get('data_date_range'),
            n_samples=data.get('n_samples', 0),
            n_features=data.get('n_features', 0)
        )


def create_experiment_metadata(
    experiment_name: str,
    config: Dict[str, Any],
    model_key: str,
    model_type: str,
    seed: int = 42,
    scaler_params: Dict[str, tuple] = None,  # {ticker: (mean, std)}
    transaction_cost: float = 0.001,
    is_causal: bool = True
) -> ExperimentMetadata:
    """
    Create experiment metadata with all reproducibility information.

    Args:
        experiment_name: Human-readable name
        config: Full training configuration
        model_key: Model key from registry
        model_type: Model type (baseline, pinn, advanced)
        seed: Random seed
        scaler_params: Dict mapping ticker to (close_mean, close_std) tuple
        transaction_cost: Transaction cost for backtesting
        is_causal: Whether model is causal (no look-ahead)

    Returns:
        ExperimentMetadata ready to be saved
    """
    import uuid

    # Generate experiment ID
    experiment_id = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Compute config hash
    config_hash = compute_config_hash(config)

    # Get environment
    environment = get_environment_info()

    # Create metadata
    metadata = ExperimentMetadata(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
        config_hash=config_hash,
        config=config,
        model_key=model_key,
        model_type=model_type,
        is_causal=is_causal,
        seed=seed,
        torch_seed=seed,
        numpy_seed=seed,
        python_seed=seed,
        environment=environment,
        execution=ExecutionAssumptions(transaction_cost=transaction_cost)
    )

    # Add scaler params
    if scaler_params:
        for ticker, (mean, std) in scaler_params.items():
            metadata.add_scaler_params(ticker, mean, std)

    return metadata


class MultiSeedRunner:
    """
    Run experiments with multiple seeds for statistical robustness.

    Example:
        runner = MultiSeedRunner(base_seed=42, n_seeds=5)
        for seed in runner.seeds:
            runner.set_seed(seed)
            # Run experiment
            result = train_model(...)
            runner.record_result(seed, result)
        summary = runner.get_summary()
    """

    def __init__(self, base_seed: int = 42, n_seeds: int = 5):
        self.base_seed = base_seed
        self.n_seeds = n_seeds
        self.seeds = [base_seed + i * 1000 for i in range(n_seeds)]
        self.results: Dict[int, Dict[str, Any]] = {}

    def set_seed(self, seed: int):
        """Set seed for current run"""
        set_seed(seed)

    def record_result(self, seed: int, result: Dict[str, Any]):
        """Record result for a seed"""
        self.results[seed] = result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all seeds"""
        if not self.results:
            return {}

        # Collect all metrics
        all_metrics = {}
        for result in self.results.values():
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # Compute statistics
        summary = {
            'n_seeds': len(self.results),
            'seeds': list(self.results.keys()),
            'metrics': {}
        }

        for metric, values in all_metrics.items():
            arr = np.array(values)
            summary['metrics'][metric] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr))
            }

        return summary
