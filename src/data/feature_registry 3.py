"""
Feature Provenance Registry

Provides centralized tracking of all features used in the PINN financial forecasting system:
- Feature definitions with formulas and computation details
- Data source and availability time tracking
- Lookahead bias prevention
- Feature dependency graphs
- Point-in-time correctness validation

This ensures research reproducibility and prevents data leakage.
"""

import yaml
import hashlib
import json
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureType(Enum):
    """Types of features in the system"""
    RAW = "raw"  # Direct from data source
    DERIVED = "derived"  # Computed from other features
    TECHNICAL = "technical"  # Technical indicators
    FUNDAMENTAL = "fundamental"  # Fundamental data
    MACRO = "macro"  # Macroeconomic indicators
    SENTIMENT = "sentiment"  # Sentiment/alternative data
    TARGET = "target"  # Prediction targets


class DataSource(Enum):
    """Data sources for features"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    COMPUTED = "computed"
    EXTERNAL = "external"


class AvailabilityType(Enum):
    """When the data becomes available"""
    REALTIME = "realtime"  # Available immediately
    END_OF_DAY = "end_of_day"  # Available after market close
    DELAYED = "delayed"  # Available with some delay (e.g., 15 min)
    NEXT_DAY = "next_day"  # Available next trading day
    WEEKLY = "weekly"  # Updated weekly
    MONTHLY = "monthly"  # Updated monthly
    QUARTERLY = "quarterly"  # Updated quarterly


@dataclass
class FeatureDefinition:
    """Complete definition of a feature"""
    name: str
    description: str
    feature_type: FeatureType
    data_source: DataSource
    formula: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    lag: int = 0  # Minimum lag required to avoid lookahead
    availability: AvailabilityType = AvailabilityType.REALTIME
    availability_time: Optional[str] = None  # e.g., "16:00" for EOD
    unit: Optional[str] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    nullable: bool = True
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d['feature_type'] = self.feature_type.value
        d['data_source'] = self.data_source.value
        d['availability'] = self.availability.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureDefinition':
        """Create from dictionary"""
        d = d.copy()
        d['feature_type'] = FeatureType(d['feature_type'])
        d['data_source'] = DataSource(d['data_source'])
        d['availability'] = AvailabilityType(d['availability'])
        return cls(**d)


@dataclass
class FeatureValidationResult:
    """Result of feature validation"""
    feature_name: str
    is_valid: bool
    has_lookahead_risk: bool
    missing_dependencies: List[str]
    circular_dependencies: List[str]
    warnings: List[str]
    errors: List[str]


class FeatureRegistry:
    """
    Central registry for all features used in the system.

    Provides:
    - Feature definition management
    - Dependency tracking
    - Lookahead bias detection
    - Point-in-time validation
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize feature registry.

        Args:
            config_path: Path to features.yaml configuration file
        """
        self.features: Dict[str, FeatureDefinition] = {}
        self.config_path = config_path

        # Load built-in features
        self._register_builtin_features()

        # Load from config if provided
        if config_path and config_path.exists():
            self.load_from_yaml(config_path)

    def _register_builtin_features(self) -> None:
        """Register all built-in features used in the system"""

        # ========== RAW PRICE FEATURES ==========
        self.register(FeatureDefinition(
            name="open",
            description="Opening price for the trading period",
            feature_type=FeatureType.RAW,
            data_source=DataSource.YAHOO_FINANCE,
            availability=AvailabilityType.REALTIME,
            unit="USD",
            range_min=0,
            nullable=False
        ))

        self.register(FeatureDefinition(
            name="high",
            description="Highest price during the trading period",
            feature_type=FeatureType.RAW,
            data_source=DataSource.YAHOO_FINANCE,
            availability=AvailabilityType.END_OF_DAY,
            availability_time="16:00",
            unit="USD",
            range_min=0,
            nullable=False
        ))

        self.register(FeatureDefinition(
            name="low",
            description="Lowest price during the trading period",
            feature_type=FeatureType.RAW,
            data_source=DataSource.YAHOO_FINANCE,
            availability=AvailabilityType.END_OF_DAY,
            availability_time="16:00",
            unit="USD",
            range_min=0,
            nullable=False
        ))

        self.register(FeatureDefinition(
            name="close",
            description="Closing/adjusted closing price for the trading period",
            feature_type=FeatureType.RAW,
            data_source=DataSource.YAHOO_FINANCE,
            availability=AvailabilityType.END_OF_DAY,
            availability_time="16:00",
            unit="USD",
            range_min=0,
            nullable=False,
            notes="Primary target for price prediction models"
        ))

        self.register(FeatureDefinition(
            name="volume",
            description="Trading volume in shares",
            feature_type=FeatureType.RAW,
            data_source=DataSource.YAHOO_FINANCE,
            availability=AvailabilityType.END_OF_DAY,
            availability_time="16:00",
            unit="shares",
            range_min=0,
            nullable=False
        ))

        # ========== RETURN FEATURES ==========
        self.register(FeatureDefinition(
            name="log_return",
            description="Logarithmic return: ln(P_t / P_{t-1})",
            feature_type=FeatureType.DERIVED,
            data_source=DataSource.COMPUTED,
            formula="np.log(close / close.shift(1))",
            dependencies=["close"],
            lag=1,
            availability=AvailabilityType.END_OF_DAY,
            unit="log ratio",
            notes="Preferred for financial modeling due to additivity and normality"
        ))

        self.register(FeatureDefinition(
            name="simple_return",
            description="Simple percentage return: (P_t - P_{t-1}) / P_{t-1}",
            feature_type=FeatureType.DERIVED,
            data_source=DataSource.COMPUTED,
            formula="(close - close.shift(1)) / close.shift(1)",
            dependencies=["close"],
            lag=1,
            availability=AvailabilityType.END_OF_DAY,
            unit="percent"
        ))

        # ========== VOLATILITY FEATURES ==========
        for window in [5, 20, 60]:
            self.register(FeatureDefinition(
                name=f"rolling_volatility_{window}",
                description=f"Rolling standard deviation of log returns over {window} periods",
                feature_type=FeatureType.DERIVED,
                data_source=DataSource.COMPUTED,
                formula=f"log_return.rolling(window={window}).std()",
                dependencies=["log_return"],
                lag=window,
                availability=AvailabilityType.END_OF_DAY,
                unit="log ratio std",
                notes=f"Annualized by multiplying by sqrt(252) for daily data"
            ))

        # ========== MOMENTUM FEATURES ==========
        for window in [5, 10, 20, 60]:
            self.register(FeatureDefinition(
                name=f"momentum_{window}",
                description=f"Rate of change over {window} periods: (P_t / P_{{t-{window}}}) - 1",
                feature_type=FeatureType.DERIVED,
                data_source=DataSource.COMPUTED,
                formula=f"(close / close.shift({window})) - 1",
                dependencies=["close"],
                lag=window,
                availability=AvailabilityType.END_OF_DAY,
                unit="percent"
            ))

            self.register(FeatureDefinition(
                name=f"sma_{window}",
                description=f"Simple moving average over {window} periods",
                feature_type=FeatureType.DERIVED,
                data_source=DataSource.COMPUTED,
                formula=f"close.rolling(window={window}).mean()",
                dependencies=["close"],
                lag=window,
                availability=AvailabilityType.END_OF_DAY,
                unit="USD"
            ))

        # ========== TECHNICAL INDICATORS ==========
        self.register(FeatureDefinition(
            name="rsi_14",
            description="Relative Strength Index (14-period)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="100 - (100 / (1 + avg_gain / avg_loss))",
            dependencies=["close"],
            lag=14,
            availability=AvailabilityType.END_OF_DAY,
            unit="index",
            range_min=0,
            range_max=100,
            notes="Momentum oscillator measuring overbought/oversold conditions"
        ))

        self.register(FeatureDefinition(
            name="macd",
            description="Moving Average Convergence Divergence line (12,26)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="EMA(close, 12) - EMA(close, 26)",
            dependencies=["close"],
            lag=26,
            availability=AvailabilityType.END_OF_DAY,
            unit="USD",
            notes="Trend-following momentum indicator"
        ))

        self.register(FeatureDefinition(
            name="macd_signal",
            description="MACD signal line (9-period EMA of MACD)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="EMA(macd, 9)",
            dependencies=["macd"],
            lag=35,  # 26 + 9
            availability=AvailabilityType.END_OF_DAY,
            unit="USD"
        ))

        self.register(FeatureDefinition(
            name="macd_hist",
            description="MACD histogram (MACD - Signal)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="macd - macd_signal",
            dependencies=["macd", "macd_signal"],
            lag=35,
            availability=AvailabilityType.END_OF_DAY,
            unit="USD"
        ))

        self.register(FeatureDefinition(
            name="bollinger_upper",
            description="Upper Bollinger Band (SMA + 2*std)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="SMA(close, 20) + 2 * STD(close, 20)",
            dependencies=["close"],
            lag=20,
            availability=AvailabilityType.END_OF_DAY,
            unit="USD"
        ))

        self.register(FeatureDefinition(
            name="bollinger_middle",
            description="Middle Bollinger Band (20-period SMA)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="SMA(close, 20)",
            dependencies=["close"],
            lag=20,
            availability=AvailabilityType.END_OF_DAY,
            unit="USD"
        ))

        self.register(FeatureDefinition(
            name="bollinger_lower",
            description="Lower Bollinger Band (SMA - 2*std)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="SMA(close, 20) - 2 * STD(close, 20)",
            dependencies=["close"],
            lag=20,
            availability=AvailabilityType.END_OF_DAY,
            unit="USD"
        ))

        self.register(FeatureDefinition(
            name="atr_14",
            description="Average True Range (14-period)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="SMA(max(high-low, |high-prev_close|, |low-prev_close|), 14)",
            dependencies=["high", "low", "close"],
            lag=14,
            availability=AvailabilityType.END_OF_DAY,
            unit="USD",
            notes="Volatility indicator measuring price range"
        ))

        self.register(FeatureDefinition(
            name="obv",
            description="On-Balance Volume",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="cumsum(volume * sign(close - prev_close))",
            dependencies=["volume", "close"],
            lag=1,
            availability=AvailabilityType.END_OF_DAY,
            unit="shares",
            notes="Volume-based momentum indicator"
        ))

        self.register(FeatureDefinition(
            name="stoch_k",
            description="Stochastic %K (14-period)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="100 * (close - low_14) / (high_14 - low_14)",
            dependencies=["close", "high", "low"],
            lag=14,
            availability=AvailabilityType.END_OF_DAY,
            unit="percent",
            range_min=0,
            range_max=100
        ))

        self.register(FeatureDefinition(
            name="stoch_d",
            description="Stochastic %D (3-period SMA of %K)",
            feature_type=FeatureType.TECHNICAL,
            data_source=DataSource.COMPUTED,
            formula="SMA(stoch_k, 3)",
            dependencies=["stoch_k"],
            lag=17,  # 14 + 3
            availability=AvailabilityType.END_OF_DAY,
            unit="percent",
            range_min=0,
            range_max=100
        ))

        # ========== TARGET FEATURES ==========
        self.register(FeatureDefinition(
            name="target_price",
            description="Future closing price (prediction target)",
            feature_type=FeatureType.TARGET,
            data_source=DataSource.COMPUTED,
            formula="close.shift(-forecast_horizon)",
            dependencies=["close"],
            lag=-1,  # Negative indicates future data
            availability=AvailabilityType.END_OF_DAY,
            unit="USD",
            notes="WARNING: This is a target variable - never use as input feature"
        ))

        self.register(FeatureDefinition(
            name="target_return",
            description="Future log return (prediction target)",
            feature_type=FeatureType.TARGET,
            data_source=DataSource.COMPUTED,
            formula="log_return.shift(-forecast_horizon)",
            dependencies=["log_return"],
            lag=-1,
            availability=AvailabilityType.END_OF_DAY,
            unit="log ratio",
            notes="WARNING: This is a target variable - never use as input feature"
        ))

        self.register(FeatureDefinition(
            name="target_direction",
            description="Future price direction (1=up, 0=down)",
            feature_type=FeatureType.TARGET,
            data_source=DataSource.COMPUTED,
            formula="(close.shift(-forecast_horizon) > close).astype(int)",
            dependencies=["close"],
            lag=-1,
            availability=AvailabilityType.END_OF_DAY,
            unit="binary",
            range_min=0,
            range_max=1,
            notes="WARNING: This is a target variable - never use as input feature"
        ))

        logger.info(f"Registered {len(self.features)} built-in features")

    def register(self, feature: FeatureDefinition) -> None:
        """
        Register a feature definition.

        Args:
            feature: Feature definition to register
        """
        if feature.name in self.features:
            logger.warning(f"Overwriting existing feature: {feature.name}")
        self.features[feature.name] = feature

    def get(self, name: str) -> Optional[FeatureDefinition]:
        """
        Get a feature definition by name.

        Args:
            name: Feature name

        Returns:
            FeatureDefinition or None if not found
        """
        return self.features.get(name)

    def list_features(
        self,
        feature_type: Optional[FeatureType] = None,
        data_source: Optional[DataSource] = None
    ) -> List[str]:
        """
        List features filtered by type or source.

        Args:
            feature_type: Filter by feature type
            data_source: Filter by data source

        Returns:
            List of feature names
        """
        result = []
        for name, feature in self.features.items():
            if feature_type and feature.feature_type != feature_type:
                continue
            if data_source and feature.data_source != data_source:
                continue
            result.append(name)
        return sorted(result)

    def get_dependencies(self, feature_name: str) -> Set[str]:
        """
        Get all dependencies (direct and transitive) for a feature.

        Args:
            feature_name: Feature name

        Returns:
            Set of dependency feature names
        """
        feature = self.features.get(feature_name)
        if not feature:
            return set()

        dependencies = set()
        to_process = list(feature.dependencies)

        while to_process:
            dep_name = to_process.pop()
            if dep_name in dependencies:
                continue
            dependencies.add(dep_name)

            dep_feature = self.features.get(dep_name)
            if dep_feature:
                to_process.extend(dep_feature.dependencies)

        return dependencies

    def get_minimum_lag(self, feature_name: str) -> int:
        """
        Get minimum lag required for a feature (considering all dependencies).

        Args:
            feature_name: Feature name

        Returns:
            Minimum lag in periods
        """
        feature = self.features.get(feature_name)
        if not feature:
            return 0

        # Get lag from all dependencies
        max_lag = feature.lag

        for dep_name in self.get_dependencies(feature_name):
            dep_feature = self.features.get(dep_name)
            if dep_feature:
                max_lag = max(max_lag, dep_feature.lag)

        return max_lag

    def validate_feature(self, feature_name: str) -> FeatureValidationResult:
        """
        Validate a feature definition.

        Args:
            feature_name: Feature name

        Returns:
            Validation result
        """
        feature = self.features.get(feature_name)
        if not feature:
            return FeatureValidationResult(
                feature_name=feature_name,
                is_valid=False,
                has_lookahead_risk=False,
                missing_dependencies=[],
                circular_dependencies=[],
                warnings=[],
                errors=[f"Feature '{feature_name}' not found in registry"]
            )

        warnings = []
        errors = []
        missing_deps = []
        circular_deps = []

        # Check dependencies exist
        for dep in feature.dependencies:
            if dep not in self.features:
                missing_deps.append(dep)

        # Check for circular dependencies
        visited = set()
        path = []

        def check_circular(name: str) -> bool:
            if name in path:
                circular_deps.append(" -> ".join(path[path.index(name):] + [name]))
                return True
            if name in visited:
                return False

            visited.add(name)
            path.append(name)

            dep_feature = self.features.get(name)
            if dep_feature:
                for dep in dep_feature.dependencies:
                    if check_circular(dep):
                        return True

            path.pop()
            return False

        check_circular(feature_name)

        # Check for lookahead risk
        has_lookahead_risk = feature.lag < 0 or any(
            self.features.get(dep) and self.features[dep].lag < 0
            for dep in self.get_dependencies(feature_name)
        )

        if has_lookahead_risk and feature.feature_type != FeatureType.TARGET:
            warnings.append(f"Feature has negative lag (lookahead risk)")

        # Validate feature type consistency
        if feature.feature_type == FeatureType.TARGET:
            if feature.lag >= 0:
                warnings.append("Target feature should have negative lag")

        is_valid = len(errors) == 0 and len(missing_deps) == 0 and len(circular_deps) == 0

        return FeatureValidationResult(
            feature_name=feature_name,
            is_valid=is_valid,
            has_lookahead_risk=has_lookahead_risk,
            missing_dependencies=missing_deps,
            circular_dependencies=circular_deps,
            warnings=warnings,
            errors=errors
        )

    def validate_all(self) -> Dict[str, FeatureValidationResult]:
        """
        Validate all registered features.

        Returns:
            Dictionary of validation results
        """
        results = {}
        for name in self.features:
            results[name] = self.validate_feature(name)

        # Log summary
        valid_count = sum(1 for r in results.values() if r.is_valid)
        logger.info(f"Validated {len(results)} features: {valid_count} valid, {len(results) - valid_count} invalid")

        return results

    def get_feature_hash(self, feature_names: List[str]) -> str:
        """
        Generate hash for a set of features (for versioning).

        Args:
            feature_names: List of feature names

        Returns:
            Hash string
        """
        feature_defs = []
        for name in sorted(feature_names):
            feature = self.features.get(name)
            if feature:
                feature_defs.append(feature.to_dict())

        content = json.dumps(feature_defs, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_point_in_time(
        self,
        feature_names: List[str],
        reference_time: datetime,
        market_close_time: dt_time = dt_time(16, 0)
    ) -> Dict[str, bool]:
        """
        Check if features would be available at a given point in time.

        Args:
            feature_names: Features to check
            reference_time: Reference datetime
            market_close_time: Market close time

        Returns:
            Dictionary of feature name -> availability
        """
        availability = {}
        ref_time = reference_time.time()

        for name in feature_names:
            feature = self.features.get(name)
            if not feature:
                availability[name] = False
                continue

            if feature.availability == AvailabilityType.REALTIME:
                availability[name] = True
            elif feature.availability == AvailabilityType.END_OF_DAY:
                # Check if we're past market close
                avail_time = dt_time.fromisoformat(feature.availability_time) if feature.availability_time else market_close_time
                availability[name] = ref_time >= avail_time
            elif feature.availability == AvailabilityType.NEXT_DAY:
                availability[name] = False  # Conservative: assume not available same day
            else:
                availability[name] = True  # Default to available

        return availability

    def get_safe_features(
        self,
        exclude_targets: bool = True,
        min_lag: int = 0
    ) -> List[str]:
        """
        Get list of features safe to use as model inputs.

        Args:
            exclude_targets: Whether to exclude target features
            min_lag: Minimum required lag

        Returns:
            List of safe feature names
        """
        safe_features = []

        for name, feature in self.features.items():
            # Skip targets
            if exclude_targets and feature.feature_type == FeatureType.TARGET:
                continue

            # Skip features with lookahead risk
            if feature.lag < min_lag:
                continue

            safe_features.append(name)

        return sorted(safe_features)

    def save_to_yaml(self, path: Path) -> None:
        """
        Save registry to YAML file.

        Args:
            path: Output path
        """
        data = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'features': {}
        }

        for name, feature in self.features.items():
            data['features'][name] = feature.to_dict()

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved {len(self.features)} features to {path}")

    def load_from_yaml(self, path: Path) -> None:
        """
        Load additional features from YAML file.

        Args:
            path: YAML file path
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if 'features' not in data:
            logger.warning(f"No features found in {path}")
            return

        count = 0
        for name, feature_dict in data['features'].items():
            try:
                feature = FeatureDefinition.from_dict(feature_dict)
                self.register(feature)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load feature '{name}': {e}")

        logger.info(f"Loaded {count} features from {path}")

    def generate_documentation(self) -> str:
        """
        Generate markdown documentation for all features.

        Returns:
            Markdown string
        """
        lines = [
            "# Feature Registry Documentation",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total features: {len(self.features)}",
            "",
        ]

        # Group by type
        by_type: Dict[FeatureType, List[FeatureDefinition]] = {}
        for feature in self.features.values():
            if feature.feature_type not in by_type:
                by_type[feature.feature_type] = []
            by_type[feature.feature_type].append(feature)

        for feature_type in FeatureType:
            if feature_type not in by_type:
                continue

            features = by_type[feature_type]
            lines.append(f"## {feature_type.value.title()} Features ({len(features)})")
            lines.append("")
            lines.append("| Name | Description | Formula | Lag | Source |")
            lines.append("|------|-------------|---------|-----|--------|")

            for f in sorted(features, key=lambda x: x.name):
                formula = f.formula or "-"
                if len(formula) > 40:
                    formula = formula[:37] + "..."
                lines.append(
                    f"| `{f.name}` | {f.description[:50]}{'...' if len(f.description) > 50 else ''} "
                    f"| {formula} | {f.lag} | {f.data_source.value} |"
                )
            lines.append("")

        return "\n".join(lines)


# Singleton instance
_registry: Optional[FeatureRegistry] = None


def get_feature_registry(config_path: Optional[Path] = None) -> FeatureRegistry:
    """
    Get the global feature registry instance.

    Args:
        config_path: Optional path to features.yaml

    Returns:
        FeatureRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = FeatureRegistry(config_path)
    return _registry


def validate_feature_set(
    feature_names: List[str],
    registry: Optional[FeatureRegistry] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a set of features for use in a model.

    Args:
        feature_names: List of feature names to validate
        registry: Optional registry instance

    Returns:
        Tuple of (is_valid, list of issues)
    """
    reg = registry or get_feature_registry()
    issues = []

    for name in feature_names:
        result = reg.validate_feature(name)
        if not result.is_valid:
            issues.extend(result.errors)
        if result.has_lookahead_risk:
            issues.append(f"Feature '{name}' has lookahead bias risk")
        issues.extend(result.warnings)

    return len(issues) == 0, issues
