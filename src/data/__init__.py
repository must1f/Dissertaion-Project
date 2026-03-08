"""Data pipeline modules"""

from .fetcher import DataFetcher
from .preprocessor import DataPreprocessor
from .dataset import FinancialDataset, create_dataloaders
from .pipeline import build_benchmark_dataset
from .universe import UniverseDefinition, default_universe, universe_from_config
from .cache import CacheManager, CachePaths
from .calendar import build_master_calendar, align_to_calendar
from .quality import run_qa
from .targets import add_next_day_log_return, add_realized_vol
from .split import temporal_split_with_gap
from .scaling import fit_scaler, apply_scaler, save_scaler, load_scaler
from .sequence import build_sequences
from .leakage_auditor import (
    LeakageAuditor,
    LeakageWarning,
    LeakageAuditResult,
    LeakageSeverity,
    audit_train_test_split,
    verify_no_future_leakage
)
from .dataset_versioner import (
    DatasetVersioner,
    DatasetVersion,
    TransformationRecord,
    version_dataset,
    validate_data_version
)
from .feature_registry import (
    FeatureRegistry,
    FeatureDefinition,
    FeatureType,
    DataSource,
    AvailabilityType,
    FeatureValidationResult,
    get_feature_registry,
    validate_feature_set
)
from .data_cleaner import (
    DataCleaner,
    ImputationMethod,
    OutlierMethod,
    OutlierTreatment,
    CleaningRecord,
    DataQualityReport,
    CleaningResult,
    clean_financial_data,
    validate_cleaned_data
)

__all__ = [
    "DataFetcher",
    "DataPreprocessor",
    "FinancialDataset",
    "create_dataloaders",
    "build_benchmark_dataset",
    "UniverseDefinition",
    "default_universe",
    "universe_from_config",
    "CacheManager",
    "CachePaths",
    "build_master_calendar",
    "align_to_calendar",
    "run_qa",
    "add_next_day_log_return",
    "add_realized_vol",
    "temporal_split_with_gap",
    "fit_scaler",
    "apply_scaler",
    "save_scaler",
    "load_scaler",
    "build_sequences",
    # Leakage auditing
    "LeakageAuditor",
    "LeakageWarning",
    "LeakageAuditResult",
    "LeakageSeverity",
    "audit_train_test_split",
    "verify_no_future_leakage",
    # Dataset versioning
    "DatasetVersioner",
    "DatasetVersion",
    "TransformationRecord",
    "version_dataset",
    "validate_data_version",
    # Feature registry
    "FeatureRegistry",
    "FeatureDefinition",
    "FeatureType",
    "DataSource",
    "AvailabilityType",
    "FeatureValidationResult",
    "get_feature_registry",
    "validate_feature_set",
    # Data cleaning
    "DataCleaner",
    "ImputationMethod",
    "OutlierMethod",
    "OutlierTreatment",
    "CleaningRecord",
    "DataQualityReport",
    "CleaningResult",
    "clean_financial_data",
    "validate_cleaned_data",
]
