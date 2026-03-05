"""Data pipeline modules"""

from .fetcher import DataFetcher
from .preprocessor import DataPreprocessor
from .dataset import FinancialDataset, create_dataloaders
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
