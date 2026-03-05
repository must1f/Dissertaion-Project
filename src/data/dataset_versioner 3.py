"""
Dataset Versioning System

Provides hash-based versioning for reproducible datasets:
- Computes deterministic hash from data + config
- Stores version metadata in SQLite
- Validates versions on experiment load
- Tracks data lineage and transformations

Critical for research reproducibility - ensures experiments
can be exactly reproduced with the same data.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetVersion:
    """A versioned dataset snapshot"""
    version_id: str
    name: str
    created_at: str

    # Data properties
    n_samples: int
    n_features: int
    feature_names: List[str]
    date_range: Tuple[str, str]
    tickers: List[str]

    # Hash components
    data_hash: str
    config_hash: str
    feature_config_hash: str

    # Split information
    train_size: int
    val_size: int
    test_size: int
    split_seed: int

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        # Handle tuple conversion
        if 'date_range' in data and isinstance(data['date_range'], list):
            data['date_range'] = tuple(data['date_range'])
        return cls(**data)


@dataclass
class TransformationRecord:
    """Record of a data transformation"""
    transform_id: str
    version_id: str
    transform_type: str
    parameters: Dict[str, Any]
    applied_at: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]


class DatasetVersioner:
    """
    Dataset versioning and lineage tracking.

    Usage:
        versioner = DatasetVersioner(db_path='data/versions.db')

        # Create version
        version = versioner.create_version(
            data=df,
            name='stock_data_v1',
            feature_config=feature_config,
            split_config=split_config
        )

        # Validate on load
        if versioner.validate_version(data, version.version_id):
            print("Data matches version")

        # Get version info
        info = versioner.get_version(version.version_id)
    """

    def __init__(self, db_path: Union[str, Path] = 'data/dataset_versions.db'):
        """
        Initialize versioner.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                n_samples INTEGER,
                n_features INTEGER,
                feature_names TEXT,
                date_range TEXT,
                tickers TEXT,
                data_hash TEXT NOT NULL,
                config_hash TEXT,
                feature_config_hash TEXT,
                train_size INTEGER,
                val_size INTEGER,
                test_size INTEGER,
                split_seed INTEGER,
                description TEXT,
                tags TEXT,
                parent_version TEXT
            )
        ''')

        # Transformations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transformations (
                transform_id TEXT PRIMARY KEY,
                version_id TEXT,
                transform_type TEXT NOT NULL,
                parameters TEXT,
                applied_at TEXT NOT NULL,
                input_shape TEXT,
                output_shape TEXT,
                FOREIGN KEY (version_id) REFERENCES versions(version_id)
            )
        ''')

        # Experiments table (links experiments to data versions)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_data (
                experiment_id TEXT,
                version_id TEXT,
                used_at TEXT,
                PRIMARY KEY (experiment_id, version_id),
                FOREIGN KEY (version_id) REFERENCES versions(version_id)
            )
        ''')

        conn.commit()
        conn.close()

    def compute_data_hash(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        include_index: bool = False
    ) -> str:
        """
        Compute deterministic hash of data.

        Args:
            data: DataFrame or array to hash
            include_index: Whether to include index in hash

        Returns:
            SHA256 hash string (first 16 chars)
        """
        if isinstance(data, pd.DataFrame):
            # Sort columns for consistency
            data = data.sort_index(axis=1)

            if include_index:
                data_bytes = pd.util.hash_pandas_object(data).values.tobytes()
            else:
                data_bytes = pd.util.hash_pandas_object(data, index=False).values.tobytes()
        else:
            data_bytes = np.ascontiguousarray(data).tobytes()

        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def compute_version_id(
        self,
        data_hash: str,
        config_hash: str,
        feature_hash: str
    ) -> str:
        """
        Compute unique version ID from component hashes.

        Args:
            data_hash: Hash of raw data
            config_hash: Hash of processing config
            feature_hash: Hash of feature config

        Returns:
            Version ID (12 chars)
        """
        combined = f"{data_hash}_{config_hash}_{feature_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:12]

    def create_version(
        self,
        data: pd.DataFrame,
        name: str,
        feature_config: Optional[Dict[str, Any]] = None,
        split_config: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        parent_version: Optional[str] = None
    ) -> DatasetVersion:
        """
        Create a new dataset version.

        Args:
            data: DataFrame to version
            name: Version name
            feature_config: Feature engineering configuration
            split_config: Train/val/test split configuration
            description: Version description
            tags: Optional tags
            parent_version: Parent version ID (for lineage)

        Returns:
            DatasetVersion object
        """
        # Compute hashes
        data_hash = self.compute_data_hash(data)
        config_hash = self.compute_config_hash(split_config or {})
        feature_hash = self.compute_config_hash(feature_config or {})
        version_id = self.compute_version_id(data_hash, config_hash, feature_hash)

        # Extract metadata
        feature_names = list(data.columns)
        n_samples, n_features = data.shape

        # Date range
        if 'time' in data.columns:
            date_range = (str(data['time'].min()), str(data['time'].max()))
        elif 'date' in data.columns:
            date_range = (str(data['date'].min()), str(data['date'].max()))
        else:
            date_range = ("", "")

        # Tickers
        if 'ticker' in data.columns:
            tickers = sorted(data['ticker'].unique().tolist())
        else:
            tickers = []

        # Split sizes
        split_config = split_config or {}
        train_ratio = split_config.get('train_ratio', 0.7)
        val_ratio = split_config.get('val_ratio', 0.15)
        test_ratio = split_config.get('test_ratio', 0.15)

        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        test_size = n_samples - train_size - val_size

        version = DatasetVersion(
            version_id=version_id,
            name=name,
            created_at=datetime.now().isoformat(),
            n_samples=n_samples,
            n_features=n_features,
            feature_names=feature_names,
            date_range=date_range,
            tickers=tickers,
            data_hash=data_hash,
            config_hash=config_hash,
            feature_config_hash=feature_hash,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            split_seed=split_config.get('seed', 42),
            description=description,
            tags=tags or [],
            parent_version=parent_version
        )

        # Save to database
        self._save_version(version)

        logger.info(f"Created dataset version: {version_id}")
        logger.info(f"  Name: {name}")
        logger.info(f"  Samples: {n_samples}, Features: {n_features}")
        logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")

        return version

    def _save_version(self, version: DatasetVersion):
        """Save version to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO versions
            (version_id, name, created_at, n_samples, n_features, feature_names,
             date_range, tickers, data_hash, config_hash, feature_config_hash,
             train_size, val_size, test_size, split_seed, description, tags, parent_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            version.version_id,
            version.name,
            version.created_at,
            version.n_samples,
            version.n_features,
            json.dumps(version.feature_names),
            json.dumps(version.date_range),
            json.dumps(version.tickers),
            version.data_hash,
            version.config_hash,
            version.feature_config_hash,
            version.train_size,
            version.val_size,
            version.test_size,
            version.split_seed,
            version.description,
            json.dumps(version.tags),
            version.parent_version
        ))

        conn.commit()
        conn.close()

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get version by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM versions WHERE version_id = ?', (version_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_version(row)

    def _row_to_version(self, row: tuple) -> DatasetVersion:
        """Convert database row to DatasetVersion"""
        return DatasetVersion(
            version_id=row[0],
            name=row[1],
            created_at=row[2],
            n_samples=row[3],
            n_features=row[4],
            feature_names=json.loads(row[5]),
            date_range=tuple(json.loads(row[6])),
            tickers=json.loads(row[7]),
            data_hash=row[8],
            config_hash=row[9],
            feature_config_hash=row[10],
            train_size=row[11],
            val_size=row[12],
            test_size=row[13],
            split_seed=row[14],
            description=row[15] or "",
            tags=json.loads(row[16]) if row[16] else [],
            parent_version=row[17]
        )

    def validate_version(
        self,
        data: pd.DataFrame,
        version_id: str,
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate that data matches a version.

        Args:
            data: DataFrame to validate
            version_id: Version ID to check against
            strict: If True, require exact hash match

        Returns:
            Tuple of (is_valid, list of issues)
        """
        version = self.get_version(version_id)
        if version is None:
            return False, [f"Version {version_id} not found"]

        issues = []

        # Check data hash
        current_hash = self.compute_data_hash(data)
        if current_hash != version.data_hash:
            issues.append(f"Data hash mismatch: {current_hash} != {version.data_hash}")

        # Check shape
        if data.shape[0] != version.n_samples:
            issues.append(f"Sample count mismatch: {data.shape[0]} != {version.n_samples}")

        if data.shape[1] != version.n_features:
            issues.append(f"Feature count mismatch: {data.shape[1]} != {version.n_features}")

        # Check columns
        current_cols = set(data.columns)
        version_cols = set(version.feature_names)
        if current_cols != version_cols:
            missing = version_cols - current_cols
            extra = current_cols - version_cols
            if missing:
                issues.append(f"Missing columns: {missing}")
            if extra:
                issues.append(f"Extra columns: {extra}")

        is_valid = len(issues) == 0 if strict else (
            current_hash == version.data_hash
        )

        if is_valid:
            logger.info(f"Data validated against version {version_id}")
        else:
            logger.warning(f"Data validation failed for version {version_id}")
            for issue in issues:
                logger.warning(f"  - {issue}")

        return is_valid, issues

    def list_versions(
        self,
        name_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[DatasetVersion]:
        """
        List available versions.

        Args:
            name_filter: Filter by name (contains)
            tag_filter: Filter by tag
            limit: Maximum number of results

        Returns:
            List of DatasetVersion objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM versions WHERE 1=1"
        params = []

        if name_filter:
            query += " AND name LIKE ?"
            params.append(f"%{name_filter}%")

        if tag_filter:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag_filter}"%')

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_version(row) for row in rows]

    def record_transformation(
        self,
        version_id: str,
        transform_type: str,
        parameters: Dict[str, Any],
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...]
    ) -> str:
        """
        Record a transformation applied to a version.

        Args:
            version_id: Source version
            transform_type: Type of transformation
            parameters: Transformation parameters
            input_shape: Shape before transformation
            output_shape: Shape after transformation

        Returns:
            Transform ID
        """
        transform_id = hashlib.sha256(
            f"{version_id}_{transform_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO transformations
            (transform_id, version_id, transform_type, parameters, applied_at, input_shape, output_shape)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            transform_id,
            version_id,
            transform_type,
            json.dumps(parameters),
            datetime.now().isoformat(),
            json.dumps(input_shape),
            json.dumps(output_shape)
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Recorded transformation {transform_id} for version {version_id}")

        return transform_id

    def get_transformations(self, version_id: str) -> List[TransformationRecord]:
        """Get all transformations for a version"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM transformations WHERE version_id = ? ORDER BY applied_at',
            (version_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            TransformationRecord(
                transform_id=row[0],
                version_id=row[1],
                transform_type=row[2],
                parameters=json.loads(row[3]) if row[3] else {},
                applied_at=row[4],
                input_shape=tuple(json.loads(row[5])) if row[5] else (),
                output_shape=tuple(json.loads(row[6])) if row[6] else ()
            )
            for row in rows
        ]

    def link_experiment(self, experiment_id: str, version_id: str):
        """Link an experiment to a data version"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO experiment_data (experiment_id, version_id, used_at)
            VALUES (?, ?, ?)
        ''', (experiment_id, version_id, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_lineage(self, version_id: str) -> List[DatasetVersion]:
        """
        Get version lineage (parent chain).

        Args:
            version_id: Starting version

        Returns:
            List of versions from oldest to newest
        """
        lineage = []
        current_id = version_id

        while current_id:
            version = self.get_version(current_id)
            if version is None:
                break
            lineage.append(version)
            current_id = version.parent_version

        return list(reversed(lineage))

    def export_manifest(self, version_id: str, output_path: Path) -> Path:
        """
        Export version manifest to JSON.

        Args:
            version_id: Version to export
            output_path: Output file path

        Returns:
            Path to manifest file
        """
        version = self.get_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        transforms = self.get_transformations(version_id)
        lineage = self.get_lineage(version_id)

        manifest = {
            'version': version.to_dict(),
            'transformations': [
                {
                    'transform_id': t.transform_id,
                    'type': t.transform_type,
                    'parameters': t.parameters,
                    'applied_at': t.applied_at
                }
                for t in transforms
            ],
            'lineage': [v.version_id for v in lineage],
            'exported_at': datetime.now().isoformat()
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Exported manifest to {output_path}")

        return output_path


# Convenience functions

def version_dataset(
    data: pd.DataFrame,
    name: str,
    db_path: str = 'data/dataset_versions.db',
    **kwargs
) -> DatasetVersion:
    """
    Quick versioning of a dataset.

    Args:
        data: DataFrame to version
        name: Version name
        db_path: Database path
        **kwargs: Additional arguments for create_version

    Returns:
        DatasetVersion
    """
    versioner = DatasetVersioner(db_path=db_path)
    return versioner.create_version(data, name, **kwargs)


def validate_data_version(
    data: pd.DataFrame,
    version_id: str,
    db_path: str = 'data/dataset_versions.db'
) -> bool:
    """
    Quick validation of data against version.

    Args:
        data: DataFrame to validate
        version_id: Version to check
        db_path: Database path

    Returns:
        True if valid
    """
    versioner = DatasetVersioner(db_path=db_path)
    is_valid, _ = versioner.validate_version(data, version_id)
    return is_valid
