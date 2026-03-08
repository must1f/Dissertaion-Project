"""
Leakage-safe scaling helpers with serialization.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from sklearn.preprocessing import StandardScaler
except ImportError:  # fallback
    class StandardScaler:  # type: ignore
        def fit(self, X):
            X = np.asarray(X)
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0) + 1e-8
            return self

        def transform(self, X):
            X = np.asarray(X)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def save_scaler(
    scaler: StandardScaler,
    path: Path,
    feature_cols: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": np.asarray(getattr(scaler, "mean_", [])).tolist(),
        "scale": np.asarray(getattr(scaler, "scale_", [])).tolist(),
        "feature_cols": feature_cols,
        "metadata": metadata or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_scaler(path: Path) -> StandardScaler:
    with open(path, "r") as f:
        payload = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(payload.get("mean", []))
    scaler.scale_ = np.array(payload.get("scale", []))
    return scaler
