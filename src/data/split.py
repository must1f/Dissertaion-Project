"""
Temporal split utilities with optional gap to prevent leakage.
"""

import pandas as pd
from typing import Tuple


def temporal_split_with_gap(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, gap: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train+val+test ratios must sum to 1.0")

    df = df.sort_values("time")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end + gap : val_end]
    test = df.iloc[val_end + gap :]
    return train, val, test
