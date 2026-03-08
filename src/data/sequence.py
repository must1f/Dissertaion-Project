"""
Sequence dataset builder that enforces identical windowing across models.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_sequences(df: pd.DataFrame, feature_cols: List[str], target_col: str, sequence_length: int, horizon: int = 1, group_col: str = "ticker") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create fixed-length sequences for all assets with the same lookback and forecast horizon.
    """

    X_list, y_list, tickers = [], [], []

    for ticker, grp in df.groupby(group_col):
        g = grp.sort_values("time")
        feat = g[feature_cols].values
        target = g[target_col].values
        n = len(g)
        for i in range(n - sequence_length - horizon + 1):
            X_list.append(feat[i : i + sequence_length])
            y_list.append(target[i + sequence_length + horizon - 1])
            tickers.append(ticker)

    return np.array(X_list), np.array(y_list), tickers
