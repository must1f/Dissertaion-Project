import pandas as pd
from typing import List


def add_momentum_trend(df: pd.DataFrame, return_col: str = "adj_return_1d", windows: List[int] | None = None, group_col: str = "ticker") -> pd.DataFrame:
    windows = windows or [20, 60]
    df = df.sort_values([group_col, "time"]).copy()
    for w in windows:
        roll_mean = df.groupby(group_col)[return_col].transform(lambda x: x.rolling(w).mean())
        roll_std = df.groupby(group_col)[return_col].transform(lambda x: x.rolling(w).std())
        df[f"momentum_{w}_z"] = roll_mean / (roll_std + 1e-8)
    return df
