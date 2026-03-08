import pandas as pd
from typing import List


def add_rolling_volatility(df: pd.DataFrame, return_col: str = "adj_return_1d", windows: List[int] | None = None, group_col: str = "ticker") -> pd.DataFrame:
    windows = windows or [10, 20]
    df = df.sort_values([group_col, "time"]).copy()
    for w in windows:
        df[f"rolling_vol_{w}"] = df.groupby(group_col)[return_col].transform(lambda x: x.rolling(w).std())
    return df
