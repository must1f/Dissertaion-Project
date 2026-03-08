import numpy as np
import pandas as pd
from typing import List


def add_adjusted_returns(df: pd.DataFrame, price_col: str = "adjusted_close", horizons: List[int] | None = None, group_col: str = "ticker") -> pd.DataFrame:
    """Add log returns over given horizons using adjusted prices."""
    horizons = horizons or [1, 5]
    df = df.sort_values([group_col, "time"]).copy()
    for h in horizons:
        df[f"adj_return_{h}d"] = df.groupby(group_col)[price_col].transform(lambda x: np.log(x / x.shift(h)))
    return df
