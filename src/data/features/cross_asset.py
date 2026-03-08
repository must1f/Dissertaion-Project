import numpy as np
import pandas as pd
from typing import List, Tuple


def add_cross_asset_spreads(df: pd.DataFrame, pairs: List[Tuple[str, str]], price_col: str = "adjusted_close") -> pd.DataFrame:
    """Add cross-asset log-price spreads merged on time for all tickers."""
    df = df.copy()
    pivot = df.pivot_table(index="time", columns="ticker", values=price_col)
    spread_cols = {}
    for a, b in pairs:
        if a in pivot.columns and b in pivot.columns:
            spread_cols[f"cross_{a.lower()}_{b.lower()}_spread"] = np.log(pivot[a] / pivot[b])

    spread_df = pd.DataFrame(spread_cols)
    spread_df = spread_df.reset_index()

    return df.merge(spread_df, on="time", how="left")
