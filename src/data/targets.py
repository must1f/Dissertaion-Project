"""
Target construction utilities.
"""

import numpy as np
import pandas as pd


def add_next_day_log_return(df: pd.DataFrame, price_col: str = "adjusted_close", target_col: str = "target", group_col: str = "ticker") -> pd.DataFrame:
    """Compute next-day log return target using adjusted prices."""
    df = df.sort_values([group_col, "time"]).copy()
    df[target_col] = df.groupby(group_col)[price_col].transform(lambda x: np.log(x.shift(-1) / x))
    return df


def add_realized_vol(df: pd.DataFrame, price_col: str = "adjusted_close", window: int = 5, target_col: str = "realized_vol", group_col: str = "ticker") -> pd.DataFrame:
    """Compute forward-looking realized volatility over a window based on adjusted log returns."""
    df = df.sort_values([group_col, "time"]).copy()
    log_ret = df.groupby(group_col)[price_col].transform(lambda x: np.log(x / x.shift(1)))
    df[target_col] = (
        log_ret.rolling(window=window)
        .apply(lambda x: np.sqrt(np.sum(np.square(x))), raw=True)
        .shift(-window + 1)
    )
    return df


def add_joint_return_and_vol(
    df: pd.DataFrame,
    price_col: str = "adjusted_close",
    return_col: str = "target_return",
    vol_col: str = "target_vol",
    vol_window: int = 5,
    group_col: str = "ticker",
) -> pd.DataFrame:
    """Add both next-day return and forward realized vol targets."""
    df = add_next_day_log_return(df, price_col=price_col, target_col=return_col, group_col=group_col)
    df = add_realized_vol(df, price_col=price_col, window=vol_window, target_col=vol_col, group_col=group_col)
    return df
