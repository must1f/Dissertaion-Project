import pandas as pd


def add_regime_markers(df: pd.DataFrame, vol_col: str = "rolling_vol_20", threshold: float = 0.02) -> pd.DataFrame:
    """Simple regime marker based on rolling volatility level."""
    df = df.copy()
    if vol_col in df.columns:
        df["regime_high_vol"] = (df[vol_col] > threshold).astype(int)
    return df
