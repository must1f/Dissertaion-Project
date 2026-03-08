"""
Calendar-aware alignment utilities.

Uses a master trading calendar to align multiple assets and avoid leakage from
improper forward-filling across non-trading days.
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd


def build_master_calendar(start_date: str, end_date: str, calendar: str = "NYSE", holidays: Optional[List[str]] = None) -> pd.DatetimeIndex:
    """
    Build a master trading calendar. Falls back to business-day frequency
    if an exchange-specific calendar is not available in the environment.
    """

    holidays = holidays or []
    # Simple fallback: business days excluding provided holidays
    cal = pd.bdate_range(start=start_date, end=end_date, freq="C", holidays=holidays)
    return cal


def align_to_calendar(
    df: pd.DataFrame,
    master_calendar: pd.DatetimeIndex,
    default_forward_fill_limit: int = 1,
    per_symbol_forward_fill_limit: Optional[Dict[str, int]] = None,
    forward_fill_limit: Optional[int] = None,
    return_report: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Align asset-level data to the master calendar with bounded forward-fill.

    Args:
        df: DataFrame with columns ['time', 'ticker', ...]
        master_calendar: DatetimeIndex of trading days
        forward_fill_limit: Maximum consecutive fills to allow per asset
        return_report: Whether to return (aligned_df, report)
    """

    aligned_frames = []
    per_ticker_report: Dict[str, Dict[str, int]] = {}

    numeric_columns: List[str] = [str(c) for c in df.columns if c not in {"ticker", "time"}]
    per_symbol_forward_fill_limit = per_symbol_forward_fill_limit or {}
    if forward_fill_limit is not None:
        default_forward_fill_limit = forward_fill_limit

    for ticker, grp in df.groupby("ticker"):
        tdf = grp.copy()
        tdf = tdf.set_index("time").reindex(master_calendar)
        missing_mask = tdf.isna().all(axis=1)
        added_rows = int(pd.Series(missing_mask).sum())
        ticker_str = str(ticker)
        tdf["ticker"] = ticker_str
        tdf.index.name = "time"

        limit = per_symbol_forward_fill_limit.get(ticker_str, default_forward_fill_limit)

        filled = tdf.copy()
        filled_counts = 0
        for col in numeric_columns:
            col_name = str(col)
            series = filled[col_name]
            before_na = int(series.isna().sum())
            if limit > 0:
                filled[col_name] = series.ffill(limit=limit)
            after_na = int(filled[col_name].isna().sum())
            filled_counts += max(before_na - after_na, 0)

        rem_mask = filled[numeric_columns].isna().any(axis=1)
        remaining_na = int(pd.Series(rem_mask).sum())

        per_ticker_report[ticker_str] = {
            "added_rows": added_rows,
            "remaining_na_rows": remaining_na,
            "forward_filled_cells": int(filled_counts),
            "forward_fill_limit": limit,
        }

        aligned_frames.append(filled.reset_index())

    aligned = pd.concat(aligned_frames, ignore_index=True)

    if return_report:
        return aligned, {
            "per_ticker": per_ticker_report,
            "calendar_length": int(len(master_calendar)),
            "default_forward_fill_limit": default_forward_fill_limit,
        }
    return aligned
