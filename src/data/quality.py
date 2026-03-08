"""
Quality assurance checks for financial time series.
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def _coverage(df: pd.DataFrame) -> Dict[str, Any]:
    by_ticker = (
        df.groupby("ticker")
        .agg(rows=("ticker", "count"), start=("time", "min"), end=("time", "max"))
        .reset_index()
        .to_dict(orient="records")
        if len(df)
        else []
    )
    return {
        "rows": int(len(df)),
        "tickers": df["ticker"].nunique(),
        "start": str(df["time"].min()) if not df.empty else None,
        "end": str(df["time"].max()) if not df.empty else None,
        "by_ticker": by_ticker,
    }


def _duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    dup_mask = df.duplicated(subset=["ticker", "time"], keep=False)
    dup_rows = df.loc[dup_mask]
    return {"count": int(len(dup_rows)), "examples": dup_rows.head(5).to_dict(orient="records") if len(dup_rows) else []}


def _bad_prices(df: pd.DataFrame) -> Dict[str, Any]:
    zero_or_negative = (df[[c for c in df.columns if c in {"open", "high", "low", "close", "adjusted_close"}]] <= 0).any(axis=1)
    bad = df.loc[zero_or_negative]
    return {"count": int(len(bad)), "examples": bad.head(5).to_dict(orient="records") if len(bad) else []}


def _extreme_jumps(df: pd.DataFrame, threshold: float = 0.2) -> Dict[str, Any]:
    # Detect jumps greater than threshold on adjusted_close if available otherwise close
    price_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    df = df.sort_values(["ticker", "time"]).copy()
    df["log_return_tmp"] = df.groupby("ticker")[price_col].transform(lambda x: np.log(x / x.shift(1)))
    jumps = df[df["log_return_tmp"].abs() > threshold]
    return {
        "count": int(len(jumps)),
        "examples": jumps.head(5).to_dict(orient="records") if len(jumps) else [],
        "threshold": threshold,
    }


def _missing_dates(df: pd.DataFrame, expected_calendar: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    if expected_calendar is None:
        return {"calendar": None, "per_ticker_missing": {}}

    missing_info: Dict[str, Any] = {}
    expected_dates = set(pd.to_datetime(expected_calendar).date)

    for ticker, grp in df.groupby("ticker"):
        present = pd.to_datetime(grp["time"]).dropna().unique()
        present_dates = set(pd.to_datetime(present).date)
        missing = expected_dates - present_dates
        missing_info[ticker] = {
            "missing_count": len(missing),
            "missing_sample": list(sorted(missing))[:5],
        }

    return {"calendar": {
        "start": str(expected_calendar.min()) if len(expected_calendar) else None,
        "end": str(expected_calendar.max()) if len(expected_calendar) else None,
        "length": int(len(expected_calendar)),
    }, "per_ticker_missing": missing_info}


def _corporate_action_flags(df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
    """Flag potential corporate actions via adjusted/close ratio jumps."""
    if "adjusted_close" not in df.columns or "close" not in df.columns:
        return {"count": 0, "examples": []}

    df = df.sort_values(["ticker", "time"]).copy()
    ratio = df["adjusted_close"] / df["close"].replace(0, np.nan)
    df["adj_close_ratio"] = ratio
    df["ratio_change"] = df.groupby("ticker")["adj_close_ratio"].transform(lambda x: x.pct_change())
    jumps = df[df["ratio_change"].abs() > threshold]
    return {
        "count": int(len(jumps)),
        "examples": jumps.head(5).to_dict(orient="records") if len(jumps) else [],
        "threshold": threshold,
    }


def _post_alignment_missing(df: pd.DataFrame, expected_calendar: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    if expected_calendar is None:
        return {}
    report: Dict[str, Dict[str, int]] = {}
    expected_dates = set(pd.to_datetime(expected_calendar).date)
    for ticker, grp in df.groupby("ticker"):
        present = pd.to_datetime(grp["time"]).dropna().unique()
        present_dates = set(pd.to_datetime(present).date)
        missing = expected_dates - present_dates
        report[ticker] = {
            "missing_after_alignment": len(missing)
        }
    return report


def _missingness(df: pd.DataFrame) -> Dict[str, Any]:
    missing = df.isna().mean().to_dict()
    return {"missing_fraction": missing}


def run_qa(df: pd.DataFrame, expected_calendar: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
    """Run core QA checks and return a JSON-serializable report."""
    if df.empty:
        return {"error": "empty dataframe"}

    coverage = _coverage(df)
    missing_dates = _missing_dates(df, expected_calendar)
    extreme = _extreme_jumps(df)
    corp = _corporate_action_flags(df)

    report = {
        "coverage": coverage,
        "duplicates": _duplicates(df),
        "bad_prices": _bad_prices(df),
        "extreme_jumps": extreme,
        "missingness": _missingness(df),
        "missing_dates": missing_dates,
        "corporate_action_flags": corp,
    }

    if expected_calendar is not None:
        report["post_alignment_missing"] = _post_alignment_missing(df, expected_calendar)

    report["summary"] = {
        "rows": coverage.get("rows"),
        "tickers": coverage.get("tickers"),
        "extreme_jump_threshold": extreme.get("threshold"),
        "corporate_action_threshold": corp.get("threshold"),
        "post_alignment_missing_by_ticker": report.get("post_alignment_missing", {}),
    }

    return report
