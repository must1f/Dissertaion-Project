#!/usr/bin/env python3
"""
Macro Feature Merger

Merges macroeconomic indicators (from CSVs in data/macro/) with price data
for downstream model training. Produces a unified CSV with forward-filled
macro columns aligned on date.
"""

import pandas as pd
from pathlib import Path

PRICE_PATH = Path("data/prices.csv")
MACRO_DIR = Path("data/macro")
OUT_PATH = Path("data/prices_with_macro.csv")


def main():
    if not PRICE_PATH.exists():
        print(f"⚠️ Price file missing: {PRICE_PATH}")
        return

    price_df = pd.read_csv(PRICE_PATH, parse_dates=["date"])
    price_df = price_df.sort_values("date")

    if not MACRO_DIR.exists():
        print(f"⚠️ Macro directory missing: {MACRO_DIR}")
        price_df.to_csv(OUT_PATH, index=False)
        print(f"Saved prices without macro to {OUT_PATH}")
        return

    macro_files = list(MACRO_DIR.glob("*.csv"))
    if not macro_files:
        print("⚠️ No macro CSVs found; copying prices.")
        price_df.to_csv(OUT_PATH, index=False)
        return

    macro_frames = []
    for mf in macro_files:
        df = pd.read_csv(mf, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        macro_frames.append(df)

    macro_df = pd.concat(macro_frames, axis=1).sort_index().ffill()
    merged = price_df.set_index("date").join(macro_df, how="left").ffill().reset_index()
    merged.to_csv(OUT_PATH, index=False)
    print(f"✅ Merged macro features saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
