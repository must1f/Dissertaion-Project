#!/usr/bin/env python3
"""
Generate a leaderboard from the results database.

Usage:
    python scripts/generate_leaderboard.py --metric sharpe_ratio --top-n 15 --format markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.leaderboard import ResultsDatabase, LeaderboardGenerator, RankingMetric  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leaderboard from evaluation results.")
    parser.add_argument("--db", type=Path, default=Path("results/experiments.db"), help="Path to SQLite DB")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", help="Metric to rank by")
    parser.add_argument("--top-n", type=int, default=10, help="Number of rows")
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "markdown", "latex"],
        help="Output format",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional file to save")
    args = parser.parse_args()

    db = ResultsDatabase(args.db)
    generator = LeaderboardGenerator(db)

    metric_enum = next((m for m in RankingMetric if m.value == args.metric), RankingMetric.SHARPE)
    leaderboard = generator.generate_leaderboard(metric_enum, top_n=args.top_n)

    # Build table
    rows = []
    headers = ["Rank", "Experiment", "Model", args.metric]
    for e in leaderboard.entries:
        rows.append([e.rank, e.experiment_id, e.model_name, f"{e.metric_value:.4f}"])

    if args.format == "latex":
        df = generator.generate_comparison_table(metrics=[args.metric])
        table = generator.export_latex(df, caption=f"Leaderboard by {args.metric}", label="tab:leaderboard")
    elif args.format == "markdown":
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            print("pandas is required for markdown output", file=sys.stderr)
            sys.exit(1)
        df = pd.DataFrame(rows, columns=headers)
        table = df.to_markdown(index=False)
    else:
        # Plain text
        col_widths = [max(len(str(col)), max(len(str(r[i])) for r in rows)) for i, col in enumerate(headers)]
        header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
        sep = "-+-".join("-" * w for w in col_widths)
        body = "\n".join(" | ".join(f"{str(r[i]):<{col_widths[i]}}" for i in range(len(headers))) for r in rows)
        table = f"{header_line}\n{sep}\n{body}"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)
        print(f"Saved leaderboard to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()

