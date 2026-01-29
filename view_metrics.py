#!/usr/bin/env python3
"""
Terminal CLI for Viewing Comprehensive Financial Metrics

Usage:
    python3 view_metrics.py                    # View all models
    python3 view_metrics.py --model pinn_gbm   # View specific model
    python3 view_metrics.py --compare          # Compare all models
    python3 view_metrics.py --summary          # Quick summary
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.text import Text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_config

console = Console()


class MetricsViewer:
    """CLI tool for viewing model metrics"""

    def __init__(self):
        self.config = get_config()
        self.results_dir = self.config.project_root / 'results'

        # Model registry
        self.all_models = {
            # Baseline models
            'lstm': 'LSTM',
            'gru': 'GRU',
            'bilstm': 'BiLSTM',
            'attention_lstm': 'Attention LSTM',
            'transformer': 'Transformer',

            # PINN variants
            'pinn_baseline': 'PINN Baseline (Data-only)',
            'baseline': 'PINN Baseline (Data-only)',
            'pinn_gbm': 'PINN GBM (Trend)',
            'gbm': 'PINN GBM (Trend)',
            'pinn_ou': 'PINN OU (Mean-Reversion)',
            'ou': 'PINN OU (Mean-Reversion)',
            'pinn_black_scholes': 'PINN Black-Scholes',
            'black_scholes': 'PINN Black-Scholes',
            'pinn_gbm_ou': 'PINN GBM+OU Hybrid',
            'gbm_ou': 'PINN GBM+OU Hybrid',
            'pinn_global': 'PINN Global Constraint',
            'global': 'PINN Global Constraint',
        }

    def load_model_results(self, model_key: str) -> Optional[Dict]:
        """Load results for a specific model"""
        patterns = [
            self.results_dir / f'{model_key}_results.json',
            self.results_dir / f'pinn_{model_key}_results.json',
        ]

        for path in patterns:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)

        return None

    def load_all_results(self) -> Dict[str, Dict]:
        """Load all available model results"""
        results = {}

        # Scan results directory for all result files
        if self.results_dir.exists():
            for result_file in self.results_dir.glob('*_results.json'):
                model_key = result_file.stem.replace('_results', '')
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        results[model_key] = data
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load {result_file.name}: {e}[/yellow]")

        return results

    def format_metric(self, value, format_type='float'):
        """Format metric value for display"""
        if value is None or (isinstance(value, float) and (value != value)):  # NaN check
            return 'N/A'

        if isinstance(value, str):
            return value

        if format_type == 'float':
            if abs(value) > 1000:
                return f'{value:,.2f}'
            elif abs(value) > 10:
                return f'{value:.3f}'
            else:
                return f'{value:.6f}'
        elif format_type == 'percent':
            return f'{value * 100:.2f}%'
        elif format_type == 'integer':
            return f'{int(value):,}'

        return str(value)

    def display_model_summary(self, model_key: str, results: Dict):
        """Display comprehensive summary for a single model"""
        model_name = results.get('model_name', model_key)

        console.print(Panel.fit(
            f"[bold cyan]{model_name}[/bold cyan]",
            border_style="cyan"
        ))

        # ML Metrics
        ml_metrics = results.get('ml_metrics', {})
        if ml_metrics:
            table = Table(title="Traditional ML Metrics", box=box.ROUNDED, show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")

            table.add_row("MSE", self.format_metric(ml_metrics.get('mse')))
            table.add_row("MAE", self.format_metric(ml_metrics.get('mae')))
            table.add_row("RMSE", self.format_metric(ml_metrics.get('rmse')))
            table.add_row("R²", self.format_metric(ml_metrics.get('r2')))
            if 'mape' in ml_metrics:
                table.add_row("MAPE", self.format_metric(ml_metrics.get('mape')))

            console.print(table)
            console.print()

        # Financial Metrics
        financial_metrics = results.get('financial_metrics', {})
        if financial_metrics:
            # Risk-Adjusted Performance
            table = Table(title="Risk-Adjusted Performance", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            sharpe = financial_metrics.get('sharpe_ratio')
            sharpe_str = self.format_metric(sharpe)
            if isinstance(sharpe, (int, float)) and sharpe > 2.0:
                sharpe_str = f"[bold green]{sharpe_str} ⭐[/bold green]"
            table.add_row("Sharpe Ratio", sharpe_str)

            table.add_row("Sortino Ratio", self.format_metric(financial_metrics.get('sortino_ratio')))
            table.add_row("Volatility", f"{self.format_metric(financial_metrics.get('volatility'))}%")

            console.print(table)
            console.print()

            # Capital Preservation
            table = Table(title="Capital Preservation", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            max_dd = financial_metrics.get('max_drawdown')
            table.add_row("Max Drawdown", f"{self.format_metric(max_dd)}%")
            table.add_row("Drawdown Duration", f"{self.format_metric(financial_metrics.get('drawdown_duration'))} years")
            table.add_row("Calmar Ratio", self.format_metric(financial_metrics.get('calmar_ratio')))

            console.print(table)
            console.print()

            # Trading Viability
            table = Table(title="Trading Viability", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Annualized Return", f"{self.format_metric(financial_metrics.get('annualized_return'))}%")
            table.add_row("Profit Factor", self.format_metric(financial_metrics.get('profit_factor')))
            table.add_row("Win Rate", self.format_metric(financial_metrics.get('win_rate'), 'percent'))

            console.print(table)
            console.print()

            # Signal Quality
            table = Table(title="Signal Quality", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            dir_acc = financial_metrics.get('directional_accuracy')
            if dir_acc is not None:
                table.add_row("Directional Accuracy", self.format_metric(dir_acc, 'percent'))
            table.add_row("Information Coefficient", self.format_metric(financial_metrics.get('information_coefficient')))
            table.add_row("Precision", self.format_metric(financial_metrics.get('precision'), 'percent'))
            table.add_row("Recall", self.format_metric(financial_metrics.get('recall'), 'percent'))

            console.print(table)
            console.print()

        # Rolling Metrics
        rolling_metrics = results.get('rolling_metrics', {})
        if rolling_metrics and 'stability' in rolling_metrics:
            stability = rolling_metrics['stability']

            table = Table(title="Robustness & Stability", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Rolling Windows", str(rolling_metrics.get('n_windows', 'N/A')))
            table.add_row("Sharpe CV", self.format_metric(stability.get('sharpe_ratio_cv')))
            table.add_row("Sharpe Consistency", self.format_metric(stability.get('sharpe_ratio_consistency'), 'percent'))
            table.add_row("DirAcc Consistency", self.format_metric(stability.get('directional_accuracy_consistency'), 'percent'))

            console.print(table)
            console.print()

    def display_comparison_table(self, results: Dict[str, Dict]):
        """Display comparison table for all models"""
        console.print(Panel.fit(
            "[bold cyan]Financial Metrics Comparison - All Models[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        # Create comparison table
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("RMSE", style="yellow", justify="right")
        table.add_column("R²", style="yellow", justify="right")
        table.add_column("Sharpe", style="green", justify="right")
        table.add_column("Sortino", style="green", justify="right")
        table.add_column("Dir Acc", style="green", justify="right")
        table.add_column("Win Rate", style="green", justify="right")
        table.add_column("Profit Factor", style="green", justify="right")

        # Sort by Sharpe ratio (descending)
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1].get('financial_metrics', {}).get('sharpe_ratio', -999),
            reverse=True
        )

        for model_key, data in sorted_models:
            model_name = data.get('model_name', model_key)
            ml = data.get('ml_metrics', {})
            fm = data.get('financial_metrics', {})

            # Check if model has comprehensive metrics
            has_metrics = 'sharpe_ratio' in fm and fm['sharpe_ratio'] is not None
            status = "✅" if has_metrics else "⚪"

            table.add_row(
                model_name[:30],  # Truncate long names
                status,
                self.format_metric(ml.get('rmse')),
                self.format_metric(ml.get('r2')),
                self.format_metric(fm.get('sharpe_ratio')),
                self.format_metric(fm.get('sortino_ratio')),
                self.format_metric(fm.get('directional_accuracy'), 'percent'),
                self.format_metric(fm.get('win_rate'), 'percent'),
                self.format_metric(fm.get('profit_factor'))
            )

        console.print(table)
        console.print()

        # Summary statistics
        with_metrics = sum(1 for _, data in results.items()
                          if data.get('financial_metrics', {}).get('sharpe_ratio') is not None)

        console.print(f"[cyan]Total Models: {len(results)}[/cyan]")
        console.print(f"[green]With Comprehensive Metrics: {with_metrics}[/green]")
        console.print(f"[yellow]Awaiting Metrics: {len(results) - with_metrics}[/yellow]")
        console.print()

    def display_quick_summary(self, results: Dict[str, Dict]):
        """Display quick summary of all models"""
        console.print(Panel.fit(
            "[bold cyan]Quick Summary - All Models[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        for model_key, data in sorted(results.items()):
            model_name = data.get('model_name', model_key)
            fm = data.get('financial_metrics', {})

            sharpe = fm.get('sharpe_ratio')
            dir_acc = fm.get('directional_accuracy')

            if sharpe is not None:
                status = "✅"
                sharpe_str = self.format_metric(sharpe)
                dir_acc_str = self.format_metric(dir_acc, 'percent')
                summary = f"Sharpe: {sharpe_str} | Dir Acc: {dir_acc_str}"
                color = "green"
            else:
                status = "⚪"
                summary = "No comprehensive metrics"
                color = "yellow"

            console.print(f"{status} [bold]{model_name}[/bold]: [{color}]{summary}[/{color}]")

        console.print()

    def run(self, args):
        """Main execution"""
        if args.model:
            # Display specific model
            results = self.load_model_results(args.model)
            if results:
                self.display_model_summary(args.model, results)
            else:
                console.print(f"[red]Error: Model '{args.model}' not found or has no results[/red]")
                console.print(f"[yellow]Available models: {', '.join(self.all_models.keys())}[/yellow]")
                return 1

        elif args.compare:
            # Compare all models
            results = self.load_all_results()
            if results:
                self.display_comparison_table(results)
            else:
                console.print("[red]Error: No model results found[/red]")
                console.print("[yellow]Run training and evaluation first[/yellow]")
                return 1

        elif args.summary:
            # Quick summary
            results = self.load_all_results()
            if results:
                self.display_quick_summary(results)
            else:
                console.print("[red]Error: No model results found[/red]")
                return 1

        else:
            # Default: display all models in detail
            results = self.load_all_results()
            if results:
                for i, (model_key, data) in enumerate(sorted(results.items())):
                    if i > 0:
                        console.print("\n" + "="*80 + "\n")
                    self.display_model_summary(model_key, data)
            else:
                console.print("[red]Error: No model results found[/red]")
                console.print("[yellow]Run training and evaluation first[/yellow]")
                return 1

        return 0


def main():
    parser = argparse.ArgumentParser(
        description='View comprehensive financial metrics for neural network models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # View all models in detail
  %(prog)s --model pinn_gbm         # View specific model
  %(prog)s --compare                # Compare all models (table)
  %(prog)s --summary                # Quick summary of all models
        """
    )

    parser.add_argument('--model', type=str, help='View specific model by key')
    parser.add_argument('--compare', action='store_true', help='Compare all models in table format')
    parser.add_argument('--summary', action='store_true', help='Show quick summary of all models')

    args = parser.parse_args()

    viewer = MetricsViewer()
    return viewer.run(args)


if __name__ == '__main__':
    sys.exit(main())
