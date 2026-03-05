"""
Report Generator for Dissertation Artifacts

Generates publication-ready reports including:
- LaTeX tables and figures
- Markdown summaries
- Complete experiment reports
- Model comparison tables
- Auto-generated figure captions

Designed to produce dissertation-quality documentation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import textwrap

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Output format for reports"""
    LATEX = "latex"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class TableStyle(Enum):
    """LaTeX table styles"""
    BOOKTABS = "booktabs"  # Publication style
    PLAIN = "plain"
    COLORED = "colored"


@dataclass
class FigureCaption:
    """Auto-generated figure caption"""
    short_caption: str
    long_caption: str
    label: str
    figure_path: str


@dataclass
class TableConfig:
    """Configuration for table generation"""
    style: TableStyle = TableStyle.BOOKTABS
    float_format: str = ".4f"
    highlight_best: bool = True
    highlight_color: str = "lightgray"
    include_std: bool = True
    bold_best: bool = True


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    format: ReportFormat = ReportFormat.LATEX
    include_figures: bool = True
    include_tables: bool = True
    include_appendix: bool = True
    auto_captions: bool = True
    table_config: TableConfig = field(default_factory=TableConfig)
    figure_dpi: int = 300
    output_dir: str = "reports"


@dataclass
class ExperimentSummary:
    """Summary of an experiment for reporting"""
    experiment_id: str
    model_name: str
    model_type: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    n_parameters: int
    timestamp: str
    seed: int
    notes: Optional[str] = None


class ReportGenerator:
    """
    Generates comprehensive reports for dissertation documentation.

    Supports LaTeX, Markdown, and HTML output formats with
    publication-quality tables and auto-generated captions.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()

    def generate_latex_table(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        highlight_cols: Optional[List[str]] = None,
        minimize: bool = True
    ) -> str:
        """
        Generate a LaTeX table from DataFrame.

        Args:
            df: Data to tabulate
            caption: Table caption
            label: LaTeX label for referencing
            highlight_cols: Columns where best values should be highlighted
            minimize: If True, lower is better; if False, higher is better

        Returns:
            LaTeX table string
        """
        # Create copy for formatting
        df_formatted = df.copy()

        # Format numeric columns
        for col in df_formatted.select_dtypes(include=[np.number]).columns:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:{self.config.table_config.float_format}}" if pd.notna(x) else "-"
            )

        # Highlight best values
        if highlight_cols and self.config.table_config.bold_best:
            for col in highlight_cols:
                if col in df.columns:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if minimize:
                        best_idx = numeric_col.idxmin()
                    else:
                        best_idx = numeric_col.idxmax()

                    if pd.notna(best_idx):
                        current_val = df_formatted.loc[best_idx, col]
                        df_formatted.loc[best_idx, col] = f"\\textbf{{{current_val}}}"

        # Build LaTeX
        lines = []

        if self.config.table_config.style == TableStyle.BOOKTABS:
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            lines.append(f"\\caption{{{caption}}}")
            lines.append(f"\\label{{tab:{label}}}")

            col_format = "l" + "r" * (len(df.columns) - 1)
            lines.append(f"\\begin{{tabular}}{{{col_format}}}")
            lines.append("\\toprule")

            # Header
            header = " & ".join(df_formatted.columns)
            lines.append(f"{header} \\\\")
            lines.append("\\midrule")

            # Data rows
            for _, row in df_formatted.iterrows():
                row_str = " & ".join(str(v) for v in row.values)
                lines.append(f"{row_str} \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")

        return "\n".join(lines)

    def generate_markdown_table(
        self,
        df: pd.DataFrame,
        caption: str
    ) -> str:
        """
        Generate a Markdown table from DataFrame.

        Args:
            df: Data to tabulate
            caption: Table caption

        Returns:
            Markdown table string
        """
        lines = [f"### {caption}", ""]

        # Format DataFrame
        df_formatted = df.copy()
        for col in df_formatted.select_dtypes(include=[np.number]).columns:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:{self.config.table_config.float_format}}" if pd.notna(x) else "-"
            )

        # Header
        lines.append("| " + " | ".join(df_formatted.columns) + " |")
        lines.append("|" + "|".join(["---"] * len(df_formatted.columns)) + "|")

        # Data rows
        for _, row in df_formatted.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")

        return "\n".join(lines)

    def generate_figure_caption(
        self,
        figure_type: str,
        model_name: str,
        metric_value: Optional[float] = None,
        additional_info: Optional[str] = None
    ) -> FigureCaption:
        """
        Auto-generate figure caption based on content.

        Args:
            figure_type: Type of figure (learning_curve, predictions, etc.)
            model_name: Model name
            metric_value: Optional key metric to include
            additional_info: Additional context

        Returns:
            FigureCaption object
        """
        captions = {
            "learning_curve": (
                f"Training progress for {model_name}",
                f"Training and validation loss curves for the {model_name} model over all epochs. "
                f"The training loss (solid line) and validation loss (dashed line) demonstrate "
                f"{'convergence' if metric_value and metric_value < 0.1 else 'the learning dynamics'}."
            ),
            "predictions": (
                f"Prediction accuracy for {model_name}",
                f"Comparison of predicted values against actual values for {model_name}. "
                f"The scatter plot shows the correlation between predictions and actuals, "
                f"with the diagonal line representing perfect prediction."
            ),
            "residual_histogram": (
                f"PDE residual distribution for {model_name}",
                f"Histogram of physics-informed residuals for {model_name}. "
                f"A distribution centered near zero indicates good compliance with the "
                f"underlying physical constraints."
            ),
            "gradient_norms": (
                f"Gradient evolution for {model_name}",
                f"Evolution of gradient norms for different loss components during training. "
                f"Balanced gradient magnitudes indicate stable multi-task optimization."
            ),
            "drawdown": (
                f"Risk analysis for {model_name}",
                f"Cumulative returns and drawdown analysis for {model_name} trading strategy. "
                f"The upper panel shows cumulative returns, while the lower panel shows "
                f"the drawdown from peak."
            ),
            "rolling_performance": (
                f"Rolling window performance for {model_name}",
                f"Performance metrics computed over rolling windows for {model_name}. "
                f"This demonstrates the model's consistency across different time periods."
            ),
        }

        short, long = captions.get(figure_type, (
            f"Figure for {model_name}",
            f"Visualization for {model_name} model analysis."
        ))

        if additional_info:
            long += f" {additional_info}"

        label = f"fig:{figure_type}_{model_name.lower().replace(' ', '_')}"

        return FigureCaption(
            short_caption=short,
            long_caption=long,
            label=label,
            figure_path=f"figures/{figure_type}_{model_name.lower().replace(' ', '_')}.pdf"
        )

    def generate_latex_figure(
        self,
        caption: FigureCaption,
        width: str = "0.8\\textwidth"
    ) -> str:
        """
        Generate LaTeX figure environment.

        Args:
            caption: FigureCaption object
            width: Figure width

        Returns:
            LaTeX figure string
        """
        return textwrap.dedent(f"""
        \\begin{{figure}}[htbp]
            \\centering
            \\includegraphics[width={width}]{{{caption.figure_path}}}
            \\caption[{caption.short_caption}]{{{caption.long_caption}}}
            \\label{{{caption.label}}}
        \\end{{figure}}
        """).strip()

    def generate_model_comparison_table(
        self,
        results: List[ExperimentSummary],
        metrics: List[str] = None
    ) -> str:
        """
        Generate model comparison table.

        Args:
            results: List of experiment summaries
            metrics: Metrics to include in comparison

        Returns:
            Table string in configured format
        """
        if not metrics:
            # Default metrics
            metrics = ['mse', 'mae', 'rmse', 'sharpe', 'max_drawdown']

        # Build DataFrame
        rows = []
        for exp in results:
            row = {'Model': exp.model_name}
            for metric in metrics:
                if metric in exp.metrics:
                    row[metric.upper()] = exp.metrics[metric]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Determine which metrics should be minimized
        minimize_metrics = ['mse', 'mae', 'rmse', 'max_drawdown']
        highlight_cols = [m.upper() for m in metrics if m.lower() in minimize_metrics]

        if self.config.format == ReportFormat.LATEX:
            return self.generate_latex_table(
                df,
                caption="Model comparison across evaluation metrics",
                label="model_comparison",
                highlight_cols=highlight_cols,
                minimize=True
            )
        else:
            return self.generate_markdown_table(
                df,
                caption="Model Comparison"
            )

    def generate_experiment_report(
        self,
        summary: ExperimentSummary,
        figures: Dict[str, Path],
        detailed_metrics: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate complete experiment report.

        Args:
            summary: Experiment summary
            figures: Dictionary of figure_type -> path
            detailed_metrics: Optional detailed metrics DataFrame

        Returns:
            Complete report string
        """
        sections = []

        # Header
        if self.config.format == ReportFormat.LATEX:
            sections.append(f"\\section{{{summary.model_name} Experiment Report}}")
            sections.append(f"\\label{{sec:{summary.experiment_id}}}")
        else:
            sections.append(f"# {summary.model_name} Experiment Report")
            sections.append("")

        # Metadata
        sections.append(self._generate_metadata_section(summary))

        # Configuration
        sections.append(self._generate_config_section(summary))

        # Results
        sections.append(self._generate_results_section(summary))

        # Figures
        if self.config.include_figures and figures:
            sections.append(self._generate_figures_section(summary, figures))

        # Detailed metrics
        if detailed_metrics is not None:
            sections.append(self._generate_detailed_metrics_section(detailed_metrics))

        return "\n\n".join(sections)

    def _generate_metadata_section(self, summary: ExperimentSummary) -> str:
        """Generate metadata section"""
        if self.config.format == ReportFormat.LATEX:
            return textwrap.dedent(f"""
            \\subsection{{Experiment Metadata}}
            \\begin{{itemize}}
                \\item Experiment ID: \\texttt{{{summary.experiment_id}}}
                \\item Model Type: {summary.model_type}
                \\item Timestamp: {summary.timestamp}
                \\item Random Seed: {summary.seed}
                \\item Parameters: {summary.n_parameters:,}
                \\item Training Time: {summary.training_time:.2f}s
            \\end{{itemize}}
            """).strip()
        else:
            return textwrap.dedent(f"""
            ## Experiment Metadata

            - **Experiment ID**: `{summary.experiment_id}`
            - **Model Type**: {summary.model_type}
            - **Timestamp**: {summary.timestamp}
            - **Random Seed**: {summary.seed}
            - **Parameters**: {summary.n_parameters:,}
            - **Training Time**: {summary.training_time:.2f}s
            """).strip()

    def _generate_config_section(self, summary: ExperimentSummary) -> str:
        """Generate configuration section"""
        config_items = []
        for key, value in summary.config.items():
            if isinstance(value, dict):
                continue  # Skip nested dicts for brevity
            config_items.append((key, value))

        if self.config.format == ReportFormat.LATEX:
            items = "\n".join(
                f"    \\item {k}: {v}" for k, v in config_items[:10]
            )
            return textwrap.dedent(f"""
            \\subsection{{Configuration}}
            \\begin{{itemize}}
            {items}
            \\end{{itemize}}
            """).strip()
        else:
            items = "\n".join(f"- **{k}**: {v}" for k, v in config_items[:10])
            return f"## Configuration\n\n{items}"

    def _generate_results_section(self, summary: ExperimentSummary) -> str:
        """Generate results section"""
        metrics_df = pd.DataFrame([summary.metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.index.name = 'Metric'
        metrics_df = metrics_df.reset_index()

        if self.config.format == ReportFormat.LATEX:
            table = self.generate_latex_table(
                metrics_df,
                caption=f"Performance metrics for {summary.model_name}",
                label=f"metrics_{summary.experiment_id}"
            )
            return f"\\subsection{{Results}}\n\n{table}"
        else:
            table = self.generate_markdown_table(
                metrics_df,
                caption="Performance Metrics"
            )
            return f"## Results\n\n{table}"

    def _generate_figures_section(
        self,
        summary: ExperimentSummary,
        figures: Dict[str, Path]
    ) -> str:
        """Generate figures section"""
        sections = []

        if self.config.format == ReportFormat.LATEX:
            sections.append("\\subsection{Visualizations}")
        else:
            sections.append("## Visualizations")

        for fig_type, fig_path in figures.items():
            caption = self.generate_figure_caption(
                fig_type,
                summary.model_name
            )

            if self.config.format == ReportFormat.LATEX:
                sections.append(self.generate_latex_figure(caption))
            else:
                sections.append(f"![{caption.short_caption}]({fig_path})")
                sections.append(f"*{caption.long_caption}*")

        return "\n\n".join(sections)

    def _generate_detailed_metrics_section(
        self,
        detailed_metrics: pd.DataFrame
    ) -> str:
        """Generate detailed metrics section"""
        if self.config.format == ReportFormat.LATEX:
            table = self.generate_latex_table(
                detailed_metrics,
                caption="Detailed performance metrics by window/regime",
                label="detailed_metrics"
            )
            return f"\\subsection{{Detailed Metrics}}\n\n{table}"
        else:
            table = self.generate_markdown_table(
                detailed_metrics,
                caption="Detailed Metrics"
            )
            return f"## Detailed Metrics\n\n{table}"

    def generate_full_report(
        self,
        experiments: List[ExperimentSummary],
        figures_dir: Path,
        output_path: Path,
        title: str = "PINN Financial Forecasting Evaluation Report"
    ) -> Path:
        """
        Generate a complete report with all experiments.

        Args:
            experiments: List of experiment summaries
            figures_dir: Directory containing figures
            output_path: Output file path
            title: Report title

        Returns:
            Path to generated report
        """
        sections = []

        # Document header
        if self.config.format == ReportFormat.LATEX:
            sections.append(self._latex_header(title))
        else:
            sections.append(f"# {title}\n")
            sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # Executive summary
        sections.append(self._generate_executive_summary(experiments))

        # Model comparison
        sections.append(self._generate_comparison_section(experiments))

        # Individual experiment reports
        for exp in experiments:
            # Find figures for this experiment
            exp_figures = {}
            for fig_path in figures_dir.glob(f"*{exp.experiment_id}*"):
                fig_type = fig_path.stem.split('_')[0]
                exp_figures[fig_type] = fig_path

            sections.append(self.generate_experiment_report(exp, exp_figures))

        # Document footer
        if self.config.format == ReportFormat.LATEX:
            sections.append("\\end{document}")

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = "\n\n".join(sections)
        output_path.write_text(content)

        logger.info(f"Generated report: {output_path}")
        return output_path

    def _latex_header(self, title: str) -> str:
        """Generate LaTeX document header"""
        return textwrap.dedent(f"""
        \\documentclass[11pt]{{article}}
        \\usepackage{{booktabs}}
        \\usepackage{{graphicx}}
        \\usepackage{{hyperref}}
        \\usepackage{{amsmath}}
        \\usepackage{{geometry}}
        \\geometry{{margin=1in}}

        \\title{{{title}}}
        \\author{{Generated by PINN Financial Forecasting System}}
        \\date{{\\today}}

        \\begin{{document}}
        \\maketitle
        \\tableofcontents
        \\newpage
        """).strip()

    def _generate_executive_summary(
        self,
        experiments: List[ExperimentSummary]
    ) -> str:
        """Generate executive summary"""
        # Find best model
        best_exp = min(experiments, key=lambda x: x.metrics.get('mse', float('inf')))

        if self.config.format == ReportFormat.LATEX:
            return textwrap.dedent(f"""
            \\section{{Executive Summary}}

            This report presents the evaluation of {len(experiments)} models for financial
            time series forecasting using Physics-Informed Neural Networks (PINNs).

            \\textbf{{Key Findings:}}
            \\begin{{itemize}}
                \\item Best performing model: \\textbf{{{best_exp.model_name}}}
                \\item Best MSE achieved: {best_exp.metrics.get('mse', 'N/A'):.6f}
                \\item Best Sharpe ratio: {best_exp.metrics.get('sharpe', 'N/A'):.4f}
            \\end{{itemize}}
            """).strip()
        else:
            return textwrap.dedent(f"""
            ## Executive Summary

            This report presents the evaluation of {len(experiments)} models for financial
            time series forecasting using Physics-Informed Neural Networks (PINNs).

            **Key Findings:**
            - Best performing model: **{best_exp.model_name}**
            - Best MSE achieved: {best_exp.metrics.get('mse', 'N/A'):.6f}
            - Best Sharpe ratio: {best_exp.metrics.get('sharpe', 'N/A'):.4f}
            """).strip()

    def _generate_comparison_section(
        self,
        experiments: List[ExperimentSummary]
    ) -> str:
        """Generate model comparison section"""
        comparison_table = self.generate_model_comparison_table(experiments)

        if self.config.format == ReportFormat.LATEX:
            return f"\\section{{Model Comparison}}\n\n{comparison_table}"
        else:
            return f"## Model Comparison\n\n{comparison_table}"


def generate_dissertation_report(
    experiments: List[ExperimentSummary],
    figures_dir: Union[str, Path],
    output_path: Union[str, Path],
    format: str = "latex"
) -> Path:
    """
    Convenience function for generating dissertation reports.

    Args:
        experiments: List of experiment summaries
        figures_dir: Directory containing figures
        output_path: Output file path
        format: Output format (latex, markdown)

    Returns:
        Path to generated report
    """
    config = ReportConfig(format=ReportFormat(format))
    generator = ReportGenerator(config)

    return generator.generate_full_report(
        experiments,
        Path(figures_dir),
        Path(output_path)
    )
