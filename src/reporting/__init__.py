"""Reporting and visualization modules"""

from .plot_generator import (
    PlotGenerator,
    PlotConfig,
    PlotStyle,
    ExperimentResults,
    create_standard_plots,
    save_all_figures
)
from .report_generator import (
    ReportGenerator,
    ReportConfig,
    ReportFormat,
    TableStyle,
    TableConfig,
    FigureCaption,
    ExperimentSummary,
    generate_dissertation_report
)
from .pde_visualization import (
    BurgersVisualization,
    create_comparison_visualization,
)

__all__ = [
    # Plot generation
    "PlotGenerator",
    "PlotConfig",
    "PlotStyle",
    "ExperimentResults",
    "create_standard_plots",
    "save_all_figures",
    # Report generation
    "ReportGenerator",
    "ReportConfig",
    "ReportFormat",
    "TableStyle",
    "TableConfig",
    "FigureCaption",
    "ExperimentSummary",
    "generate_dissertation_report",
    # PDE visualization
    "BurgersVisualization",
    "create_comparison_visualization",
]
