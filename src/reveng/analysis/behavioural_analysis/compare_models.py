"""Compare uncertainty analysis results across multiple LLM models.

This module provides tools to:
- Load results from multiple models (from CSV or by running analysis)
- Compare calibration, correlations, and divergence across models
- Generate publication-ready visualizations
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reveng.analysis.behavioural_analysis.analysis_utils import (
    CalibrationMetrics,
    ControlledAnalysisResult,
    DistanceToGoalMetrics,
    UncertaintyAccuracyMetrics,
    compute_calibration_metrics,
    compute_correlation,
    compute_distance_summary,
    compute_distance_to_goal_metrics,
    compute_partial_correlations,
    compute_uncertainty_accuracy_metrics,
    compute_within_stratum_correlations,
    discover_metadata_files,
    load_environments,
    sanitize_label,
)

# =============================================================================
# Paper-Ready Plot Configuration
# =============================================================================

# Publication-quality settings
PAPER_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

# Color palette for models (colorblind-friendly)
MODEL_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky blue
    "#E69F00",  # Orange
]


def setup_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(PAPER_RC)
    sns.set_palette(MODEL_COLORS)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelResults:
    """Results for a single model, used for cross-model comparison."""

    model_name: str
    df: pd.DataFrame  # Per-cell metrics
    calibration: CalibrationMetrics
    controlled_analysis: Optional[ControlledAnalysisResult] = None
    uncertainty_accuracy: Optional[UncertaintyAccuracyMetrics] = None
    distance_metrics: Optional[DistanceToGoalMetrics] = None

    @property
    def n_samples(self) -> int:
        return len(self.df)


@dataclass
class ComparisonResults:
    """Container for cross-model comparison results."""

    model_results: dict[str, ModelResults]
    summary_df: pd.DataFrame
    correlation_df: pd.DataFrame
    output_dir: Path
    metric: str = "jsd"


# =============================================================================
# Data Loading
# =============================================================================


def discover_model_directories(parent_dir: Path) -> list[Path]:
    """Discover metadata directories within a parent directory.

    Args:
        parent_dir: Directory containing subdirectories with metadata

    Returns:
        List of paths to directories containing metadata files
    """
    model_dirs = []
    for subdir in sorted(parent_dir.iterdir()):
        if subdir.is_dir():
            # Check if it contains metadata files
            metadata_files = list(subdir.glob("*_metadata.json"))
            if metadata_files:
                model_dirs.append(subdir)
    return model_dirs


def load_model_from_csv(
    csv_path: Path, model_name: Optional[str] = None
) -> ModelResults:
    """Load model results from a pre-computed CSV file.

    Args:
        csv_path: Path to uncertainty_states_*.csv file
        model_name: Optional name override (defaults to filename)

    Returns:
        ModelResults for the model
    """
    df = pd.read_csv(csv_path)

    if model_name is None:
        # Extract from filename: uncertainty_states_{model}.csv
        model_name = csv_path.stem.replace("uncertainty_states_", "")

    # Determine metric column
    divergence_col = "jsd" if "jsd" in df.columns else "cross_entropy_bits"

    calibration = compute_calibration_metrics(
        df,
        entropy_col="entropy_bits",
        optimal_entropy_col="optimal_entropy_bits",
        divergence_col=divergence_col,
    )

    # Compute uncertainty-accuracy metrics
    uncertainty_accuracy = compute_uncertainty_accuracy_metrics(df)

    # Compute distance metrics if distance_to_goal column exists
    distance_metrics = None
    if "distance_to_goal" in df.columns:
        distance_metrics = compute_distance_to_goal_metrics(
            df, divergence_col=divergence_col
        )

    return ModelResults(
        model_name=model_name,
        df=df,
        calibration=calibration,
        uncertainty_accuracy=uncertainty_accuracy,
        distance_metrics=distance_metrics,
    )


def load_model_from_metadata(
    metadata_dir: Path,
    grids_dataset: dict[str, Any],
    model_name: Optional[str] = None,
    batch_size: int = 50,
) -> ModelResults:
    """Load model results by processing metadata files.

    Args:
        metadata_dir: Directory containing LLM metadata JSON files
        grids_dataset: Pre-loaded grids dataset
        model_name: Optional name override
        batch_size: Batch size for processing

    Returns:
        ModelResults for the model
    """
    # Import here to avoid circular dependency
    from reveng.analysis.behavioural_analysis.uncertainty_analysis import (
        process_all_grids,
    )

    if model_name is None:
        model_name = sanitize_label(metadata_dir.name)

    metadata_files = discover_metadata_files(metadata_dir)
    if not metadata_files:
        raise ValueError(f"No metadata files found in {metadata_dir}")

    metrics_dicts = process_all_grids(grids_dataset, metadata_files, batch_size)
    df = pd.DataFrame(metrics_dicts)

    # Compute calibration (default to JSD)
    divergence_col = "jsd" if "jsd" in df.columns else "cross_entropy_bits"
    calibration = compute_calibration_metrics(
        df,
        entropy_col="entropy_bits",
        optimal_entropy_col="optimal_entropy_bits",
        divergence_col=divergence_col,
    )

    # Compute uncertainty-accuracy metrics
    uncertainty_accuracy = compute_uncertainty_accuracy_metrics(df)

    # Compute distance metrics
    distance_metrics = compute_distance_to_goal_metrics(
        df, divergence_col=divergence_col
    )

    return ModelResults(
        model_name=model_name,
        df=df,
        calibration=calibration,
        uncertainty_accuracy=uncertainty_accuracy,
        distance_metrics=distance_metrics,
    )


def load_models_from_directory(
    parent_dir: Path,
    dataset_path: str,
    batch_size: int = 50,
) -> dict[str, ModelResults]:
    """Load results from multiple model directories.

    Args:
        parent_dir: Directory containing model subdirectories
        dataset_path: Path to grids pickle file
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping model names to ModelResults
    """
    print(f"\nDiscovering model directories in {parent_dir}...")
    model_dirs = discover_model_directories(parent_dir)
    print(f"Found {len(model_dirs)} model directories")

    if not model_dirs:
        raise ValueError(f"No model directories found in {parent_dir}")

    print(f"\nLoading environments from {dataset_path}...")
    grids_dataset = load_environments(dataset_path)
    print(f"Loaded {len(grids_dataset)} environments")

    results = {}
    for i, model_dir in enumerate(model_dirs):
        model_name = sanitize_label(model_dir.name)
        print(f"\n[{i + 1}/{len(model_dirs)}] Processing {model_name}...")

        try:
            model_results = load_model_from_metadata(
                model_dir, grids_dataset, model_name, batch_size
            )
            results[model_name] = model_results
            print(f"   Loaded {model_results.n_samples} samples")
        except Exception as e:
            print(f"   Error processing {model_name}: {e}")

    return results


def load_models_from_csvs(csv_paths: list[Path]) -> dict[str, ModelResults]:
    """Load results from multiple CSV files.

    Args:
        csv_paths: List of paths to uncertainty_states_*.csv files

    Returns:
        Dictionary mapping model names to ModelResults
    """
    results = {}
    for csv_path in csv_paths:
        print(f"Loading {csv_path.name}...")
        model_results = load_model_from_csv(csv_path)
        results[model_results.model_name] = model_results

    return results


# =============================================================================
# Comparison Analysis
# =============================================================================


def compute_model_correlations(
    model_results: ModelResults,
    metric: str = "jsd",
) -> dict[str, dict[str, float]]:
    """Compute correlations for a single model.

    Args:
        model_results: Results for one model
        metric: 'jsd' or 'ce' for divergence metric

    Returns:
        Dictionary with raw, within-stratum, and partial correlations
    """
    df = model_results.df
    divergence_col = "jsd" if metric == "jsd" else "cross_entropy_bits"
    y_cols = ["entropy_bits", divergence_col]
    control_cols = ["grid_size", "complexity"]

    correlations = {"raw": {}, "within_stratum": {}, "partial": {}}

    # Raw correlations
    for y_col in y_cols:
        corr = compute_correlation(df["num_optimal_actions"], df[y_col])
        if corr:
            correlations["raw"][y_col] = corr.r

    # Within-stratum
    within = compute_within_stratum_correlations(
        df, "num_optimal_actions", y_cols, control_cols
    )
    for y_col, corr in within.items():
        correlations["within_stratum"][y_col] = corr.r

    # Partial
    partial = compute_partial_correlations(
        df, "num_optimal_actions", y_cols, control_cols
    )
    for y_col, corr in partial.items():
        correlations["partial"][y_col] = corr.r

    return correlations


def build_summary_dataframe(
    model_results: dict[str, ModelResults],
    metric: str = "jsd",
) -> pd.DataFrame:
    """Build summary DataFrame with one row per model.

    Args:
        model_results: Dictionary of model results
        metric: 'jsd' or 'ce' for divergence metric

    Returns:
        DataFrame with model comparison summary
    """
    divergence_col = "jsd" if metric == "jsd" else "cross_entropy_bits"
    metric_label = "JSD" if metric == "jsd" else "Cross-Entropy"

    rows = []
    for model_name, results in model_results.items():
        cal = results.calibration
        corrs = compute_model_correlations(results, metric)
        ua = results.uncertainty_accuracy
        dm = results.distance_metrics

        row = {
            "Model": model_name,
            "N": results.n_samples,
            "Mean Entropy": cal.mean_entropy,
            "Mean Opt. Entropy": cal.mean_optimal_entropy,
            f"Mean {metric_label}": cal.mean_divergence,
            "Calibration Error": cal.calibration_error,
            "Calibration Bias": cal.calibration_bias,
            "H_llm ↔ H_opt": cal.entropy_correlation,
            f"r(#opt, {metric_label}) Raw": corrs["raw"].get(divergence_col, np.nan),
            f"r(#opt, {metric_label}) Partial": corrs["partial"].get(
                divergence_col, np.nan
            ),
            "r(#opt, Entropy) Raw": corrs["raw"].get("entropy_bits", np.nan),
            "r(#opt, Entropy) Partial": corrs["partial"].get("entropy_bits", np.nan),
        }

        # Add uncertainty-accuracy metrics
        if ua is not None:
            row.update(
                {
                    "Accuracy": ua.accuracy,
                    "AUROC": ua.auroc if ua.auroc is not None else np.nan,
                    "ECE": ua.ece,
                    "Entropy Gap": ua.entropy_gap,
                }
            )

        # Add distance metrics
        if dm is not None:
            row.update(
                {
                    "r(Entropy, Dist)": dm.correlation_entropy_distance,
                    "r(Accuracy, Dist)": dm.correlation_accuracy_distance,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Model")


def build_correlation_dataframe(
    model_results: dict[str, ModelResults],
    metric: str = "jsd",
) -> pd.DataFrame:
    """Build detailed correlation DataFrame for all models.

    Args:
        model_results: Dictionary of model results
        metric: 'jsd' or 'ce' for divergence metric

    Returns:
        Long-format DataFrame with correlations
    """
    divergence_col = "jsd" if metric == "jsd" else "cross_entropy_bits"

    rows = []
    for model_name, results in model_results.items():
        corrs = compute_model_correlations(results, metric)

        for y_col in ["entropy_bits", divergence_col]:
            y_label = (
                "Entropy"
                if y_col == "entropy_bits"
                else ("JSD" if metric == "jsd" else "Cross-Entropy")
            )
            for corr_type in ["raw", "within_stratum", "partial"]:
                r_val = corrs[corr_type].get(y_col, np.nan)
                rows.append(
                    {
                        "Model": model_name,
                        "Metric": y_label,
                        "Correlation Type": corr_type.replace("_", " ").title(),
                        "r": r_val,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# Visualizations (Paper-Ready)
# =============================================================================


def plot_correlation_comparison(
    model_results: dict[str, ModelResults],
    output_path: Path,
    metric: str = "jsd",
    figsize: tuple[float, float] = (7, 4),
) -> None:
    """Plot correlation comparison across models (bar chart).

    Shows raw vs partial correlations for divergence metric.

    Args:
        model_results: Dictionary of model results
        output_path: Path to save figure
        metric: 'jsd' or 'ce'
        figsize: Figure size in inches
    """
    setup_paper_style()

    divergence_col = "jsd" if metric == "jsd" else "cross_entropy_bits"
    metric_label = "JSD" if metric == "jsd" else "Cross-Entropy"

    models = sorted(model_results.keys())
    raw_rs = []
    partial_rs = []

    for model in models:
        corrs = compute_model_correlations(model_results[model], metric)
        raw_rs.append(corrs["raw"].get(divergence_col, 0))
        partial_rs.append(corrs["partial"].get(divergence_col, 0))

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, raw_rs, width, label="Raw", color=MODEL_COLORS[0])
    ax.bar(x + width / 2, partial_rs, width, label="Partial", color=MODEL_COLORS[1])

    ax.set_ylabel(f"Correlation with {metric_label}")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_model_name(m) for m in models], rotation=45, ha="right")
    ax.legend(title="Correlation Type", frameon=False)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_divergence_by_branching(
    model_results: dict[str, ModelResults],
    output_path: Path,
    metric: str = "jsd",
    figsize: tuple[float, float] = (5, 4),
) -> None:
    """Plot divergence vs number of optimal actions for all models.

    Args:
        model_results: Dictionary of model results
        output_path: Path to save figure
        metric: 'jsd' or 'ce'
        figsize: Figure size in inches
    """
    setup_paper_style()

    divergence_col = "jsd" if metric == "jsd" else "cross_entropy_bits"
    metric_label = "JSD" if metric == "jsd" else "Cross-Entropy (bits)"

    fig, ax = plt.subplots(figsize=figsize)

    for i, (model_name, results) in enumerate(sorted(model_results.items())):
        grouped = (
            results.df.groupby("num_optimal_actions")[divergence_col]
            .agg(["mean", "sem"])
            .reset_index()
        )

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.errorbar(
            grouped["num_optimal_actions"],
            grouped["mean"],
            yerr=grouped["sem"],
            marker="o",
            capsize=3,
            color=color,
            label=_format_model_name(model_name),
        )

    ax.set_xlabel("Number of Optimal Actions")
    ax.set_ylabel(f"Mean {metric_label}")
    ax.legend(frameon=False, loc="best")
    ax.set_xticks([1, 2, 3, 4])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_scatter(
    model_results: dict[str, ModelResults],
    output_path: Path,
    figsize: tuple[float, float] = (5, 4.5),
    sample_frac: float = 0.1,
) -> None:
    """Plot LLM entropy vs optimal entropy, colored by model.

    Args:
        model_results: Dictionary of model results
        output_path: Path to save figure
        figsize: Figure size in inches
        sample_frac: Fraction of points to plot (for readability)
    """
    setup_paper_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration line
    ax.plot([0, 2], [0, 2], "k--", linewidth=1, label="Perfect calibration", alpha=0.7)

    for i, (model_name, results) in enumerate(sorted(model_results.items())):
        df_sample = results.df.sample(frac=sample_frac, random_state=42)
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        ax.scatter(
            df_sample["optimal_entropy_bits"],
            df_sample["entropy_bits"],
            alpha=0.3,
            s=10,
            color=color,
            label=_format_model_name(model_name),
        )

    ax.set_xlabel("Optimal Entropy (bits)")
    ax.set_ylabel("LLM Entropy (bits)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 2.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_summary(
    model_results: dict[str, ModelResults],
    output_path: Path,
    figsize: tuple[float, float] = (6, 4),
) -> None:
    """Plot calibration metrics comparison across models.

    Args:
        model_results: Dictionary of model results
        output_path: Path to save figure
        figsize: Figure size in inches
    """
    setup_paper_style()

    models = sorted(model_results.keys())
    cal_errors = [model_results[m].calibration.calibration_error for m in models]
    cal_biases = [model_results[m].calibration.calibration_bias for m in models]
    entropy_corrs = [model_results[m].calibration.entropy_correlation for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x - width, cal_errors, width, label="Calibration Error", color=MODEL_COLORS[0]
    )
    ax.bar(x, cal_biases, width, label="Calibration Bias", color=MODEL_COLORS[1])
    ax.bar(
        x + width,
        entropy_corrs,
        width,
        label="H_llm ↔ H_opt Corr.",
        color=MODEL_COLORS[2],
    )

    ax.set_ylabel("Value")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_model_name(m) for m in models], rotation=45, ha="right")
    ax.legend(frameon=False, loc="best")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_distance_comparison(
    model_results: dict[str, ModelResults],
    output_path: Path,
    metric: str = "jsd",
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot entropy and accuracy vs distance for all models.

    Args:
        model_results: Dictionary of model results
        output_path: Path to save figure
        metric: 'jsd' or 'ce'
        figsize: Figure size
    """
    setup_paper_style()

    divergence_col = "jsd" if metric == "jsd" else "cross_entropy_bits"

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, (model_name, results) in enumerate(sorted(model_results.items())):
        if "distance_to_goal" not in results.df.columns:
            continue

        dist_summary = compute_distance_summary(
            results.df, divergence_col=divergence_col
        )
        if len(dist_summary) == 0:
            continue

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        label = _format_model_name(model_name)

        # Entropy vs distance
        axes[0].plot(
            dist_summary["distance_to_goal"],
            dist_summary["mean_entropy"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=label,
        )

        # Divergence vs distance
        axes[1].plot(
            dist_summary["distance_to_goal"],
            dist_summary["mean_divergence"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=label,
        )

        # Accuracy vs distance
        axes[2].plot(
            dist_summary["distance_to_goal"],
            dist_summary["accuracy"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=label,
        )

    metric_label = "JSD" if metric == "jsd" else "Cross-Entropy"

    axes[0].set_xlabel("Distance to Goal")
    axes[0].set_ylabel("Mean Entropy (bits)")
    axes[0].set_title("Entropy vs Distance")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Distance to Goal")
    axes[1].set_ylabel(f"Mean {metric_label}")
    axes[1].set_title(f"{metric_label} vs Distance")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Distance to Goal")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy vs Distance")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(
    summary_df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Plot heatmap of key metrics across models.

    Args:
        summary_df: Summary DataFrame from build_summary_dataframe
        output_path: Path to save figure
        figsize: Figure size in inches
    """
    setup_paper_style()

    # Select numeric columns for heatmap (exclude N)
    cols_to_plot = [c for c in summary_df.columns if c not in ["Model", "N"]]
    plot_df = summary_df.set_index("Model")[cols_to_plot]

    # Normalize for visualization
    plot_normalized = (plot_df - plot_df.mean()) / plot_df.std()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_normalized.T,
        annot=plot_df.T.round(3),
        fmt="",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Z-score"},
        xticklabels=[_format_model_name(m) for m in plot_df.index],
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_model_name(name: str) -> str:
    """Format model name for display (shorten if needed)."""
    # Common patterns to clean up
    replacements = [
        ("together_ai_meta-llama_", ""),
        ("together_ai_", ""),
        ("openai_", ""),
        ("anthropic_", ""),
        ("_", " "),
    ]
    result = name
    for old, new in replacements:
        result = result.replace(old, new)

    if result == "Llama-3 3-70B-Instruct-Turbo":
        result = "Llama-3.3-70B"

    return result


# =============================================================================
# Main Comparison Pipeline
# =============================================================================


def compare_models(
    model_results: dict[str, ModelResults],
    output_dir: str | Path,
    metric: str = "jsd",
) -> ComparisonResults:
    """Run complete cross-model comparison and generate outputs.

    Args:
        model_results: Dictionary mapping model names to ModelResults
        output_dir: Directory to save outputs
        metric: 'jsd' or 'ce' for divergence metric

    Returns:
        ComparisonResults with all comparison data
    """
    metric_label = "JSD" if metric == "jsd" else "Cross-Entropy"

    print("\n" + "=" * 70)
    print(
        f"CROSS-MODEL COMPARISON ({len(model_results)} models, metric: {metric_label})"
    )
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build summary tables
    print("\n1. Computing summary statistics...")
    summary_df = build_summary_dataframe(model_results, metric)
    correlation_df = build_correlation_dataframe(model_results, metric)

    # Save CSVs
    summary_path = output_path / "model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   Saved: {summary_path}")

    corr_path = output_path / "model_correlations.csv"
    correlation_df.to_csv(corr_path, index=False)
    print(f"   Saved: {corr_path}")

    # Generate visualizations
    print("\n2. Generating visualizations...")

    plot_correlation_comparison(
        model_results,
        output_path / "correlation_comparison.png",
        metric,
    )
    print("   - correlation_comparison.png")

    plot_divergence_by_branching(
        model_results,
        output_path / "divergence_by_branching.png",
        metric,
    )
    print("   - divergence_by_branching.png")

    plot_calibration_scatter(
        model_results,
        output_path / "calibration_scatter.png",
    )
    print("   - calibration_scatter.png")

    plot_calibration_summary(
        model_results,
        output_path / "calibration_summary.png",
    )
    print("   - calibration_summary.png")

    plot_summary_heatmap(
        summary_df,
        output_path / "summary_heatmap.png",
    )
    print("   - summary_heatmap.png")

    # Distance-to-goal visualizations
    plot_metrics_by_distance_comparison(
        model_results,
        output_path / "metrics_by_distance.png",
        metric,
    )
    print("   - metrics_by_distance.png")

    # Save text report
    report_path = output_path / "comparison_report.txt"
    _save_comparison_report(model_results, summary_df, report_path, metric)
    print(f"\n3. Saved report: {report_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    return ComparisonResults(
        model_results=model_results,
        summary_df=summary_df,
        correlation_df=correlation_df,
        output_dir=output_path,
        metric=metric,
    )


def _save_comparison_report(
    model_results: dict[str, ModelResults],
    summary_df: pd.DataFrame,
    output_path: Path,
    metric: str,
) -> None:
    """Save text comparison report."""
    metric_label = "JSD" if metric == "jsd" else "Cross-Entropy"

    lines = [
        "CROSS-MODEL UNCERTAINTY COMPARISON",
        "=" * 60,
        f"Models compared: {len(model_results)}",
        f"Divergence metric: {metric_label}",
        "",
    ]

    # Summary table
    lines.append("SUMMARY TABLE:")
    lines.append("-" * 60)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    # Key findings
    lines.append("KEY FINDINGS:")
    lines.append("-" * 60)

    # Best calibrated model
    best_cal_idx = summary_df["Calibration Error"].idxmin()
    best_cal_model = summary_df.loc[best_cal_idx, "Model"]
    best_cal_err = summary_df.loc[best_cal_idx, "Calibration Error"]
    lines.append(f"Best calibrated: {best_cal_model} (error={best_cal_err:.4f})")

    # Strongest correlation (partial)
    partial_col = f"r(#opt, {metric_label}) Partial"
    strongest_idx = summary_df[partial_col].abs().idxmax()
    strongest_model = summary_df.loc[strongest_idx, "Model"]
    strongest_r = summary_df.loc[strongest_idx, partial_col]
    lines.append(
        f"Strongest controlled correlation: {strongest_model} (r={strongest_r:.4f})"
    )

    # Lowest divergence
    div_col = f"Mean {metric_label}"
    lowest_div_idx = summary_df[div_col].idxmin()
    lowest_div_model = summary_df.loc[lowest_div_idx, "Model"]
    lowest_div = summary_df.loc[lowest_div_idx, div_col]
    lines.append(f"Lowest mean {metric_label}: {lowest_div_model} ({lowest_div:.4f})")

    output_path.write_text("\n".join(lines) + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Compare uncertainty analysis across multiple LLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing model subdirectories with metadata files",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Path to the baseline grids pickle file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/analysis/model_comparison",
        help="Directory to save comparison outputs",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["jsd", "ce"],
        default="jsd",
        help="Divergence metric: 'jsd' or 'ce' (cross-entropy)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing metadata files",
    )

    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Load from existing CSV files instead of processing metadata",
    )

    args = parser.parse_args()

    models_path = Path(args.models_dir)

    if args.from_csv:
        # Load from CSV files
        csv_files = list(models_path.glob("*/uncertainty_states_*.csv"))
        if not csv_files:
            print(f"No CSV files found in {models_path}/*/uncertainty_states_*.csv")
            return
        model_results = load_models_from_csvs(csv_files)
    else:
        # Process metadata directories
        model_results = load_models_from_directory(
            models_path,
            args.dataset,
            args.batch_size,
        )

    if not model_results:
        print("No models loaded. Exiting.")
        return

    compare_models(
        model_results,
        args.output_dir,
        args.metric,
    )


if __name__ == "__main__":
    main()
