"""Analyze how LLM action uncertainty relates to optimal branching factors."""

import argparse
import gc
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from papers.papers_code.reveng.src.reveng.analysis.behavioural_analysis.analysis_utils import (
    CellMetrics,
    ControlledAnalysisResult,
    DistanceToGoalMetrics,
    GridMetadata,
    UncertaintyAccuracyMetrics,
    compute_distance_summary,
    compute_distance_to_goal_metrics,
    compute_optimal_mass,
    compute_selective_prediction_curve,
    compute_stratified_summary,
    compute_uncertainty_accuracy_metrics,
    cross_entropy,
    discover_metadata_files,
    distribution_from_logprobs,
    jensen_shannon_divergence,
    load_environments,
    load_metadata_batch,
    optimal_entropy,
    run_controlled_analysis,
    sanitize_label,
    shannon_entropy,
)

# =============================================================================
# Data Classes (specific to this analysis)
# =============================================================================


@dataclass
class AnalysisResults:
    """Container for complete analysis results."""

    df: pd.DataFrame
    summary: pd.DataFrame
    correlations: dict[str, float]
    model_tag: str
    output_dir: Path
    metric: str = "ce"  # 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    # Controlled analysis results (controlling for grid_size and complexity)
    controlled_analysis: Optional[ControlledAnalysisResult] = None
    stratified_summary: Optional[pd.DataFrame] = None
    # Uncertainty-accuracy analysis
    uncertainty_accuracy: Optional[UncertaintyAccuracyMetrics] = None
    selective_prediction: Optional[pd.DataFrame] = None
    # Distance-to-goal analysis
    distance_metrics: Optional[DistanceToGoalMetrics] = None
    distance_summary: Optional[pd.DataFrame] = None


# =============================================================================
# Batching Utilities
# =============================================================================


def batch_metadata_files(
    metadata_files: list[Path], batch_size: int
) -> Iterator[list[Path]]:
    """Yield batches of metadata files."""
    for i in range(0, len(metadata_files), batch_size):
        yield metadata_files[i : i + batch_size]


# =============================================================================
# Core Analysis - Grid Processing
# =============================================================================


def process_all_grids(
    grids_dataset: dict[str, Any],
    metadata_files: list[Path],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Process all grids in batches and collect cell metrics."""
    dataset_keys = set(grids_dataset.keys())
    total_batches = (len(metadata_files) + batch_size - 1) // batch_size
    all_metrics: list[dict[str, Any]] = []

    for batch_idx, batch_files in enumerate(
        batch_metadata_files(metadata_files, batch_size)
    ):
        print(
            f"\n   Batch {batch_idx + 1}/{total_batches}: "
            f"loading {len(batch_files)} metadata files..."
        )

        metadata_batch = load_metadata_batch(batch_files, show_progress=True)
        common_keys = sorted(dataset_keys & set(metadata_batch.keys()))

        for key in tqdm(
            common_keys,
            desc=f"Analyzing grids (batch {batch_idx + 1}/{total_batches})",
            leave=False,
        ):
            env = grids_dataset[key]
            grid_metrics = _process_grid_for_analysis(key, env, metadata_batch[key])
            all_metrics.extend([m.to_dict() for m in grid_metrics])

        # Free memory after each batch
        metadata_batch.clear()
        gc.collect()

    return all_metrics


def _process_grid_for_analysis(
    grid_id: str, env: Any, metadata: GridMetadata
) -> list[CellMetrics]:
    """Process all cells in a grid and return their metrics.

    This is a local wrapper that uses the shared utilities.
    """
    from papers.papers_code.reveng.src.reveng.analysis.behavioural_analysis.analysis_utils import (
        ACTION_ID_TO_NAME,
        compute_optimal_actions_and_distances,
    )

    results: list[CellMetrics] = []
    optimal_actions, distance_grid = compute_optimal_actions_and_distances(env)

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue

            optimal_set = optimal_actions[y][x]
            if not optimal_set:
                continue

            dist = distribution_from_logprobs(cell.get("logprobs"))
            if dist is None:
                continue

            num_optimal = len(optimal_set)
            entropy_bits = shannon_entropy(dist)
            optimal_entropy_bits = optimal_entropy(num_optimal)
            cross_entropy_bits = cross_entropy(optimal_set, dist)
            jsd_value = jensen_shannon_divergence(optimal_set, dist)
            optimal_mass_val = compute_optimal_mass(optimal_set, dist)
            llm_action = cell.get("llm_response", -1)

            action_probs = {
                ACTION_ID_TO_NAME[aid].lower(): dist.get(aid, 0.0)
                for aid in ACTION_ID_TO_NAME
            }

            results.append(
                CellMetrics(
                    grid_id=grid_id,
                    grid_size=metadata.grid_size,
                    complexity=metadata.complexity,
                    instance_id=metadata.instance_id,
                    x=x,
                    y=y,
                    llm_action=llm_action,
                    num_optimal_actions=num_optimal,
                    entropy_bits=entropy_bits,
                    optimal_entropy_bits=optimal_entropy_bits,
                    cross_entropy_bits=cross_entropy_bits,
                    jsd=jsd_value,
                    optimal_mass=optimal_mass_val,
                    is_action_optimal=int(llm_action in optimal_set),
                    action_probs=action_probs,
                    distance_to_goal=distance_grid[y][x],
                )
            )

    return results


# =============================================================================
# Statistical Analysis
# =============================================================================


def compute_summary_statistics(df: pd.DataFrame, metric: str = "ce") -> pd.DataFrame:
    """Compute summary statistics grouped by number of optimal actions.

    Args:
        df: DataFrame with cell metrics
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"

    return (
        df.groupby("num_optimal_actions")
        .agg(
            samples=("entropy_bits", "count"),
            mean_entropy=("entropy_bits", "mean"),
            std_entropy=("entropy_bits", "std"),
            mean_optimal_entropy=("optimal_entropy_bits", "mean"),
            std_optimal_entropy=("optimal_entropy_bits", "std"),
            mean_divergence=(divergence_col, "mean"),
            std_divergence=(divergence_col, "std"),
            mean_optimal_mass=("optimal_mass", "mean"),
        )
        .reset_index()
    )


def compute_correlations(df: pd.DataFrame, metric: str = "ce") -> dict[str, float]:
    """Compute correlations between number of optimal actions and metrics.

    Args:
        df: DataFrame with cell metrics
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"

    return {
        "entropy": df["num_optimal_actions"].corr(df["entropy_bits"]),
        "divergence": df["num_optimal_actions"].corr(df[divergence_col]),
        "optimal_mass": df["num_optimal_actions"].corr(df["optimal_mass"]),
    }


def compute_controlled_analysis_for_uncertainty(
    df: pd.DataFrame, metric: str = "ce"
) -> ControlledAnalysisResult:
    """Run controlled analysis for uncertainty metrics.

    Computes correlations between num_optimal_actions and uncertainty metrics
    while controlling for grid_size and complexity.

    Args:
        df: DataFrame with cell metrics
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence

    Returns:
        ControlledAnalysisResult with raw, within-stratum, and partial correlations
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"
    y_cols = ["entropy_bits", divergence_col, "optimal_mass"]
    control_cols = ["grid_size", "complexity"]

    return run_controlled_analysis(
        df=df,
        x_col="num_optimal_actions",
        y_cols=y_cols,
        control_cols=control_cols,
        min_samples=30,
        min_stratum_size=10,
    )


def compute_stratified_uncertainty_summary(
    df: pd.DataFrame, metric: str = "ce"
) -> pd.DataFrame:
    """Compute summary statistics stratified by grid_size, complexity, and num_optimal.

    Args:
        df: DataFrame with cell metrics
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence

    Returns:
        DataFrame with stratified summary statistics
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"

    agg_config = {
        "samples": ("entropy_bits", "count"),
        "mean_entropy": ("entropy_bits", "mean"),
        "std_entropy": ("entropy_bits", "std"),
        "mean_divergence": (divergence_col, "mean"),
        "std_divergence": (divergence_col, "std"),
        "mean_optimal_mass": ("optimal_mass", "mean"),
        "std_optimal_mass": ("optimal_mass", "std"),
    }

    return compute_stratified_summary(
        df=df,
        strata_cols=["grid_size", "complexity", "num_optimal_actions"],
        agg_config=agg_config,
    )


# =============================================================================
# Visualization
# =============================================================================


def plot_heatmaps(df: pd.DataFrame, output_path: Path, metric: str = "ce") -> None:
    """Plot heatmaps of entropy and divergence by grid size and complexity.

    Args:
        df: DataFrame with cell metrics
        output_path: Path to save the figure
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"
    metric_name = "Cross-Entropy" if metric == "ce" else "JSD"
    metric_label = "Mean Cross-Entropy (bits)" if metric == "ce" else "Mean JSD"

    grouped = (
        df.groupby(["grid_size", "complexity"])
        .agg(
            mean_entropy=("entropy_bits", "mean"),
            mean_divergence=(divergence_col, "mean"),
        )
        .reset_index()
    )

    pivot_entropy = grouped.pivot(
        index="complexity", columns="grid_size", values="mean_entropy"
    )
    pivot_divergence = grouped.pivot(
        index="complexity", columns="grid_size", values="mean_divergence"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        pivot_entropy,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Mean Entropy (bits)"},
    )
    axes[0].set_title(
        "Mean Entropy by Grid Size and Complexity",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_xlabel("Grid Size", fontsize=12)
    axes[0].set_ylabel("Complexity", fontsize=12)

    sns.heatmap(
        pivot_divergence,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        ax=axes[1],
        cbar_kws={"label": metric_label},
    )
    axes[1].set_title(
        f"Mean {metric_name} by Grid Size and Complexity",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Grid Size", fontsize=12)
    axes[1].set_ylabel("Complexity", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trends(df: pd.DataFrame, output_path: Path, metric: str = "ce") -> None:
    """Plot trend lines for entropy metrics vs complexity and grid size.

    Args:
        df: DataFrame with cell metrics
        output_path: Path to save the figure
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"
    metric_name = "Cross-Entropy" if metric == "ce" else "JSD"
    metric_ylabel = "Mean Cross-Entropy (bits)" if metric == "ce" else "Mean JSD"

    grouped = (
        df.groupby(["grid_size", "complexity"])
        .agg(
            mean_entropy=("entropy_bits", "mean"),
            mean_optimal_entropy=("optimal_entropy_bits", "mean"),
            mean_divergence=(divergence_col, "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    grid_sizes = sorted(df["grid_size"].unique())
    complexities = sorted(df["complexity"].unique())

    # LLM Entropy vs Complexity by Grid Size
    for grid_size in grid_sizes:
        subset = grouped[grouped["grid_size"] == grid_size]
        axes[0].plot(
            subset["complexity"],
            subset["mean_entropy"],
            marker="o",
            linewidth=2,
            label=f"Grid {grid_size}",
        )
    axes[0].set_title("LLM Entropy vs Complexity", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Complexity", fontsize=12)
    axes[0].set_ylabel("Mean LLM Entropy (bits)", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Optimal Entropy vs Complexity by Grid Size
    for grid_size in grid_sizes:
        subset = grouped[grouped["grid_size"] == grid_size]
        axes[1].plot(
            subset["complexity"],
            subset["mean_optimal_entropy"],
            marker="s",
            linewidth=2,
            label=f"Grid {grid_size}",
        )
    axes[1].set_title("Optimal Entropy vs Complexity", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Complexity", fontsize=12)
    axes[1].set_ylabel("Mean Optimal Entropy (bits)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Divergence vs Grid Size by Complexity
    for complexity in complexities:
        subset = grouped[grouped["complexity"] == complexity]
        axes[2].plot(
            subset["grid_size"],
            subset["mean_divergence"],
            marker="o",
            linewidth=2,
            label=f"Complexity {complexity:.2f}",
        )
    axes[2].set_title(f"{metric_name} vs Grid Size", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Grid Size", fontsize=12)
    axes[2].set_ylabel(metric_ylabel, fontsize=12)
    axes[2].legend(fontsize=9, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_controlled_trends(
    df: pd.DataFrame, output_path: Path, metric: str = "ce"
) -> None:
    """Plot divergence vs num_optimal_actions, faceted by grid_size.

    Shows the relationship between branching factor and uncertainty within
    each grid size, with separate lines for each complexity level.

    Args:
        df: DataFrame with cell metrics
        output_path: Path to save the figure
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"
    metric_name = "Cross-Entropy" if metric == "ce" else "JSD"
    metric_ylabel = "Mean Cross-Entropy (bits)" if metric == "ce" else "Mean JSD"

    # Aggregate by (grid_size, complexity, num_optimal_actions)
    grouped = (
        df.groupby(["grid_size", "complexity", "num_optimal_actions"])
        .agg(
            mean_divergence=(divergence_col, "mean"),
            std_divergence=(divergence_col, "std"),
            count=(divergence_col, "count"),
        )
        .reset_index()
    )

    grid_sizes = sorted(df["grid_size"].unique())
    n_grids = len(grid_sizes)
    fig, axes = plt.subplots(1, n_grids, figsize=(6 * n_grids, 5), sharey=True)

    if n_grids == 1:
        axes = [axes]

    complexities = sorted(df["complexity"].unique())
    colors = plt.cm.viridis(
        [i / max(len(complexities) - 1, 1) for i in range(len(complexities))]
    )

    for ax, grid_size in zip(axes, grid_sizes):
        subset = grouped[grouped["grid_size"] == grid_size]

        for i, complexity in enumerate(complexities):
            cx_subset = subset[subset["complexity"] == complexity]
            if len(cx_subset) > 0:
                ax.plot(
                    cx_subset["num_optimal_actions"],
                    cx_subset["mean_divergence"],
                    marker="o",
                    linewidth=2,
                    color=colors[i],
                    label=f"cx={complexity:.2f}",
                )

        ax.set_title(f"Grid Size {grid_size}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Optimal Actions", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[0].set_ylabel(metric_ylabel, fontsize=11)

    fig.suptitle(
        f"{metric_name} vs Branching Factor (Controlled by Grid Size)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_comparison(
    controlled_result: ControlledAnalysisResult,
    output_path: Path,
    metric: str = "ce",
) -> None:
    """Plot comparison of raw vs controlled correlations.

    Shows how correlations change when controlling for grid_size and complexity.

    Args:
        controlled_result: ControlledAnalysisResult from controlled analysis
        output_path: Path to save the figure
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    """
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"
    metric_name = "Cross-Entropy" if metric == "ce" else "JSD"

    # Collect correlations for each metric
    metrics_to_plot = ["entropy_bits", divergence_col, "optimal_mass"]
    metric_labels = ["LLM Entropy", metric_name, "Optimal Mass"]

    raw_rs = []
    within_rs = []
    partial_rs = []

    for m in metrics_to_plot:
        raw_rs.append(
            controlled_result.raw_correlations.get(m, type("", (), {"r": 0})()).r
            if m in controlled_result.raw_correlations
            else 0
        )
        within_rs.append(
            controlled_result.within_stratum_correlations.get(
                m, type("", (), {"r": 0})()
            ).r
            if m in controlled_result.within_stratum_correlations
            else 0
        )
        partial_rs.append(
            controlled_result.partial_correlations.get(m, type("", (), {"r": 0})()).r
            if m in controlled_result.partial_correlations
            else 0
        )

    # Fix: properly extract r values
    raw_rs = []
    within_rs = []
    partial_rs = []
    for m in metrics_to_plot:
        raw_rs.append(
            controlled_result.raw_correlations[m].r
            if m in controlled_result.raw_correlations
            else 0
        )
        within_rs.append(
            controlled_result.within_stratum_correlations[m].r
            if m in controlled_result.within_stratum_correlations
            else 0
        )
        partial_rs.append(
            controlled_result.partial_correlations[m].r
            if m in controlled_result.partial_correlations
            else 0
        )

    x = range(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], raw_rs, width, label="Raw", color="steelblue")
    ax.bar(x, within_rs, width, label="Within-Stratum", color="darkorange")
    ax.bar(
        [i + width for i in x], partial_rs, width, label="Partial", color="forestgreen"
    )

    ax.set_ylabel("Correlation (r)", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_title(
        "Correlation with # Optimal Actions:\nRaw vs Controlled",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_distance(
    distance_summary: pd.DataFrame,
    output_path: Path,
    metric: str = "jsd",
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """Plot entropy, divergence, and accuracy vs distance to goal.

    Args:
        distance_summary: DataFrame from compute_distance_summary
        output_path: Path to save figure
        metric: 'jsd' or 'ce'
        figsize: Figure size
    """
    metric_label = "JSD" if metric == "jsd" else "Cross-Entropy"

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Entropy vs distance
    axes[0].errorbar(
        distance_summary["distance_to_goal"],
        distance_summary["mean_entropy"],
        yerr=distance_summary["std_entropy"],
        marker="o",
        capsize=3,
        color="steelblue",
    )
    axes[0].set_xlabel("Distance to Goal", fontsize=10)
    axes[0].set_ylabel("Mean Entropy (bits)", fontsize=10)
    axes[0].set_title("Entropy vs Distance", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Divergence vs distance
    axes[1].plot(
        distance_summary["distance_to_goal"],
        distance_summary["mean_divergence"],
        marker="o",
        linewidth=2,
        color="darkorange",
    )
    axes[1].set_xlabel("Distance to Goal", fontsize=10)
    axes[1].set_ylabel(f"Mean {metric_label}", fontsize=10)
    axes[1].set_title(f"{metric_label} vs Distance", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Accuracy vs distance
    axes[2].plot(
        distance_summary["distance_to_goal"],
        distance_summary["accuracy"],
        marker="o",
        linewidth=2,
        color="forestgreen",
    )
    axes[2].set_xlabel("Distance to Goal", fontsize=10)
    axes[2].set_ylabel("Accuracy", fontsize=10)
    axes[2].set_title("Accuracy vs Distance", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_visualizations(results: AnalysisResults) -> dict[str, Path]:
    """Generate all visualizations for the analysis results.

    Returns:
        Dictionary mapping visualization names to their file paths
    """
    model_dir = results.output_dir / results.model_tag
    metric_suffix = f"_{results.metric}" if results.metric != "ce" else ""

    paths: dict[str, Path] = {}

    # Original visualizations
    paths["heatmaps"] = (
        model_dir / f"uncertainty_heatmaps_{results.model_tag}{metric_suffix}.png"
    )
    paths["trends"] = (
        model_dir / f"uncertainty_trends_{results.model_tag}{metric_suffix}.png"
    )

    plot_heatmaps(results.df, paths["heatmaps"], metric=results.metric)
    plot_trends(results.df, paths["trends"], metric=results.metric)

    # Controlled analysis visualizations
    paths["controlled_trends"] = (
        model_dir
        / f"uncertainty_controlled_trends_{results.model_tag}{metric_suffix}.png"
    )
    plot_controlled_trends(
        results.df, paths["controlled_trends"], metric=results.metric
    )

    if results.controlled_analysis is not None:
        paths["correlation_comparison"] = (
            model_dir
            / f"uncertainty_correlation_comparison_{results.model_tag}{metric_suffix}.png"
        )
        plot_correlation_comparison(
            results.controlled_analysis,
            paths["correlation_comparison"],
            metric=results.metric,
        )

    # Distance-to-goal visualizations
    if results.distance_summary is not None and len(results.distance_summary) > 0:
        paths["metrics_by_distance"] = (
            model_dir / f"metrics_by_distance_{results.model_tag}{metric_suffix}.png"
        )
        plot_metrics_by_distance(
            results.distance_summary,
            paths["metrics_by_distance"],
            metric=results.metric,
        )

    return paths


# =============================================================================
# Output and Reporting
# =============================================================================


def save_csv_outputs(
    df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, model_tag: str
) -> tuple[Path, Path]:
    """Save analysis results to CSV files."""
    states_path = output_dir / f"uncertainty_states_{model_tag}.csv"
    df.to_csv(states_path, index=False)

    summary_path = output_dir / f"uncertainty_summary_{model_tag}.csv"
    summary.to_csv(summary_path, index=False)

    return states_path, summary_path


def save_findings(
    summary: pd.DataFrame,
    correlations: dict[str, float],
    output_dir: Path,
    model_tag: str,
    metric: str = "ce",
    controlled_analysis: Optional[ControlledAnalysisResult] = None,
) -> Path:
    """Save key findings to a text file.

    Args:
        summary: Summary statistics DataFrame
        correlations: Correlation values
        output_dir: Output directory
        model_tag: Model identifier
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
        controlled_analysis: Optional controlled analysis results
    """
    metric_name = "Cross-Entropy" if metric == "ce" else "JSD"
    metric_abbrev = "Cross-H" if metric == "ce" else "JSD"
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"
    unit = " bits" if metric == "ce" else ""

    lines = [
        "KEY FINDINGS",
        f"Model tag: {model_tag}",
        f"Divergence metric: {metric_name}",
        "",
        "=" * 60,
        "RAW CORRELATIONS (unadjusted):",
        "=" * 60,
        f"  #optimal vs entropy: {correlations['entropy']:.4f}",
        f"  #optimal vs {metric_name.lower()}: {correlations['divergence']:.4f}",
        f"  #optimal vs optimal mass: {correlations['optimal_mass']:.4f}",
    ]

    # Add controlled analysis if available
    if controlled_analysis is not None:
        lines.extend(
            [
                "",
                "=" * 60,
                "CONTROLLED ANALYSIS (controlling for grid_size and complexity):",
                "=" * 60,
                "",
                "Within-Stratum Correlations (averaged across strata):",
            ]
        )
        for col, corr in controlled_analysis.within_stratum_correlations.items():
            col_label = (
                "entropy"
                if col == "entropy_bits"
                else (metric_name.lower() if col == divergence_col else "optimal_mass")
            )
            lines.append(f"  #optimal vs {col_label}: r={corr.r:.4f} (n={corr.n})")

        lines.extend(
            [
                "",
                "Partial Correlations (residualized):",
            ]
        )
        for col, corr in controlled_analysis.partial_correlations.items():
            col_label = (
                "entropy"
                if col == "entropy_bits"
                else (metric_name.lower() if col == divergence_col else "optimal_mass")
            )
            p_str = f", p={corr.p_value:.4f}" if corr.p_value is not None else ""
            sig = " *" if corr.p_value is not None and corr.p_value < 0.05 else ""
            lines.append(f"  #optimal vs {col_label}: r={corr.r:.4f}{p_str}{sig}")

        if controlled_analysis.regression is not None:
            reg = controlled_analysis.regression
            lines.extend(
                [
                    "",
                    f"OLS Regression ({divergence_col} ~ num_optimal + grid_size + complexity):",
                    f"  R² = {reg.r_squared:.4f}, Adj R² = {reg.adj_r_squared:.4f}, n = {reg.n}",
                    "  Coefficients:",
                ]
            )
            for var, coef in reg.coefficients.items():
                p_val = reg.p_values.get(var)
                p_str = f", p={p_val:.4f}" if p_val is not None else ""
                sig = " *" if p_val is not None and p_val < 0.05 else ""
                lines.append(f"    {var}: {coef:.4f}{p_str}{sig}")

    lines.extend(
        [
            "",
            "=" * 60,
            "MEAN ENTROPY BY NUMBER OF OPTIMAL ACTIONS:",
            "=" * 60,
        ]
    )

    for _, row in summary.iterrows():
        lines.append(
            f"  {int(row['num_optimal_actions'])} optimal actions -> "
            f"H_llm={row['mean_entropy']:.3f} ± {row['std_entropy']:.3f}, "
            f"H_opt={row['mean_optimal_entropy']:.3f} ± {row['std_optimal_entropy']:.3f}, "
            f"{metric_abbrev}={row['mean_divergence']:.3f} ± {row['std_divergence']:.3f}{unit}"
        )

    metric_suffix = f"_{metric}" if metric != "ce" else ""
    findings_path = output_dir / f"uncertainty_findings_{model_tag}{metric_suffix}.txt"
    findings_path.write_text("\n".join(lines) + "\n")
    return findings_path


def print_summary(
    correlations: dict[str, float],
    summary: pd.DataFrame,
    model_tag: str,
    metric: str = "ce",
    controlled_analysis: Optional[ControlledAnalysisResult] = None,
) -> None:
    """Print summary findings to console.

    Args:
        correlations: Correlation values
        summary: Summary statistics DataFrame
        model_tag: Model identifier
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
        controlled_analysis: Optional controlled analysis results
    """
    metric_name = "Cross-Entropy" if metric == "ce" else "JSD"
    metric_abbrev = "Cross-H" if metric == "ce" else "JSD"
    divergence_col = "cross_entropy_bits" if metric == "ce" else "jsd"
    unit = " bits" if metric == "ce" else ""

    print("\n5. KEY FINDINGS:")
    print(f"   Model: {model_tag}")
    print(f"   Metric: {metric_name}")
    print("\n   RAW CORRELATIONS:")
    print(f"   - #optimal vs entropy: {correlations['entropy']:.4f}")
    print(f"   - #optimal vs {metric_name.lower()}: {correlations['divergence']:.4f}")
    print(f"   - #optimal vs optimal mass: {correlations['optimal_mass']:.4f}")

    if controlled_analysis is not None:
        print("\n   CONTROLLED CORRELATIONS (grid_size, complexity):")
        print("   Within-Stratum:")
        for col, corr in controlled_analysis.within_stratum_correlations.items():
            col_label = (
                "entropy"
                if col == "entropy_bits"
                else (metric_name.lower() if col == divergence_col else "optimal_mass")
            )
            print(f"   - #optimal vs {col_label}: r={corr.r:.4f}")

        print("   Partial (residualized):")
        for col, corr in controlled_analysis.partial_correlations.items():
            col_label = (
                "entropy"
                if col == "entropy_bits"
                else (metric_name.lower() if col == divergence_col else "optimal_mass")
            )
            sig = "*" if corr.p_value is not None and corr.p_value < 0.05 else ""
            print(f"   - #optimal vs {col_label}: r={corr.r:.4f}{sig}")

    print("\n   Mean entropy by # optimal actions:")

    for _, row in summary.iterrows():
        print(
            f"      {int(row['num_optimal_actions'])} optimal actions -> "
            f"H_llm={row['mean_entropy']:.3f} ± {row['std_entropy']:.3f}, "
            f"H_opt={row['mean_optimal_entropy']:.3f} ± {row['std_optimal_entropy']:.3f}, "
            f"{metric_abbrev}={row['mean_divergence']:.3f}{unit}"
        )


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def analyze_uncertainty(
    dataset_path: str,
    metadata_dir: str,
    output_dir: str,
    batch_size: int = 100,
    metric: str = "ce",
) -> Optional[AnalysisResults]:
    """Run the complete uncertainty analysis pipeline.

    This is the main entry point that orchestrates:
    1. Loading environments and metadata
    2. Computing optimal actions and uncertainty metrics
    3. Statistical analysis
    4. Saving results and generating visualizations

    Args:
        dataset_path: Path to the grids pickle file
        metadata_dir: Directory containing LLM policy metadata JSON files
        output_dir: Directory to save analysis outputs
        batch_size: Number of metadata grids to load into memory at once
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
    """
    metric_name = "Cross-Entropy" if metric == "ce" else "Jensen-Shannon Divergence"

    print("\n" + "=" * 80)
    print(f"UNCERTAINTY ANALYSIS (Metric: {metric_name})")
    print("=" * 80)

    # Step 1: Load environments
    print("\n1. Loading environments...")
    grids_dataset = load_environments(dataset_path)
    print(f"   Loaded {len(grids_dataset)} environments")

    # Step 2: Discover metadata files
    metadata_path = Path(metadata_dir)
    metadata_files = discover_metadata_files(metadata_path)
    model_tag = sanitize_label(metadata_path.name)

    print(f"\n2. Found {len(metadata_files)} metadata files in {metadata_path}")
    if not metadata_files:
        print("   No metadata files detected, aborting.")
        return None

    # Step 3: Process all grids in batches
    total_batches = (len(metadata_files) + batch_size - 1) // batch_size
    print(
        f"\n3. Processing {len(metadata_files)} grids in "
        f"{total_batches} batches of {batch_size}..."
    )

    metrics_dicts = process_all_grids(grids_dataset, metadata_files, batch_size)

    if not metrics_dicts:
        print("   No states with usable logprob information were found.")
        return None

    # Step 4: Build DataFrame and compute statistics
    df = pd.DataFrame(metrics_dicts)
    summary = compute_summary_statistics(df, metric=metric)
    correlations = compute_correlations(df, metric=metric)

    # Step 4b: Controlled analysis (controlling for grid_size and complexity)
    print("\n4. Running controlled analysis...")
    controlled_analysis = compute_controlled_analysis_for_uncertainty(df, metric=metric)
    stratified_summary = compute_stratified_uncertainty_summary(df, metric=metric)
    print("   Computed within-stratum and partial correlations")

    # Step 4c: Uncertainty-accuracy analysis
    print("\n   Computing uncertainty-accuracy metrics...")
    divergence_col = "jsd" if metric == "jsd" else "cross_entropy_bits"
    uncertainty_accuracy = compute_uncertainty_accuracy_metrics(df)
    selective_prediction = compute_selective_prediction_curve(df)
    auroc_str = (
        f"{uncertainty_accuracy.auroc:.3f}" if uncertainty_accuracy.auroc else "N/A"
    )
    print(
        f"   Accuracy: {uncertainty_accuracy.accuracy:.3f}, "
        f"AUROC: {auroc_str}, "
        f"ECE: {uncertainty_accuracy.ece:.3f}"
    )

    # Step 4d: Distance-to-goal analysis
    print("\n   Computing distance-to-goal metrics...")
    distance_metrics = compute_distance_to_goal_metrics(
        df, divergence_col=divergence_col
    )
    distance_summary = compute_distance_summary(df, divergence_col=divergence_col)
    print(
        f"   Corr(entropy, distance): {distance_metrics.correlation_entropy_distance:.3f}, "
        f"Corr(accuracy, distance): {distance_metrics.correlation_accuracy_distance:.3f}"
    )

    # Step 5: Setup output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_output_dir = output_path / model_tag
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 6: Save CSV outputs
    states_path, summary_path = save_csv_outputs(
        df, summary, model_output_dir, model_tag
    )
    print(f"\n5. Saved per-state metrics to: {states_path}")
    print(f"   Saved summary statistics to: {summary_path}")

    # Save stratified summary
    stratified_path = model_output_dir / f"uncertainty_stratified_{model_tag}.csv"
    stratified_summary.to_csv(stratified_path, index=False)
    print(f"   Saved stratified summary to: {stratified_path}")

    # Step 7: Print and save findings
    print_summary(
        correlations,
        summary,
        model_tag,
        metric=metric,
        controlled_analysis=controlled_analysis,
    )
    findings_path = save_findings(
        summary,
        correlations,
        model_output_dir,
        model_tag,
        metric=metric,
        controlled_analysis=controlled_analysis,
    )
    print(f"   Saved findings summary to: {findings_path}")

    # Step 8: Generate visualizations
    print("\n7. Generating visualizations...")
    results = AnalysisResults(
        df=df,
        summary=summary,
        correlations=correlations,
        model_tag=model_tag,
        output_dir=output_path,
        metric=metric,
        controlled_analysis=controlled_analysis,
        stratified_summary=stratified_summary,
        uncertainty_accuracy=uncertainty_accuracy,
        selective_prediction=selective_prediction,
        distance_metrics=distance_metrics,
        distance_summary=distance_summary,
    )
    viz_paths = generate_visualizations(results)
    print(f"   Saved {len(viz_paths)} plots to: {model_output_dir}")

    print("\n" + "=" * 80)
    print("UNCERTAINTY ANALYSIS COMPLETE")
    print("=" * 80)

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze how agent entropy aligns with optimal branching factors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Path to the baseline grids pickle file",
    )

    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="/Users/niall/Downloads/together_ai_openai_gpt-oss-20b",
        help="Directory containing LLM policy metadata JSON files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/analysis",
        help="Directory to save analysis outputs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of metadata grids to load into memory at once",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["ce", "jsd"],
        default="ce",
        help="Divergence metric: 'ce' (cross-entropy) or 'jsd' (Jensen-Shannon, bounded [0,1])",
    )

    args = parser.parse_args()

    analyze_uncertainty(
        dataset_path=args.dataset,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
