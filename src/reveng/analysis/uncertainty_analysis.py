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

from reveng.analysis.analysis_utils import (
    CellMetrics,
    GridMetadata,
    compute_optimal_actions,
    compute_optimal_mass,
    cross_entropy,
    discover_metadata_files,
    distribution_from_logprobs,
    load_environments,
    load_metadata_batch,
    optimal_entropy,
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
    from reveng.analysis.analysis_utils import (
        ACTION_ID_TO_NAME,
    )

    results: list[CellMetrics] = []
    optimal_actions = compute_optimal_actions(env)

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
                    optimal_mass=optimal_mass_val,
                    is_action_optimal=int(llm_action in optimal_set),
                    action_probs=action_probs,
                )
            )

    return results


# =============================================================================
# Statistical Analysis
# =============================================================================


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by number of optimal actions."""
    return (
        df.groupby("num_optimal_actions")
        .agg(
            samples=("entropy_bits", "count"),
            mean_entropy=("entropy_bits", "mean"),
            std_entropy=("entropy_bits", "std"),
            mean_optimal_entropy=("optimal_entropy_bits", "mean"),
            std_optimal_entropy=("optimal_entropy_bits", "std"),
            mean_cross_entropy=("cross_entropy_bits", "mean"),
            std_cross_entropy=("cross_entropy_bits", "std"),
            mean_optimal_mass=("optimal_mass", "mean"),
        )
        .reset_index()
    )


def compute_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Compute correlations between number of optimal actions and metrics."""
    return {
        "entropy": df["num_optimal_actions"].corr(df["entropy_bits"]),
        "cross_entropy": df["num_optimal_actions"].corr(df["cross_entropy_bits"]),
        "optimal_mass": df["num_optimal_actions"].corr(df["optimal_mass"]),
    }


# =============================================================================
# Visualization
# =============================================================================


def plot_heatmaps(df: pd.DataFrame, output_path: Path) -> None:
    """Plot heatmaps of entropy and cross-entropy by grid size and complexity."""
    grouped = (
        df.groupby(["grid_size", "complexity"])
        .agg(
            mean_entropy=("entropy_bits", "mean"),
            mean_cross_entropy=("cross_entropy_bits", "mean"),
        )
        .reset_index()
    )

    pivot_entropy = grouped.pivot(
        index="complexity", columns="grid_size", values="mean_entropy"
    )
    pivot_cross = grouped.pivot(
        index="complexity", columns="grid_size", values="mean_cross_entropy"
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
        pivot_cross,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        ax=axes[1],
        cbar_kws={"label": "Mean Cross-Entropy (bits)"},
    )
    axes[1].set_title(
        "Mean Cross-Entropy by Grid Size and Complexity",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Grid Size", fontsize=12)
    axes[1].set_ylabel("Complexity", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trends(df: pd.DataFrame, output_path: Path) -> None:
    """Plot trend lines for entropy metrics vs complexity and grid size."""
    grouped = (
        df.groupby(["grid_size", "complexity"])
        .agg(
            mean_entropy=("entropy_bits", "mean"),
            mean_optimal_entropy=("optimal_entropy_bits", "mean"),
            mean_cross_entropy=("cross_entropy_bits", "mean"),
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

    # Cross-Entropy vs Grid Size by Complexity
    for complexity in complexities:
        subset = grouped[grouped["complexity"] == complexity]
        axes[2].plot(
            subset["grid_size"],
            subset["mean_cross_entropy"],
            marker="o",
            linewidth=2,
            label=f"Complexity {complexity:.2f}",
        )
    axes[2].set_title("Cross-Entropy vs Grid Size", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Grid Size", fontsize=12)
    axes[2].set_ylabel("Mean Cross-Entropy (bits)", fontsize=12)
    axes[2].legend(fontsize=9, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_visualizations(results: AnalysisResults) -> tuple[Path, Path]:
    """Generate all visualizations for the analysis results."""
    model_dir = results.output_dir / results.model_tag
    heatmap_path = model_dir / f"uncertainty_heatmaps_{results.model_tag}.png"
    trends_path = model_dir / f"uncertainty_trends_{results.model_tag}.png"

    plot_heatmaps(results.df, heatmap_path)
    plot_trends(results.df, trends_path)

    return heatmap_path, trends_path


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
) -> Path:
    """Save key findings to a text file."""
    lines = [
        "KEY FINDINGS",
        f"Model tag: {model_tag}",
        "",
        "CORRELATIONS:",
        f"  #optimal vs entropy: {correlations['entropy']:.4f}",
        f"  #optimal vs cross-entropy: {correlations['cross_entropy']:.4f}",
        f"  #optimal vs optimal mass: {correlations['optimal_mass']:.4f}",
        "",
        "MEAN ENTROPY BY NUMBER OF OPTIMAL ACTIONS:",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"  {int(row['num_optimal_actions'])} optimal actions -> "
            f"H_llm={row['mean_entropy']:.3f} ± {row['std_entropy']:.3f}, "
            f"H_opt={row['mean_optimal_entropy']:.3f} ± {row['std_optimal_entropy']:.3f}, "
            f"Cross-H={row['mean_cross_entropy']:.3f} ± {row['std_cross_entropy']:.3f} bits"
        )

    findings_path = output_dir / f"uncertainty_findings_{model_tag}.txt"
    findings_path.write_text("\n".join(lines) + "\n")
    return findings_path


def print_summary(
    correlations: dict[str, float], summary: pd.DataFrame, model_tag: str
) -> None:
    """Print summary findings to console."""
    print("\n5. KEY FINDINGS:")
    print(f"   Model: {model_tag}")
    print(f"   - Correlation (#optimal vs entropy): {correlations['entropy']:.4f}")
    print(
        f"   - Correlation (#optimal vs cross-entropy): "
        f"{correlations['cross_entropy']:.4f}"
    )
    print(
        f"   - Correlation (#optimal vs optimal mass): "
        f"{correlations['optimal_mass']:.4f}"
    )
    print("   - Mean entropy by # optimal actions:")

    for _, row in summary.iterrows():
        print(
            f"      {int(row['num_optimal_actions'])} optimal actions -> "
            f"H_llm={row['mean_entropy']:.3f} ± {row['std_entropy']:.3f}, "
            f"H_opt={row['mean_optimal_entropy']:.3f} ± {row['std_optimal_entropy']:.3f}, "
            f"Cross-H={row['mean_cross_entropy']:.3f} bits"
        )


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def analyze_uncertainty(
    dataset_path: str,
    metadata_dir: str,
    output_dir: str,
    batch_size: int = 100,
) -> Optional[AnalysisResults]:
    """Run the complete uncertainty analysis pipeline.

    This is the main entry point that orchestrates:
    1. Loading environments and metadata
    2. Computing optimal actions and uncertainty metrics
    3. Statistical analysis
    4. Saving results and generating visualizations
    """

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
    summary = compute_summary_statistics(df)
    correlations = compute_correlations(df)

    # Step 5: Setup output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_output_dir = output_path / model_tag
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 6: Save CSV outputs
    states_path, summary_path = save_csv_outputs(
        df, summary, model_output_dir, model_tag
    )
    print(f"\n4. Saved per-state metrics to: {states_path}")
    print(f"   Saved summary statistics to: {summary_path}")

    # Step 7: Print and save findings
    print_summary(correlations, summary, model_tag)
    findings_path = save_findings(summary, correlations, model_output_dir, model_tag)
    print(f"   Saved findings summary to: {findings_path}")

    # Step 8: Generate visualizations
    print("\n6. Generating visualizations...")
    results = AnalysisResults(
        df=df,
        summary=summary,
        correlations=correlations,
        model_tag=model_tag,
        output_dir=output_path,
    )
    heatmap_path, trends_path = generate_visualizations(results)
    print(f"   Saved plots to: {heatmap_path} and {trends_path}")

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
        default=100,
        help="Number of metadata grids to load into memory at once",
    )

    args = parser.parse_args()

    analyze_uncertainty(
        dataset_path=args.dataset,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
