"""Compare uncertainty and error metrics across models with isotransform variants.

This module analyzes how different grid transformations (ReflectEnv, RotateEnv,
StartGoalSwap, TransposeEnv) affect model performance compared to baseline grids.

Key analyses:
- Metrics by distance to goal, faceted by transform type
- OLS regression: metric ~ transform_type + grid_size + complexity
- Cross-model comparison of transform effects
"""

import argparse
import gc
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from reveng.analysis.analysis_utils import (
    TRANSFORM_TYPES,
    IsotransformGridMetadata,
    TransformRegressionResult,
    compute_distance_summary_by_transform,
    compute_optimal_actions_and_distances,
    compute_optimal_mass,
    compute_transform_summary,
    cross_entropy,
    distribution_from_logprobs,
    jensen_shannon_divergence,
    load_environments,
    load_metadata_batch_isotransform,
    optimal_entropy,
    run_transform_regression,
    sanitize_label,
    shannon_entropy,
)

# =============================================================================
# Paper-Ready Plot Configuration
# =============================================================================

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

# Colorblind-friendly palette for transforms
TRANSFORM_COLORS = {
    "baseline": "#000000",  # Black
    "ReflectEnv": "#0072B2",  # Blue
    "RotateEnv": "#D55E00",  # Vermillion
    "StartGoalSwap": "#009E73",  # Bluish green
    "TransposeEnv": "#CC79A7",  # Reddish purple
}

MODEL_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
]


def setup_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(PAPER_RC)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class IsotransformModelResults:
    """Results for a single model with isotransform data."""

    model_name: str
    df: pd.DataFrame  # Per-cell metrics with transform_type column
    transform_summary: pd.DataFrame
    distance_by_transform: pd.DataFrame
    regression_results: dict[str, TransformRegressionResult]  # outcome -> result

    @property
    def n_samples(self) -> int:
        return len(self.df)

    @property
    def transforms(self) -> list[str]:
        return sorted(self.df["transform_type"].unique())


@dataclass
class IsotransformComparisonResults:
    """Container for cross-model isotransform comparison."""

    model_results: dict[str, IsotransformModelResults]
    regression_summary: pd.DataFrame  # Coefficients across models
    output_dir: Path


# =============================================================================
# Saving and Loading Cached Results
# =============================================================================


def save_model_results(
    model_results: IsotransformModelResults,
    output_dir: Path,
) -> Path:
    """Save model results to disk for later loading.

    Creates a directory for the model with:
    - isotransform_states_{model_name}.csv (full per-cell data)
    - transform_summary.csv
    - distance_by_transform.csv

    Args:
        model_results: Results to save
        output_dir: Parent output directory

    Returns:
        Path to the model's output directory
    """
    model_dir = output_dir / model_results.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save full per-cell data
    states_path = model_dir / f"isotransform_states_{model_results.model_name}.csv"
    model_results.df.to_csv(states_path, index=False)
    print(f"   Saved states: {states_path}")

    # Save summaries
    model_results.transform_summary.to_csv(
        model_dir / "transform_summary.csv", index=False
    )
    model_results.distance_by_transform.to_csv(
        model_dir / "distance_by_transform.csv", index=False
    )

    return model_dir


def load_model_from_cache(
    model_dir: Path,
    model_name: Optional[str] = None,
) -> IsotransformModelResults:
    """Load model results from cached CSV files.

    Args:
        model_dir: Directory containing the model's cached files
        model_name: Optional model name override

    Returns:
        IsotransformModelResults loaded from disk
    """
    if model_name is None:
        model_name = sanitize_label(model_dir.name)

    # Find the states CSV
    states_files = list(model_dir.glob("isotransform_states_*.csv"))
    if not states_files:
        raise ValueError(f"No isotransform_states_*.csv found in {model_dir}")

    df = pd.read_csv(states_files[0])

    # Load summaries if they exist, otherwise recompute
    transform_summary_path = model_dir / "transform_summary.csv"
    if transform_summary_path.exists():
        transform_summary = pd.read_csv(transform_summary_path)
    else:
        transform_summary = compute_transform_summary(
            df, metrics=["entropy_bits", "jsd", "is_action_optimal"]
        )

    distance_by_transform_path = model_dir / "distance_by_transform.csv"
    if distance_by_transform_path.exists():
        distance_by_transform = pd.read_csv(distance_by_transform_path)
    else:
        distance_by_transform = compute_distance_summary_by_transform(df)

    # Recompute regression results (cheap operation)
    regression_results = {}
    for outcome in ["is_action_optimal", "entropy_bits", "jsd"]:
        if outcome in df.columns:
            result = run_transform_regression(df, outcome)
            if result:
                regression_results[outcome] = result

    return IsotransformModelResults(
        model_name=model_name,
        df=df,
        transform_summary=transform_summary,
        distance_by_transform=distance_by_transform,
        regression_results=regression_results,
    )


def discover_cached_models(output_dir: Path) -> dict[str, Path]:
    """Discover cached model results in an output directory.

    Args:
        output_dir: Directory to search for cached results

    Returns:
        Dictionary mapping model names to their cache directories
    """
    cached = {}
    if not output_dir.exists():
        return cached

    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            states_files = list(subdir.glob("isotransform_states_*.csv"))
            if states_files:
                model_name = sanitize_label(subdir.name)
                cached[model_name] = subdir

    return cached


def load_models_from_cache(
    output_dir: Path,
    model_names: Optional[list[str]] = None,
) -> dict[str, IsotransformModelResults]:
    """Load all cached model results from output directory.

    Args:
        output_dir: Directory containing cached model subdirectories
        model_names: Optional list of specific models to load

    Returns:
        Dictionary mapping model names to results
    """
    cached_paths = discover_cached_models(output_dir)

    if model_names:
        cached_paths = {k: v for k, v in cached_paths.items() if k in model_names}

    results = {}
    for model_name, model_dir in cached_paths.items():
        print(f"Loading cached: {model_name}...")
        try:
            model_results = load_model_from_cache(model_dir, model_name)
            results[model_name] = model_results
            print(
                f"   Loaded {model_results.n_samples} samples, "
                f"{len(model_results.transforms)} transforms"
            )
        except Exception as e:
            print(f"   Error loading {model_name}: {e}")

    return results


# =============================================================================
# Data Loading
# =============================================================================


def batch_metadata_files(
    metadata_files: list[Path], batch_size: int
) -> Iterator[list[Path]]:
    """Yield batches of metadata files."""
    for i in range(0, len(metadata_files), batch_size):
        yield metadata_files[i : i + batch_size]


def discover_isotransform_model_directories(parent_dir: Path) -> list[Path]:
    """Discover model directories containing isotransform metadata.

    Args:
        parent_dir: Directory containing model subdirectories

    Returns:
        List of paths to model directories with isotransform metadata
    """
    model_dirs = []
    for subdir in sorted(parent_dir.iterdir()):
        if subdir.is_dir():
            # Check for isotransform metadata files (contain transform type in name)
            metadata_files = list(subdir.glob("*_metadata.json"))
            if metadata_files:
                # Check if any file has transform type pattern
                sample_name = metadata_files[0].stem
                parts = sample_name.split("_")
                if len(parts) >= 5:  # Has transform type
                    model_dirs.append(subdir)
    return model_dirs


def process_isotransform_grid(
    grid_id: str,
    env: Any,
    metadata: IsotransformGridMetadata,
) -> list[dict[str, Any]]:
    """Process a single isotransform grid and return cell metrics.

    Args:
        grid_id: Grid identifier
        env: Environment instance
        metadata: Isotransform grid metadata

    Returns:
        List of cell metric dictionaries
    """
    results = []
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

            results.append(
                {
                    "grid_id": grid_id,
                    "grid_size": metadata.grid_size,
                    "complexity": metadata.complexity,
                    "instance_id": metadata.instance_id,
                    "transform_type": metadata.transform_type,
                    "x": x,
                    "y": y,
                    "llm_action": llm_action,
                    "num_optimal_actions": num_optimal,
                    "entropy_bits": entropy_bits,
                    "optimal_entropy_bits": optimal_entropy_bits,
                    "cross_entropy_bits": cross_entropy_bits,
                    "jsd": jsd_value,
                    "optimal_mass": optimal_mass_val,
                    "is_action_optimal": int(llm_action in optimal_set),
                    "distance_to_goal": distance_grid[y][x],
                }
            )

    return results


def process_all_isotransform_grids(
    grids_dataset: dict[str, Any],
    metadata_files: list[Path],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Process all isotransform grids in batches and collect cell metrics.

    Args:
        grids_dataset: Pre-loaded grids dataset
        metadata_files: List of metadata file paths
        batch_size: Number of files to load per batch

    Returns:
        List of all cell metric dictionaries
    """
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

        metadata_batch = load_metadata_batch_isotransform(
            batch_files, show_progress=True
        )
        common_keys = sorted(dataset_keys & set(metadata_batch.keys()))

        for key in tqdm(
            common_keys,
            desc=f"Analyzing grids (batch {batch_idx + 1}/{total_batches})",
            leave=False,
        ):
            env = grids_dataset[key]
            grid_metrics = process_isotransform_grid(key, env, metadata_batch[key])
            all_metrics.extend(grid_metrics)

        # Free memory after each batch
        metadata_batch.clear()
        gc.collect()

    return all_metrics


def process_baseline_grids(
    baseline_dataset: dict[str, Any],
    metadata_files: list[Path],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Process baseline grids (no transform) in batches.

    Baseline files don't have a transform suffix in their filename.
    We use the standard metadata loader and assign transform_type='baseline'.

    Args:
        baseline_dataset: Pre-loaded baseline grids dataset
        metadata_files: List of baseline metadata file paths
        batch_size: Number of files to load per batch

    Returns:
        List of all cell metric dictionaries
    """
    from reveng.analysis.analysis_utils import load_metadata_batch

    dataset_keys = set(baseline_dataset.keys())
    total_batches = (len(metadata_files) + batch_size - 1) // batch_size
    all_metrics: list[dict[str, Any]] = []

    for batch_idx, batch_files in enumerate(
        batch_metadata_files(metadata_files, batch_size)
    ):
        print(
            f"\n   Baseline batch {batch_idx + 1}/{total_batches}: "
            f"loading {len(batch_files)} metadata files..."
        )

        # Use standard metadata loader (no transform in filename)
        metadata_batch = load_metadata_batch(batch_files, show_progress=True)
        common_keys = sorted(dataset_keys & set(metadata_batch.keys()))

        for key in tqdm(
            common_keys,
            desc=f"Analyzing baselines (batch {batch_idx + 1}/{total_batches})",
            leave=False,
        ):
            env = baseline_dataset[key]
            metadata = metadata_batch[key]

            # Create an IsotransformGridMetadata with transform_type='baseline'
            iso_metadata = IsotransformGridMetadata(
                grid_size=metadata.grid_size,
                complexity=metadata.complexity,
                instance_id=metadata.instance_id,
                transform_type="baseline",
                policy_metadata=metadata.policy_metadata,
            )
            grid_metrics = process_isotransform_grid(key, env, iso_metadata)
            all_metrics.extend(grid_metrics)

        # Free memory after each batch
        metadata_batch.clear()
        gc.collect()

    return all_metrics


def load_isotransform_model(
    isotransform_dir: Path,
    isotransform_dataset: dict[str, Any],
    model_name: Optional[str] = None,
    batch_size: int = 100,
    baseline_dir: Optional[Path] = None,
    baseline_dataset: Optional[dict[str, Any]] = None,
) -> IsotransformModelResults:
    """Load and process isotransform data for a single model.

    Args:
        isotransform_dir: Directory containing isotransform metadata files
        isotransform_dataset: Pre-loaded isotransform grids dataset
        model_name: Optional model name override
        batch_size: Number of metadata files to load per batch
        baseline_dir: Optional directory containing baseline metadata files
        baseline_dataset: Optional pre-loaded baseline grids dataset

    Returns:
        IsotransformModelResults for the model
    """
    if model_name is None:
        model_name = sanitize_label(isotransform_dir.name)

    all_metrics: list[dict[str, Any]] = []

    # Process isotransform files
    isotransform_files = sorted(isotransform_dir.glob("*_metadata.json"))
    if isotransform_files:
        n_iso = len(isotransform_files)
        total_batches = (n_iso + batch_size - 1) // batch_size
        print(
            f"   Found {n_iso} isotransform files, "
            f"processing in {total_batches} batches..."
        )
        iso_metrics = process_all_isotransform_grids(
            isotransform_dataset, isotransform_files, batch_size
        )
        all_metrics.extend(iso_metrics)
        print(f"   Processed {len(iso_metrics)} isotransform cell metrics")

    # Process baseline files if provided
    if baseline_dir and baseline_dataset and baseline_dir.exists():
        baseline_files = sorted(baseline_dir.glob("*_metadata.json"))
        if baseline_files:
            n_base = len(baseline_files)
            total_batches = (n_base + batch_size - 1) // batch_size
            print(
                f"   Found {n_base} baseline files, "
                f"processing in {total_batches} batches..."
            )
            baseline_metrics = process_baseline_grids(
                baseline_dataset, baseline_files, batch_size
            )
            all_metrics.extend(baseline_metrics)
            print(f"   Processed {len(baseline_metrics)} baseline cell metrics")

    if not all_metrics:
        raise ValueError(f"No valid metrics extracted for {model_name}")

    df = pd.DataFrame(all_metrics)

    # Compute summaries
    transform_summary = compute_transform_summary(
        df, metrics=["entropy_bits", "jsd", "is_action_optimal"]
    )
    distance_by_transform = compute_distance_summary_by_transform(df)

    # Run regressions for key outcomes
    regression_results = {}
    for outcome in ["is_action_optimal", "entropy_bits", "jsd"]:
        if outcome in df.columns:
            result = run_transform_regression(df, outcome)
            if result:
                regression_results[outcome] = result

    return IsotransformModelResults(
        model_name=model_name,
        df=df,
        transform_summary=transform_summary,
        distance_by_transform=distance_by_transform,
        regression_results=regression_results,
    )


def load_isotransform_models(
    isotransform_dir: Path,
    isotransform_dataset_path: str,
    batch_size: int = 100,
    output_dir: Optional[Path] = None,
    skip_cached: bool = False,
    baseline_dir: Optional[Path] = None,
    baseline_dataset_path: Optional[str] = None,
) -> dict[str, IsotransformModelResults]:
    """Load isotransform results from multiple model directories.

    Args:
        isotransform_dir: Directory containing model subdirectories with isotransform data
        isotransform_dataset_path: Path to isotransform grids pickle file
        batch_size: Number of metadata files to load per batch
        output_dir: Optional directory to save results (enables caching)
        skip_cached: If True and output_dir is set, skip models already cached
        baseline_dir: Optional directory containing baseline model subdirectories
        baseline_dataset_path: Optional path to baseline grids pickle file

    Returns:
        Dictionary mapping model names to results
    """
    print(f"\nDiscovering model directories in {isotransform_dir}...")
    model_dirs = discover_isotransform_model_directories(isotransform_dir)
    print(f"Found {len(model_dirs)} model directories")

    if not model_dirs:
        raise ValueError(
            f"No isotransform model directories found in {isotransform_dir}"
        )

    # Check for cached models if output_dir provided
    cached_models: set[str] = set()
    if output_dir and skip_cached:
        cached_paths = discover_cached_models(output_dir)
        cached_models = set(cached_paths.keys())
        if cached_models:
            print(
                f"Found {len(cached_models)} cached models, will skip: {sorted(cached_models)}"
            )

    # Filter out cached models
    model_dirs_to_process = [
        d for d in model_dirs if sanitize_label(d.name) not in cached_models
    ]

    if not model_dirs_to_process:
        print("All models already cached, nothing to process.")
        return {}

    print(f"Will process {len(model_dirs_to_process)} model(s)")

    # Load isotransform dataset
    print(f"\nLoading isotransform environments from {isotransform_dataset_path}...")
    isotransform_dataset = load_environments(isotransform_dataset_path)
    print(f"Loaded {len(isotransform_dataset)} isotransform environments")

    # Load baseline dataset if provided
    baseline_dataset = None
    if baseline_dataset_path:
        print(f"Loading baseline environments from {baseline_dataset_path}...")
        baseline_dataset = load_environments(baseline_dataset_path)
        print(f"Loaded {len(baseline_dataset)} baseline environments")

    # Build mapping of model names to baseline directories
    baseline_model_dirs: dict[str, Path] = {}
    if baseline_dir and baseline_dir.exists():
        for subdir in baseline_dir.iterdir():
            if subdir.is_dir():
                baseline_model_dirs[sanitize_label(subdir.name)] = subdir
        print(f"Found {len(baseline_model_dirs)} baseline model directories")

    results = {}
    for i, model_dir in enumerate(model_dirs_to_process):
        model_name = sanitize_label(model_dir.name)
        print(f"\n[{i + 1}/{len(model_dirs_to_process)}] Processing {model_name}...")

        # Find matching baseline directory
        model_baseline_dir = baseline_model_dirs.get(model_name)
        if model_baseline_dir:
            print(f"   Found matching baseline directory: {model_baseline_dir.name}")
        elif baseline_dir:
            print(f"   Warning: No matching baseline directory found for {model_name}")

        try:
            model_results = load_isotransform_model(
                isotransform_dir=model_dir,
                isotransform_dataset=isotransform_dataset,
                model_name=model_name,
                batch_size=batch_size,
                baseline_dir=model_baseline_dir,
                baseline_dataset=baseline_dataset,
            )
            results[model_name] = model_results
            print(
                f"   Computed {model_results.n_samples} samples, "
                f"{len(model_results.transforms)} transforms"
            )

            # Save to cache if output_dir provided
            if output_dir:
                save_model_results(model_results, output_dir)

        except Exception as e:
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()

    return results


def load_or_compute_models(
    isotransform_dir: Optional[Path],
    isotransform_dataset_path: str,
    output_dir: Path,
    batch_size: int = 100,
    use_cached: bool = True,
    baseline_dir: Optional[Path] = None,
    baseline_dataset_path: Optional[str] = None,
) -> dict[str, IsotransformModelResults]:
    """Load models from cache and/or compute from metadata.

    This function supports incremental computation:
    1. First loads any cached results from output_dir
    2. Then computes results for any new models found in isotransform_dir
    3. Saves newly computed results to output_dir

    Args:
        isotransform_dir: Optional directory containing model subdirectories
        isotransform_dataset_path: Path to isotransform grids pickle file
        output_dir: Directory for cached results
        batch_size: Number of metadata files to load per batch
        use_cached: Whether to use cached results
        baseline_dir: Optional directory containing baseline model subdirectories
        baseline_dataset_path: Optional path to baseline grids pickle file

    Returns:
        Dictionary mapping model names to results
    """
    results: dict[str, IsotransformModelResults] = {}

    # Step 1: Load cached results
    if use_cached:
        print("\n" + "=" * 60)
        print("LOADING CACHED RESULTS")
        print("=" * 60)
        cached_results = load_models_from_cache(output_dir)
        results.update(cached_results)
        print(f"Loaded {len(cached_results)} cached model(s)")

    # Step 2: Compute new models if isotransform_dir provided
    if isotransform_dir and isotransform_dir.exists():
        print("\n" + "=" * 60)
        print("PROCESSING NEW MODELS")
        print("=" * 60)
        new_results = load_isotransform_models(
            isotransform_dir=isotransform_dir,
            isotransform_dataset_path=isotransform_dataset_path,
            batch_size=batch_size,
            output_dir=output_dir,
            skip_cached=use_cached,
            baseline_dir=baseline_dir,
            baseline_dataset_path=baseline_dataset_path,
        )
        results.update(new_results)
        print(f"Computed {len(new_results)} new model(s)")

    return results


# =============================================================================
# Regression Summary
# =============================================================================


def build_regression_summary(
    model_results: dict[str, IsotransformModelResults],
    outcome: str = "is_action_optimal",
) -> pd.DataFrame:
    """Build summary table of regression coefficients across models.

    Args:
        model_results: Dictionary of model results
        outcome: Outcome variable to summarize

    Returns:
        DataFrame with coefficients and significance per model
    """
    rows = []
    for model_name, results in model_results.items():
        if outcome not in results.regression_results:
            # Model has no regression results (likely missing baseline)
            # Add a row with basic info
            transforms = results.transforms
            print(
                f"   Warning: {model_name} has no regression for {outcome}. "
                f"Transforms: {transforms}"
            )
            row = {
                "Model": model_name,
                "N": results.n_samples,
                "R²": float("nan"),
            }
            rows.append(row)
            continue

        reg = results.regression_results[outcome]

        row = {"Model": model_name, "N": reg.n_samples, "R²": reg.r_squared}

        # Add transform coefficients
        for coef_name, coef_val in reg.coefficients.items():
            if coef_name.startswith("transform_type"):
                transform = coef_name.replace("transform_type_", "")
                p_val = reg.p_values.get(coef_name, 1.0)
                sig = (
                    "***"
                    if p_val < 0.001
                    else "**"
                    if p_val < 0.01
                    else "*"
                    if p_val < 0.05
                    else ""
                )
                row[f"{transform}_coef"] = coef_val
                row[f"{transform}_p"] = p_val
                row[f"{transform}_sig"] = sig

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Model", "N", "R²"])

    return pd.DataFrame(rows).sort_values("Model")


# =============================================================================
# Visualizations
# =============================================================================


def plot_metrics_by_distance_by_transform(
    model_results: IsotransformModelResults,
    output_path: Path,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """Plot entropy, divergence, and error rate vs distance, faceted by transform.

    Args:
        model_results: Results for a single model
        output_path: Path to save figure
        figsize: Figure size
    """
    setup_paper_style()

    df = model_results.distance_by_transform
    transforms = sorted(df["transform_type"].unique())
    n_transforms = len(transforms)

    fig, axes = plt.subplots(n_transforms, 3, figsize=figsize, sharex=True)

    if n_transforms == 1:
        axes = [axes]

    for i, transform in enumerate(transforms):
        subset = df[df["transform_type"] == transform]
        color = TRANSFORM_COLORS.get(transform, "gray")

        # Entropy
        axes[i][0].errorbar(
            subset["distance_to_goal"],
            subset["mean_entropy"],
            yerr=subset["std_entropy"],
            marker="o",
            capsize=3,
            color=color,
        )
        if i == 0:
            axes[i][0].set_title("Mean Entropy (bits)")
        axes[i][0].set_ylabel(transform, rotation=0, labelpad=60, fontsize=10)
        axes[i][0].grid(True, alpha=0.3)

        # JSD
        axes[i][1].errorbar(
            subset["distance_to_goal"],
            subset["mean_divergence"],
            yerr=subset["std_divergence"],
            marker="o",
            capsize=3,
            color=color,
        )
        if i == 0:
            axes[i][1].set_title("Mean JSD")
        axes[i][1].grid(True, alpha=0.3)

        # Error rate
        axes[i][2].plot(
            subset["distance_to_goal"],
            subset["error_rate"],
            marker="o",
            color=color,
        )
        if i == 0:
            axes[i][2].set_title("Error Rate")
        axes[i][2].set_ylim(0, 1)
        axes[i][2].grid(True, alpha=0.3)

    # X labels on bottom row
    for ax in axes[-1]:
        ax.set_xlabel("Distance to Goal")

    fig.suptitle(
        f"Metrics by Distance to Goal — {model_results.model_name}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_distance_transforms_overlay(
    model_results: IsotransformModelResults,
    output_path: Path,
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot metrics vs distance with all transforms overlaid.

    Args:
        model_results: Results for a single model
        output_path: Path to save figure
        figsize: Figure size
    """
    setup_paper_style()

    df = model_results.distance_by_transform
    transforms = sorted(df["transform_type"].unique())

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for transform in transforms:
        subset = df[df["transform_type"] == transform]
        color = TRANSFORM_COLORS.get(transform, "gray")
        label = transform

        # Entropy
        axes[0].plot(
            subset["distance_to_goal"],
            subset["mean_entropy"],
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )

        # JSD
        axes[1].plot(
            subset["distance_to_goal"],
            subset["mean_divergence"],
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )

        # Error rate
        axes[2].plot(
            subset["distance_to_goal"],
            subset["error_rate"],
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )

    axes[0].set_xlabel("Distance to Goal")
    axes[0].set_ylabel("Mean Entropy (bits)")
    axes[0].set_title("Entropy vs Distance")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Distance to Goal")
    axes[1].set_ylabel("Mean JSD")
    axes[1].set_title("JSD vs Distance")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Distance to Goal")
    axes[2].set_ylabel("Error Rate")
    axes[2].set_title("Error Rate vs Distance")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transform_coefficients(
    model_results: dict[str, IsotransformModelResults],
    output_path: Path,
    outcome: str = "is_action_optimal",
    figsize: tuple[float, float] = (10, 5),
) -> None:
    """Plot transform regression coefficients across models.

    Args:
        model_results: Dictionary of model results
        output_path: Path to save figure
        outcome: Outcome variable
        figsize: Figure size
    """
    setup_paper_style()

    # Collect coefficients
    models = sorted(model_results.keys())
    transforms = [t for t in TRANSFORM_TYPES if t != "baseline"]

    coef_data = []
    for model in models:
        if outcome not in model_results[model].regression_results:
            continue
        reg = model_results[model].regression_results[outcome]
        for transform in transforms:
            coef_name = f"transform_type_{transform}"
            if coef_name in reg.coefficients:
                coef_data.append(
                    {
                        "Model": _format_model_name(model),
                        "Transform": transform,
                        "Coefficient": reg.coefficients[coef_name],
                        "p_value": reg.p_values.get(coef_name, 1.0),
                    }
                )

    if not coef_data:
        return

    df = pd.DataFrame(coef_data)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(transforms))
    width = 0.8 / len(models)
    models_in_data = sorted(df["Model"].unique())

    for i, model in enumerate(models_in_data):
        model_df = df[df["Model"] == model]
        coefs = [
            model_df[model_df["Transform"] == t]["Coefficient"].values[0]
            if len(model_df[model_df["Transform"] == t]) > 0
            else 0
            for t in transforms
        ]
        offset = (i - len(models_in_data) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            coefs,
            width,
            label=model,
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
        )

        # Add significance markers
        for j, t in enumerate(transforms):
            t_df = model_df[model_df["Transform"] == t]
            if len(t_df) > 0 and t_df["p_value"].values[0] < 0.05:
                ax.text(
                    x[j] + offset,
                    coefs[j] + 0.005 * np.sign(coefs[j]),
                    "*",
                    ha="center",
                    va="bottom" if coefs[j] >= 0 else "top",
                    fontsize=12,
                )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(transforms, rotation=45, ha="right")
    ax.set_ylabel(f"Coefficient ({outcome})")
    ax.set_xlabel("Transform Type")
    ax.legend(frameon=False, loc="best", fontsize=8)

    outcome_label = (
        "Accuracy"
        if outcome == "is_action_optimal"
        else outcome.replace("_", " ").title()
    )
    ax.set_title(f"Transform Effects on {outcome_label} (vs Baseline)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_rate_by_transform_comparison(
    model_results: dict[str, IsotransformModelResults],
    output_path: Path,
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """Plot error rate difference from baseline across models and transforms.

    Args:
        model_results: Dictionary of model results
        output_path: Path to save figure
        figsize: Figure size
    """
    setup_paper_style()

    # Compute mean error rate by transform for each model
    data = []
    for model_name, results in model_results.items():
        summary = results.transform_summary
        baseline_err = (
            1
            - summary[summary["transform_type"] == "baseline"][
                "is_action_optimal_mean"
            ].values[0]
        )

        for _, row in summary.iterrows():
            transform = row["transform_type"]
            err_rate = 1 - row["is_action_optimal_mean"]
            data.append(
                {
                    "Model": _format_model_name(model_name),
                    "Transform": transform,
                    "Error Rate": err_rate,
                    "Δ Error Rate": err_rate - baseline_err
                    if transform != "baseline"
                    else 0,
                }
            )

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Absolute error rate
    transforms = [t for t in TRANSFORM_TYPES if t in df["Transform"].values]
    pivot = df.pivot(index="Model", columns="Transform", values="Error Rate")[
        transforms
    ]

    pivot.plot(
        kind="bar",
        ax=axes[0],
        color=[TRANSFORM_COLORS.get(t, "gray") for t in transforms],
    )
    axes[0].set_ylabel("Error Rate")
    axes[0].set_xlabel("")
    axes[0].set_title("Error Rate by Transform")
    axes[0].legend(title="Transform", frameon=False, fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    # Delta from baseline
    df_delta = df[df["Transform"] != "baseline"]
    pivot_delta = df_delta.pivot(
        index="Model", columns="Transform", values="Δ Error Rate"
    )

    pivot_delta.plot(
        kind="bar",
        ax=axes[1],
        color=[TRANSFORM_COLORS.get(t, "gray") for t in pivot_delta.columns],
    )
    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Δ Error Rate (vs Baseline)")
    axes[1].set_xlabel("")
    axes[1].set_title("Change from Baseline")
    axes[1].legend(title="Transform", frameon=False, fontsize=8)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_model_name(name: str) -> str:
    """Format model name for display."""
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


def compare_isotransform_models(
    model_results: dict[str, IsotransformModelResults],
    output_dir: str | Path,
) -> IsotransformComparisonResults:
    """Run complete isotransform comparison across models.

    Args:
        model_results: Dictionary mapping model names to results
        output_dir: Directory to save outputs

    Returns:
        IsotransformComparisonResults
    """
    print("\n" + "=" * 70)
    print(f"ISOTRANSFORM MODEL COMPARISON ({len(model_results)} models)")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build regression summary
    print("\n1. Building regression summaries...")
    regression_summary = build_regression_summary(model_results, "is_action_optimal")

    # Save CSVs
    summary_path = output_path / "isotransform_regression_summary.csv"
    regression_summary.to_csv(summary_path, index=False)
    print(f"   Saved: {summary_path}")

    # Save per-model metrics
    for model_name, results in model_results.items():
        model_dir = output_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save transform summary
        results.transform_summary.to_csv(
            model_dir / "transform_summary.csv", index=False
        )

        # Save distance by transform
        results.distance_by_transform.to_csv(
            model_dir / "distance_by_transform.csv", index=False
        )

    # Generate visualizations
    print("\n2. Generating visualizations...")

    # Per-model visualizations
    for model_name, results in model_results.items():
        model_dir = output_path / model_name

        plot_metrics_by_distance_by_transform(
            results,
            model_dir / "metrics_by_distance_faceted.png",
        )
        print(f"   - {model_name}/metrics_by_distance_faceted.png")

        plot_metrics_by_distance_transforms_overlay(
            results,
            model_dir / "metrics_by_distance_overlay.png",
        )
        print(f"   - {model_name}/metrics_by_distance_overlay.png")

    # Cross-model visualizations
    plot_transform_coefficients(
        model_results,
        output_path / "transform_coefficients_accuracy.png",
        outcome="is_action_optimal",
    )
    print("   - transform_coefficients_accuracy.png")

    plot_transform_coefficients(
        model_results,
        output_path / "transform_coefficients_entropy.png",
        outcome="entropy_bits",
    )
    print("   - transform_coefficients_entropy.png")

    plot_transform_coefficients(
        model_results,
        output_path / "transform_coefficients_jsd.png",
        outcome="jsd",
    )
    print("   - transform_coefficients_jsd.png")

    plot_error_rate_by_transform_comparison(
        model_results,
        output_path / "error_rate_comparison.png",
    )
    print("   - error_rate_comparison.png")

    # Save text report
    report_path = output_path / "isotransform_report.txt"
    _save_isotransform_report(model_results, regression_summary, report_path)
    print(f"\n3. Saved report: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("REGRESSION SUMMARY (Accuracy ~ Transform + Controls)")
    print("=" * 70)
    print(regression_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("ISOTRANSFORM COMPARISON COMPLETE")
    print("=" * 70)

    return IsotransformComparisonResults(
        model_results=model_results,
        regression_summary=regression_summary,
        output_dir=output_path,
    )


def _save_isotransform_report(
    model_results: dict[str, IsotransformModelResults],
    regression_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save text report of isotransform analysis."""
    lines = [
        "ISOTRANSFORM ANALYSIS REPORT",
        "=" * 60,
        f"Models analyzed: {len(model_results)}",
        "",
        "REGRESSION: is_action_optimal ~ transform_type + grid_size + complexity",
        "(baseline is reference category)",
        "",
    ]

    # Per-model summaries
    for model_name, results in model_results.items():
        lines.append(f"\n{model_name}")
        lines.append("-" * 40)

        if "is_action_optimal" in results.regression_results:
            reg = results.regression_results["is_action_optimal"]
            lines.append(f"N = {reg.n_samples}, R² = {reg.r_squared:.4f}")
            lines.append(f"Baseline accuracy: {reg.baseline_mean:.4f}")
            lines.append("Transform coefficients (effect on accuracy):")

            for coef_name, coef_val in reg.coefficients.items():
                if coef_name.startswith("transform_type"):
                    transform = coef_name.replace("transform_type_", "")
                    p_val = reg.p_values.get(coef_name, 1.0)
                    se = reg.std_errors.get(coef_name, 0)
                    sig = (
                        "***"
                        if p_val < 0.001
                        else "**"
                        if p_val < 0.01
                        else "*"
                        if p_val < 0.05
                        else ""
                    )
                    lines.append(
                        f"  {transform}: {coef_val:+.4f} (SE={se:.4f}, p={p_val:.4f}) {sig}"
                    )

    lines.append("\n\nSIGNIFICANCE CODES: *** p<0.001, ** p<0.01, * p<0.05")

    output_path.write_text("\n".join(lines) + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Compare isotransform effects across multiple LLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--isotransform-dir",
        type=str,
        default=None,
        help="Directory containing model subdirectories with isotransform metadata. "
        "Optional if using --use-cached with pre-computed results.",
    )

    parser.add_argument(
        "--isotransform-dataset",
        type=str,
        default="src/reveng/experiments/datasets/isodifficulty_grids.pkl",
        help="Path to the isotransform grids pickle file",
    )

    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="src/reveng/experiments/prob_policy_results_threads",
        help="Directory containing model subdirectories with baseline metadata. "
        "Model subdirectory names should match those in --isotransform-dir.",
    )

    parser.add_argument(
        "--baseline-dataset",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Path to the baseline grids pickle file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/analysis/isotransform_comparison",
        help="Directory to save comparison outputs and cached model results",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of metadata files to load per batch (limits RAM usage)",
    )

    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Load pre-computed results from output-dir and skip already-cached models",
    )

    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only compute and save per-model results without running comparison. "
        "Useful for incremental processing one model at a time.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    isotransform_path = Path(args.isotransform_dir) if args.isotransform_dir else None
    baseline_path = Path(args.baseline_dir) if args.baseline_dir else None

    # Validate arguments
    if not args.use_cached and not isotransform_path:
        parser.error("--isotransform-dir is required unless --use-cached is specified")

    # Load/compute models
    model_results = load_or_compute_models(
        isotransform_dir=isotransform_path,
        isotransform_dataset_path=args.isotransform_dataset,
        output_dir=output_path,
        batch_size=args.batch_size,
        use_cached=args.use_cached,
        baseline_dir=baseline_path,
        baseline_dataset_path=args.baseline_dataset if baseline_path else None,
    )

    if not model_results:
        print("No models loaded. Exiting.")
        return

    print(f"\nTotal models available: {len(model_results)}")

    # Run comparison unless cache-only mode
    if args.cache_only:
        print("\n--cache-only mode: skipping comparison, results saved to cache.")
    else:
        compare_isotransform_models(model_results, args.output_dir)


if __name__ == "__main__":
    main()
