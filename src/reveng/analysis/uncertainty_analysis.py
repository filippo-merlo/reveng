"""Analyze how LLM action uncertainty relates to optimal branching factors."""

import argparse
import gc
import heapq
import json
import math
import pickle
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================

ACTION_NAME_TO_ID = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}
ACTION_NAMES_UPPER = set(ACTION_NAME_TO_ID.keys())
LOGPROB_EPS = 1e-12

# Type aliases for clarity
ActionID = int
ActionDist = dict[ActionID, float]
OptimalActionSet = set[ActionID]
GridCoord = tuple[int, int]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GridMetadata:
    """Metadata for a single grid instance."""

    grid_size: int
    complexity: float
    instance_id: int
    policy_metadata: list[list[dict[str, Any]]]


@dataclass
class CellMetrics:
    """Computed metrics for a single grid cell."""

    grid_id: str
    grid_size: int
    complexity: float
    instance_id: int
    x: int
    y: int
    llm_action: int
    num_optimal_actions: int
    entropy_bits: float
    optimal_entropy_bits: float
    cross_entropy_bits: Optional[float]
    optimal_mass: float
    is_action_optimal: int
    action_probs: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        base = {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "instance_id": self.instance_id,
            "x": self.x,
            "y": self.y,
            "llm_action": self.llm_action,
            "num_optimal_actions": self.num_optimal_actions,
            "entropy_bits": self.entropy_bits,
            "optimal_entropy_bits": self.optimal_entropy_bits,
            "cross_entropy_bits": self.cross_entropy_bits,
            "optimal_mass": self.optimal_mass,
            "is_action_optimal": self.is_action_optimal,
        }
        base.update({f"p_{name}": prob for name, prob in self.action_probs.items()})
        return base


@dataclass
class AnalysisResults:
    """Container for complete analysis results."""

    df: pd.DataFrame
    summary: pd.DataFrame
    correlations: dict[str, float]
    model_tag: str
    output_dir: Path


# =============================================================================
# File Parsing Utilities
# =============================================================================


def sanitize_label(value: str) -> str:
    if not value:
        return "model"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def parse_filename(filepath: Path) -> tuple[int, float, int]:
    """Extract grid_size, complexity, and instance_id from filename.

    Expected format: grid_size{N}_complexity{X.XX}_{NNNN}_metadata.json
    """
    name = filepath.stem
    parts = name.split("_")
    try:
        grid_size = int(parts[1].replace("size", ""))
        complexity = float(parts[2].replace("complexity", ""))
        instance_id = int(parts[3])
        return grid_size, complexity, instance_id
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid filename format: {filepath.name}") from e


# =============================================================================
# Optimal Actions Computation
# =============================================================================


def compute_optimal_actions(env: Any) -> list[list[OptimalActionSet]]:
    base_env = getattr(env, "unwrapped", env)
    grid = base_env.grid
    goal: GridCoord = tuple(base_env.goal_pos)
    width, height = grid.width, grid.height

    def is_passable(x: int, y: int) -> bool:
        """Check if a position is within bounds and not blocked."""
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        cell = grid.get(x, y)
        return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

    # (dx, dy, action_id): LEFT, RIGHT, UP, DOWN
    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    # Backward Dijkstra from goal
    distances: dict[GridCoord, float] = {goal: 0}
    heap: list[tuple[float, GridCoord]] = [(0, goal)]

    while heap:
        dist, (x, y) = heapq.heappop(heap)
        if dist > distances.get((x, y), float("inf")):
            continue

        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap, (new_dist, (nx, ny)))

    # Determine optimal actions for each cell
    optimal_actions: list[list[OptimalActionSet]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue

            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue

            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    neighbor_dist = distances.get((nx, ny), float("inf"))
                    if neighbor_dist == current_dist - 1:
                        optimal_actions[y][x].add(action)

    # Goal cell has no optimal actions (already at goal)
    gx, gy = goal
    optimal_actions[gy][gx] = set()

    return optimal_actions


# =============================================================================
# Logprob Parsing (More Robust Version)
# =============================================================================


def normalize_action_token(token: Optional[str]) -> str:
    if token is None:
        return ""
    # Remove quotes, whitespace, and convert to uppercase
    return token.strip().strip('"').strip("'").upper()


def is_action_token(token: str) -> bool:
    """Check if a token is a valid action name.

    Args:
        token: Token string to check

    Returns:
        True if token is a valid action (LEFT/RIGHT/UP/DOWN)
    """
    normalized = normalize_action_token(token)
    return normalized in ACTION_NAMES_UPPER


def find_action_token_entry(logprobs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not logprobs:
        return None

    n = len(logprobs)

    # Strategy 1: Structured search for "action": "<VALUE>" pattern
    for i, entry in enumerate(logprobs):
        token = entry.get("token", "")

        # Check if this is the "action" key in JSON
        if "action" not in token.lower():
            continue

        # Look for colon after the "action" key
        colon_idx = None
        for j in range(i + 1, min(i + 5, n)):
            if ":" in logprobs[j].get("token", ""):
                colon_idx = j
                break

        if colon_idx is None:
            continue

        # Search for action value after colon
        for k in range(colon_idx + 1, min(colon_idx + 5, n)):
            candidate_token = logprobs[k].get("token", "")

            # Skip whitespace and quote tokens
            if candidate_token.strip() in ("", '"', "'"):
                continue

            # Check if this is a valid action
            if is_action_token(candidate_token):
                return logprobs[k]

            # Stop if we hit end of JSON object
            if "}" in candidate_token:
                break

    # Strategy 2: Fallback - find any high-confidence standalone action token
    # This handles cases where JSON structure differs from expected
    for entry in logprobs:
        token = entry.get("token", "")
        if is_action_token(token):
            logprob = entry.get("logprob")
            # Only accept high-confidence actions (>36% probability)
            if logprob is not None and logprob > -1.0:
                return entry

    return None


def distribution_from_logprobs(
    logprobs: Optional[list[dict[str, Any]]],
) -> Optional[ActionDist]:
    if not logprobs:
        return None

    token_entry = find_action_token_entry(logprobs)
    if not token_entry:
        return None

    # Collect logprobs for each action
    entries: dict[ActionID, float] = {}

    def register_action(token_value: Optional[str], logprob: Optional[float]) -> None:
        """Register an action and its logprob."""
        action = normalize_action_token(token_value)
        if action in ACTION_NAME_TO_ID and logprob is not None:
            action_id = ACTION_NAME_TO_ID[action]
            # Keep highest logprob if we see the same action multiple times
            entries[action_id] = max(entries.get(action_id, -math.inf), logprob)

    # Register the chosen token
    register_action(token_entry.get("token"), token_entry.get("logprob"))

    # Register alternatives from top_logprobs
    for candidate in token_entry.get("top_logprobs") or []:
        register_action(candidate.get("token"), candidate.get("logprob"))

    if not entries:
        return None

    # Convert logprobs to probabilities using numerically stable softmax
    max_logprob = max(entries.values())
    probs = {aid: math.exp(lp - max_logprob) for aid, lp in entries.items()}

    total = sum(probs.values())
    if total <= 0:
        return None

    # Normalize to sum to 1.0 and ensure all actions have entries (0 for unseen)
    return {aid: probs.get(aid, 0.0) / total for aid in ACTION_ID_TO_NAME}


# =============================================================================
# Entropy and Information Theory Metrics
# =============================================================================


def shannon_entropy(dist: ActionDist) -> float:
    """Compute Shannon entropy of a probability distribution.

    H(X) = -sum_i p_i * log2(p_i)
    """
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def cross_entropy(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Compute cross-entropy between uniform optimal and model distributions.

    H(p_opt, q_model) = -sum_a p_opt(a) * log2(q_model(a))

    Where p_opt is uniform over the optimal action set.
    """
    if not optimal_actions:
        return None

    weight = 1.0 / len(optimal_actions)
    ce = 0.0
    for action in optimal_actions:
        model_prob = max(dist.get(action, 0.0), eps)
        ce -= weight * math.log2(model_prob)
    return ce


def compute_optimal_mass(optimal_actions: OptimalActionSet, dist: ActionDist) -> float:
    return sum(dist.get(action, 0.0) for action in optimal_actions)


def optimal_entropy(num_optimal_actions: int) -> float:
    """Entropy of uniform distribution over optimal actions: H = log2(k)."""
    if num_optimal_actions <= 0:
        return 0.0
    return math.log2(num_optimal_actions)


# =============================================================================
# Data Loading and Batching
# =============================================================================


def load_environments(dataset_path: str) -> dict[str, Any]:
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def discover_metadata_files(metadata_dir: Path) -> list[Path]:
    return sorted(metadata_dir.glob("*_metadata.json"))


def _load_single_metadata(fpath: Path) -> Optional[tuple[str, GridMetadata]]:
    try:
        grid_size, complexity, instance_id = parse_filename(fpath)
        key = f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}"

        with open(fpath, "r") as f:
            policy_metadata = json.load(f)

        return key, GridMetadata(
            grid_size=grid_size,
            complexity=complexity,
            instance_id=instance_id,
            policy_metadata=policy_metadata,
        )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Skipping invalid file {fpath.name}: {e}")
        return None


def load_metadata_batch(
    metadata_files: list[Path], max_workers: int = 8, show_progress: bool = False
) -> dict[str, GridMetadata]:
    metadata_dict: dict[str, GridMetadata] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_single_metadata, fpath): fpath
            for fpath in metadata_files
        }
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(
                completed, total=len(futures), desc="Loading metadata", leave=False
            )
        for future in completed:
            result = future.result()
            if result is not None:
                key, metadata = result
                metadata_dict[key] = metadata

    return metadata_dict


def batch_metadata_files(
    metadata_files: list[Path], batch_size: int
) -> Iterator[list[Path]]:
    for i in range(0, len(metadata_files), batch_size):
        yield metadata_files[i : i + batch_size]


# =============================================================================
# Core Analysis - Cell and Grid Processing
# =============================================================================


def process_cell(
    cell: Any,
    optimal_set: OptimalActionSet,
    grid_id: str,
    metadata: GridMetadata,
    x: int,
    y: int,
) -> Optional[CellMetrics]:
    if not isinstance(cell, dict):
        return None

    if not optimal_set:
        return None

    # Extract action distribution from logprobs
    dist = distribution_from_logprobs(cell.get("logprobs"))
    if dist is None:
        return None

    num_optimal = len(optimal_set)
    entropy_bits = shannon_entropy(dist)
    optimal_entropy_bits = optimal_entropy(num_optimal)
    cross_entropy_bits = cross_entropy(optimal_set, dist)
    optimal_mass_val = compute_optimal_mass(optimal_set, dist)
    llm_action = cell.get("llm_response", -1)

    action_probs = {
        ACTION_ID_TO_NAME[aid].lower(): dist.get(aid, 0.0) for aid in ACTION_ID_TO_NAME
    }

    return CellMetrics(
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


def process_grid(grid_id: str, env: Any, metadata: GridMetadata) -> list[CellMetrics]:
    results: list[CellMetrics] = []
    optimal_actions = compute_optimal_actions(env)

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            cell_result = process_cell(
                cell=cell,
                optimal_set=optimal_actions[y][x],
                grid_id=grid_id,
                metadata=metadata,
                x=x,
                y=y,
            )
            if cell_result:
                results.append(cell_result)

    return results


def process_all_grids(
    grids_dataset: dict[str, Any],
    metadata_files: list[Path],
    batch_size: int,
) -> list[dict[str, Any]]:
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
            grid_metrics = process_grid(key, env, metadata_batch[key])
            all_metrics.extend([m.to_dict() for m in grid_metrics])

        # Free memory after each batch
        metadata_batch.clear()
        gc.collect()

    return all_metrics


# =============================================================================
# Statistical Analysis
# =============================================================================


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
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
    return {
        "entropy": df["num_optimal_actions"].corr(df["entropy_bits"]),
        "cross_entropy": df["num_optimal_actions"].corr(df["cross_entropy_bits"]),
        "optimal_mass": df["num_optimal_actions"].corr(df["optimal_mass"]),
    }


# =============================================================================
# Visualization
# =============================================================================


def plot_heatmaps(df: pd.DataFrame, output_path: Path) -> None:
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
