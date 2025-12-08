#!/usr/bin/env python3
"""CLI tool for inspecting grids and comparing optimal vs LLM action distributions.

Usage:
    # Search for grids with high cross-entropy
    uv run src/reveng/analysis/grid_inspector_cli.py search --dataset path/to/grids.pkl --metadata-dir path/to/metadata

    # Find grids with high branching factor (many optimal actions per cell)
    uv run src/reveng/analysis/grid_inspector_cli.py find-branching --dataset path/to/grids.pkl

    # Visualize a specific grid
    uv run src/reveng/analysis/grid_inspector_cli.py visualize --grid-id grid_size5_complexity0.30_0001 ...

    # Visualize a random grid
    uv run src/reveng/analysis/grid_inspector_cli.py visualize --random ...
"""

import argparse
import gc
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

from reveng.analysis.analysis_utils import (
    ActionDist,
    GridMetadata,
    compute_grid_mean_cross_entropy,
    compute_grid_mean_jsd,
    compute_optimal_actions,
    compute_optimal_distribution,
    cross_entropy,
    discover_metadata_files,
    distribution_from_logprobs,
    jensen_shannon_divergence,
    load_environments,
    load_metadata_batch,
    load_single_grid_metadata,
    optimal_entropy,
    shannon_entropy,
)

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
# Visualization Helpers
# =============================================================================


def _create_figure_ax(width: int, height: int, figsize_scale: float = 1.5):
    """Create a figure and axes for grid visualization."""
    fig, ax = plt.subplots(figsize=(width * figsize_scale, height * figsize_scale))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return fig, ax


def _draw_grid_base(
    ax,
    env: Any,
    width: int,
    height: int,
    wall_color: str = "#808080",
    goal_color: str = "#90EE90",
    valid_color: str = "#FFFFFF",
) -> None:
    """Draw the base grid structure (walls, goal, valid cells)."""
    goal_pos = tuple(env.goal_pos) if hasattr(env, "goal_pos") else None

    for j in range(height):
        for i in range(width):
            cell = env.grid.get(i, j)
            is_wall = cell is not None and cell.type == "wall"

            if (i, j) == goal_pos:
                color = goal_color
            elif is_wall:
                color = wall_color
            else:
                color = valid_color

            rect = patches.Rectangle(
                (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)


def _draw_arrow(
    ax, x: int, y: int, action: int, color: str = "#000000", alpha: float = 1.0
) -> None:
    """Draw an arrow indicating an action direction."""
    cx, cy = x + 0.5, y + 0.5
    arrow_length = 0.35

    if action == 0:  # LEFT
        start_x, start_y = cx + arrow_length / 2, cy
        end_x, end_y = cx - arrow_length / 2, cy
    elif action == 1:  # RIGHT
        start_x, start_y = cx - arrow_length / 2, cy
        end_x, end_y = cx + arrow_length / 2, cy
    elif action == 2:  # UP
        start_x, start_y = cx, cy + arrow_length / 2
        end_x, end_y = cx, cy - arrow_length / 2
    elif action == 3:  # DOWN
        start_x, start_y = cx, cy - arrow_length / 2
        end_x, end_y = cx, cy + arrow_length / 2
    else:
        return

    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2,
        color=color,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def _draw_distribution_text(
    ax, x: int, y: int, dist: ActionDist, fontsize: int = 7
) -> None:
    """Draw probability distribution text in a cell."""
    positions = {
        2: (x + 0.5, y + 0.2),  # UP
        3: (x + 0.5, y + 0.8),  # DOWN
        0: (x + 0.2, y + 0.5),  # LEFT
        1: (x + 0.8, y + 0.5),  # RIGHT
    }

    for action_id, pos in positions.items():
        prob = dist.get(action_id, 0.0)
        if prob > 0.01:  # Only show non-negligible probabilities
            text = f"{prob:.2f}"
            ax.text(
                pos[0],
                pos[1],
                text,
                color="#333333",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold" if prob > 0.5 else "normal",
            )


def _entropy_to_color(entropy: float, max_entropy: float = 2.0) -> str:
    """Convert entropy value to a color (low=blue, high=red)."""
    # Normalize entropy to [0, 1]
    normalized = min(entropy / max_entropy, 1.0)
    # Blue (low entropy) to Red (high entropy)
    r = int(255 * normalized)
    b = int(255 * (1 - normalized))
    return f"#{r:02x}00{b:02x}"


def visualize_grid_distributions(
    env: Any,
    metadata: GridMetadata,
    grid_id: str,
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> Figure:
    """Visualize a grid showing both optimal and LLM action distributions.

    Creates a side-by-side visualization with:
    - Left: Optimal action distribution (uniform over optimal actions)
    - Right: LLM action distribution from logprobs
    - Both panels show entropy values color-coded per cell

    Args:
        env: MiniGrid environment instance
        metadata: Grid metadata with policy information
        grid_id: Grid identifier for the title
        output_path: Optional path to save the figure
        show_plot: Whether to display the plot interactively

    Returns:
        The matplotlib Figure object
    """
    height = len(metadata.policy_metadata)
    width = len(metadata.policy_metadata[0]) if height > 0 else 0
    goal_pos = tuple(env.goal_pos) if hasattr(env, "goal_pos") else None

    # Compute optimal actions
    optimal_actions_grid = compute_optimal_actions(env)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(width * 2.5, height * 2.5))
    fig.suptitle(f"Grid: {grid_id}", fontsize=14, fontweight="bold")

    # Color definitions
    wall_color = "#808080"
    goal_color = "#90EE90"

    # Panel titles
    panel_titles = [
        ("Optimal Distribution", "Optimal Entropy"),
        ("LLM Distribution", "LLM Entropy"),
    ]

    for col, (dist_title, entropy_title) in enumerate(panel_titles):
        ax_dist = axes[0, col]
        ax_entropy = axes[1, col]

        # Setup axes
        for ax in [ax_dist, ax_entropy]:
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks(range(width + 1))
            ax.set_yticks(range(height + 1))
            ax.grid(True, alpha=0.3)

        ax_dist.set_title(dist_title, fontsize=12, fontweight="bold")
        ax_entropy.set_title(entropy_title, fontsize=12, fontweight="bold")

        # Draw each cell
        for j in range(height):
            for i in range(width):
                cell_data = metadata.policy_metadata[j][i]
                cell = env.grid.get(i, j)
                is_wall = cell is not None and cell.type == "wall"
                is_goal = (i, j) == goal_pos
                optimal_set = optimal_actions_grid[j][i]

                # Determine cell color for distribution panel
                if is_goal:
                    dist_color = goal_color
                elif is_wall:
                    dist_color = wall_color
                else:
                    dist_color = "#FFFFFF"

                # Draw cell background (distribution panel)
                rect = patches.Rectangle(
                    (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=dist_color
                )
                ax_dist.add_patch(rect)

                # Get distribution based on column
                if col == 0:  # Optimal
                    dist = compute_optimal_distribution(optimal_set)
                    entropy = optimal_entropy(len(optimal_set))
                else:  # LLM
                    if isinstance(cell_data, dict):
                        dist = distribution_from_logprobs(cell_data.get("logprobs"))
                    else:
                        dist = None
                    entropy = shannon_entropy(dist) if dist else 0.0

                # Draw entropy cell background
                if is_goal:
                    entropy_color = goal_color
                elif is_wall:
                    entropy_color = wall_color
                elif dist:
                    entropy_color = _entropy_to_color(entropy)
                else:
                    entropy_color = "#CCCCCC"

                rect_entropy = patches.Rectangle(
                    (i, j),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=entropy_color,
                )
                ax_entropy.add_patch(rect_entropy)

                # Draw distribution and entropy text for valid cells
                if not is_wall and not is_goal and dist:
                    _draw_distribution_text(ax_dist, i, j, dist, fontsize=6)

                    # Draw entropy value
                    ax_entropy.text(
                        i + 0.5,
                        j + 0.5,
                        f"{entropy:.2f}",
                        color="white" if entropy > 1.0 else "black",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )

                    # Draw arrow for highest probability action
                    if dist:
                        best_action = max(dist, key=lambda a: dist.get(a, 0))
                        if dist.get(best_action, 0) > 0.3:
                            _draw_arrow(
                                ax_dist, i, j, best_action, color="#333333", alpha=0.7
                            )

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=goal_color, edgecolor="black", label="Goal"),
        patches.Patch(facecolor=wall_color, edgecolor="black", label="Wall"),
        patches.Patch(facecolor="#0000FF", edgecolor="black", label="Low Entropy"),
        patches.Patch(facecolor="#FF0000", edgecolor="black", label="High Entropy"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    if show_plot:
        plt.show()

    return fig


def visualize_grid_with_metric(
    env: Any,
    metadata: GridMetadata,
    grid_id: str,
    metric: str = "ce",
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> Figure:
    """Visualize a grid showing divergence between optimal and LLM distributions.

    Creates a visualization showing:
    - Cell colors based on the metric (blue=low/good, red=high/bad)
    - Arrows showing LLM's chosen action (green=optimal, red=suboptimal)
    - Text showing metric values

    Args:
        env: MiniGrid environment instance
        metadata: Grid metadata with policy information
        grid_id: Grid identifier for the title
        metric: 'ce' for cross-entropy or 'jsd' for Jensen-Shannon divergence
        output_path: Optional path to save the figure
        show_plot: Whether to display the plot interactively

    Returns:
        The matplotlib Figure object
    """
    # Metric-specific configuration
    if metric == "jsd":
        metric_name = "Jensen-Shannon Divergence"
        metric_abbrev = "JSD"
        max_color_value = 1.0  # JSD is bounded [0, 1]
        text_threshold = 0.5  # Threshold for white vs black text
        metric_fn = jensen_shannon_divergence
        low_label = "Low JSD (similar)"
        high_label = "High JSD (different)"
        unit = ""
    else:  # ce
        metric_name = "Cross-Entropy"
        metric_abbrev = "CE"
        max_color_value = 4.0  # CE can be larger
        text_threshold = 2.0
        metric_fn = cross_entropy
        low_label = "Low CE"
        high_label = "High CE"
        unit = " bits"

    height = len(metadata.policy_metadata)
    width = len(metadata.policy_metadata[0]) if height > 0 else 0
    goal_pos = tuple(env.goal_pos) if hasattr(env, "goal_pos") else None

    # Compute optimal actions
    optimal_actions_grid = compute_optimal_actions(env)

    fig, ax = plt.subplots(figsize=(width * 1.8, height * 1.8))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{metric_name}: {grid_id}", fontsize=14, fontweight="bold")

    wall_color = "#808080"
    goal_color = "#90EE90"

    metric_values: list[float] = []

    for j in range(height):
        for i in range(width):
            cell_data = metadata.policy_metadata[j][i]
            cell = env.grid.get(i, j)
            is_wall = cell is not None and cell.type == "wall"
            is_goal = (i, j) == goal_pos
            optimal_set = optimal_actions_grid[j][i]

            if is_goal:
                color = goal_color
            elif is_wall:
                color = wall_color
            else:
                # Compute metric
                if isinstance(cell_data, dict):
                    dist = distribution_from_logprobs(cell_data.get("logprobs"))
                else:
                    dist = None

                if dist and optimal_set:
                    value = metric_fn(optimal_set, dist)
                    if value is not None:
                        metric_values.append(value)
                        color = _entropy_to_color(value, max_entropy=max_color_value)

                        # Draw metric value
                        ax.text(
                            i + 0.5,
                            j + 0.5,
                            f"{value:.2f}",
                            color="white" if value > text_threshold else "black",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                        )

                        # Draw arrow for LLM's chosen action
                        llm_action = cell_data.get("llm_response", -1)
                        if llm_action >= 0:
                            arrow_color = (
                                "#00FF00" if llm_action in optimal_set else "#FF0000"
                            )
                            _draw_arrow(
                                ax, i, j, llm_action, color=arrow_color, alpha=0.8
                            )
                    else:
                        color = "#CCCCCC"
                else:
                    color = "#CCCCCC"

            rect = patches.Rectangle(
                (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

    # Add statistics
    if metric_values:
        mean_val = sum(metric_values) / len(metric_values)
        max_val = max(metric_values)
        ax.text(
            0.02,
            0.98,
            f"Mean {metric_abbrev}: {mean_val:.3f}{unit}\nMax {metric_abbrev}: {max_val:.3f}{unit}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=goal_color, edgecolor="black", label="Goal"),
        patches.Patch(facecolor=wall_color, edgecolor="black", label="Wall"),
        patches.Patch(facecolor="#0000FF", edgecolor="black", label=low_label),
        patches.Patch(facecolor="#FF0000", edgecolor="black", label=high_label),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    if show_plot:
        plt.show()

    return fig


# Convenience wrappers for backward compatibility
def visualize_grid_with_cross_entropy(
    env: Any,
    metadata: GridMetadata,
    grid_id: str,
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> Figure:
    """Visualize a grid showing cross-entropy. See visualize_grid_with_metric."""
    return visualize_grid_with_metric(
        env,
        metadata,
        grid_id,
        metric="ce",
        output_path=output_path,
        show_plot=show_plot,
    )


def visualize_grid_with_jsd(
    env: Any,
    metadata: GridMetadata,
    grid_id: str,
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> Figure:
    """Visualize a grid showing JSD. See visualize_grid_with_metric."""
    return visualize_grid_with_metric(
        env,
        metadata,
        grid_id,
        metric="jsd",
        output_path=output_path,
        show_plot=show_plot,
    )


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_search(args: argparse.Namespace) -> None:
    """Search for grids with high divergence from optimal (cross-entropy or JSD)."""
    metric = args.metric
    metric_name = "Cross-Entropy" if metric == "ce" else "Jensen-Shannon Divergence"
    metric_abbrev = "CE" if metric == "ce" else "JSD"
    metric_unit = "bits" if metric == "ce" else ""

    print("=" * 60)
    print(f"GRID {metric_name.upper()} SEARCH")
    print("=" * 60)

    # Load environments
    print(f"\n1. Loading environments from: {args.dataset}")
    grids_dataset = load_environments(args.dataset)
    print(f"   Loaded {len(grids_dataset)} environments")

    # Discover metadata files
    metadata_path = Path(args.metadata_dir)
    metadata_files = discover_metadata_files(metadata_path)
    print(f"\n2. Found {len(metadata_files)} metadata files in: {metadata_path}")

    if not metadata_files:
        print("   No metadata files found. Exiting.")
        return

    # Process in batches to avoid running out of RAM
    batch_size = args.batch_size
    total_batches = (len(metadata_files) + batch_size - 1) // batch_size
    print(
        f"\n3. Processing {len(metadata_files)} grids in "
        f"{total_batches} batches of {batch_size}..."
    )

    grid_scores: list[tuple[str, float]] = []
    dataset_keys = set(grids_dataset.keys())

    for batch_idx, batch_files in enumerate(
        batch_metadata_files(metadata_files, batch_size)
    ):
        print(
            f"\n   Batch {batch_idx + 1}/{total_batches}: "
            f"loading {len(batch_files)} metadata files..."
        )

        # Load metadata for this batch
        metadata_batch = load_metadata_batch(batch_files, show_progress=True)
        common_keys = sorted(dataset_keys & set(metadata_batch.keys()))

        # Compute metric for each grid in this batch
        for grid_id in tqdm(
            common_keys,
            desc=f"Computing {metric_abbrev} (batch {batch_idx + 1}/{total_batches})",
            leave=False,
        ):
            env = grids_dataset[grid_id]
            metadata = metadata_batch[grid_id]

            if metric == "ce":
                score = compute_grid_mean_cross_entropy(grid_id, env, metadata)
            else:  # jsd
                score = compute_grid_mean_jsd(grid_id, env, metadata)

            if score is not None:
                grid_scores.append((grid_id, score))

        # Free memory after each batch
        metadata_batch.clear()
        gc.collect()

    # Sort by score (descending)
    grid_scores.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print("\n" + "=" * 60)
    print(f"TOP {args.top_k} GRIDS BY MEAN {metric_name.upper()}")
    print("=" * 60)

    for rank, (grid_id, score) in enumerate(grid_scores[: args.top_k], 1):
        unit_str = f" {metric_unit}" if metric_unit else ""
        print(f"{rank:3d}. {grid_id:45s} {metric_abbrev}: {score:.4f}{unit_str}")

    # Print summary statistics
    if grid_scores:
        all_scores = [s for _, s in grid_scores]
        print("\n" + "-" * 60)
        print("SUMMARY STATISTICS")
        print("-" * 60)
        print(f"Total grids analyzed: {len(grid_scores)}")
        print(f"Mean {metric_abbrev}:   {sum(all_scores) / len(all_scores):.4f}")
        print(f"Min {metric_abbrev}:    {min(all_scores):.4f}")
        print(f"Max {metric_abbrev}:    {max(all_scores):.4f}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        col_name = "mean_cross_entropy_bits" if metric == "ce" else "mean_jsd"
        with open(output_path, "w") as f:
            f.write(f"rank,grid_id,{col_name}\n")
            for rank, (grid_id, score) in enumerate(grid_scores, 1):
                f.write(f"{rank},{grid_id},{score:.6f}\n")
        print(f"\nSaved full rankings to: {output_path}")


def cmd_find_branching(args: argparse.Namespace) -> None:
    """Find grids with cells that have high numbers of optimal actions."""
    print("=" * 60)
    print("FIND GRIDS WITH HIGH BRANCHING FACTOR")
    print("=" * 60)

    # Load environments
    print(f"\n1. Loading environments from: {args.dataset}")
    grids_dataset = load_environments(args.dataset)
    print(f"   Loaded {len(grids_dataset)} environments")

    min_branching = args.min_branching

    # Track results
    # grid_id -> {max_branching, cells_with_target, total_cells, cell_positions}
    results: list[dict[str, Any]] = []

    # Global stats
    global_branching_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    grids_with_branching: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    print(
        f"\n2. Scanning {len(grids_dataset)} grids for cells with >= {min_branching} optimal actions..."
    )

    for grid_id in tqdm(grids_dataset.keys(), desc="Analyzing grids"):
        env = grids_dataset[grid_id]
        optimal_actions_grid = compute_optimal_actions(env)

        # Count cells by branching factor
        branching_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        high_branching_cells: list[tuple[int, int, int]] = []  # (x, y, num_optimal)
        max_branching = 0

        height = len(optimal_actions_grid)
        width = len(optimal_actions_grid[0]) if height > 0 else 0

        for y in range(height):
            for x in range(width):
                num_optimal = len(optimal_actions_grid[y][x])
                if num_optimal > 0:
                    if num_optimal in branching_counts:
                        branching_counts[num_optimal] += 1
                        global_branching_counts[num_optimal] += 1
                    max_branching = max(max_branching, num_optimal)

                    if num_optimal >= min_branching:
                        high_branching_cells.append((x, y, num_optimal))

        # Track which grids have each branching level
        for b in [1, 2, 3, 4]:
            if branching_counts[b] > 0:
                grids_with_branching[b].append(grid_id)

        # Only include grids that meet the threshold
        if high_branching_cells:
            results.append(
                {
                    "grid_id": grid_id,
                    "max_branching": max_branching,
                    "num_high_branching_cells": len(high_branching_cells),
                    "branching_counts": branching_counts,
                    "high_branching_cells": high_branching_cells,
                }
            )

    # Sort by max branching (descending), then by number of high-branching cells
    results.sort(
        key=lambda x: (x["max_branching"], x["num_high_branching_cells"]), reverse=True
    )

    # Print global statistics
    print("\n" + "=" * 60)
    print("GLOBAL BRANCHING STATISTICS")
    print("=" * 60)
    total_cells = sum(global_branching_counts.values())
    print(f"Total cells analyzed: {total_cells}")
    for b in [1, 2, 3, 4]:
        count = global_branching_counts[b]
        pct = 100 * count / total_cells if total_cells > 0 else 0
        num_grids = len(grids_with_branching[b])
        print(
            f"  {b} optimal action(s): {count:6d} cells ({pct:5.2f}%) in {num_grids} grids"
        )

    # Print results
    print("\n" + "=" * 60)
    print(f"GRIDS WITH CELLS HAVING >= {min_branching} OPTIMAL ACTIONS")
    print("=" * 60)

    if not results:
        print(f"\nNo grids found with cells having >= {min_branching} optimal actions.")
    else:
        print(f"\nFound {len(results)} grids with high-branching cells:\n")

        for i, r in enumerate(results[: args.top_k], 1):
            grid_id = r["grid_id"]
            max_b = r["max_branching"]
            num_cells = r["num_high_branching_cells"]
            print(f"{i:3d}. {grid_id}")
            print(
                f"      Max branching: {max_b}, Cells with >= {min_branching} optimal: {num_cells}"
            )

            # Show cell positions if verbose
            if args.verbose:
                cells = r["high_branching_cells"][:10]  # Limit to first 10
                positions = ", ".join(f"({x},{y}):{n}" for x, y, n in cells)
                if len(r["high_branching_cells"]) > 10:
                    positions += f" ... and {len(r['high_branching_cells']) - 10} more"
                print(f"      Cells: {positions}")
            print()

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(
                "rank,grid_id,max_branching,num_high_branching_cells,cell_positions\n"
            )
            for rank, r in enumerate(results, 1):
                cells_str = ";".join(
                    f"{x}:{y}:{n}" for x, y, n in r["high_branching_cells"]
                )
                f.write(
                    f"{rank},{r['grid_id']},{r['max_branching']},{r['num_high_branching_cells']},{cells_str}\n"
                )
        print(f"\nSaved full results to: {output_path}")

    # Special check for 4-way branching
    if global_branching_counts[4] > 0:
        print("\n" + "=" * 60)
        print("🎉 FOUND CELLS WITH 4 OPTIMAL ACTIONS! 🎉")
        print("=" * 60)
        grids_4 = grids_with_branching[4]
        print(
            f"Found {global_branching_counts[4]} cells with 4 optimal actions in {len(grids_4)} grids:\n"
        )

        # Show details for each grid with 4-optimal cells
        for gid in grids_4[:20]:
            # Find the result entry for this grid
            grid_result = next((r for r in results if r["grid_id"] == gid), None)
            if grid_result:
                cells_4 = [
                    (x, y) for x, y, n in grid_result["high_branching_cells"] if n == 4
                ]
                positions = ", ".join(f"({x},{y})" for x, y in cells_4[:10])
                if len(cells_4) > 10:
                    positions += f" ... +{len(cells_4) - 10} more"
                print(f"  {gid}")
                print(f"    Cell positions: {positions}")

        if len(grids_4) > 20:
            print(f"\n  ... and {len(grids_4) - 20} more grids")
    else:
        print("\n" + "-" * 60)
        print("No cells with 4 optimal actions found in any grid.")
        print("-" * 60)


def cmd_visualize(args: argparse.Namespace) -> None:
    """Visualize a grid showing optimal and LLM distributions."""
    print("=" * 60)
    print("GRID DISTRIBUTION VISUALIZATION")
    print("=" * 60)

    # Load environments
    print(f"\n1. Loading environments from: {args.dataset}")
    grids_dataset = load_environments(args.dataset)
    print(f"   Loaded {len(grids_dataset)} environments")

    # Determine which grid to visualize
    metadata_path = Path(args.metadata_dir)

    if args.random:
        # Pick a random grid
        metadata_files = discover_metadata_files(metadata_path)
        if not metadata_files:
            print("No metadata files found. Exiting.")
            return

        # Find a grid that exists in both dataset and metadata
        available_grid_ids = []
        for mf in metadata_files:
            grid_id = mf.stem.replace("_metadata", "")
            if grid_id in grids_dataset:
                available_grid_ids.append(grid_id)

        if not available_grid_ids:
            print("No matching grids found between dataset and metadata. Exiting.")
            return

        grid_id = random.choice(available_grid_ids)
        print(f"\n2. Randomly selected grid: {grid_id}")
    else:
        grid_id = args.grid_id
        if grid_id not in grids_dataset:
            print(f"Error: Grid '{grid_id}' not found in dataset.")
            print(f"Available grids: {len(grids_dataset)}")
            return
        print(f"\n2. Visualizing grid: {grid_id}")

    # Load metadata for the specific grid
    print("\n3. Loading metadata...")
    metadata = load_single_grid_metadata(grid_id, metadata_path)
    if metadata is None:
        print(f"Error: Metadata not found for grid '{grid_id}'")
        return

    env = grids_dataset[grid_id]

    # Generate visualization
    metric = args.metric
    print(f"\n4. Generating visualization (metric: {metric})...")

    output_path = args.output if args.output else None

    if metric == "full":
        visualize_grid_distributions(
            env=env,
            metadata=metadata,
            grid_id=grid_id,
            output_path=output_path,
            show_plot=not args.no_show,
        )
    else:  # ce or jsd
        visualize_grid_with_metric(
            env=env,
            metadata=metadata,
            grid_id=grid_id,
            metric=metric,
            output_path=output_path,
            show_plot=not args.no_show,
        )

    print("\nDone!")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect grids and compare optimal vs LLM action distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for grids with high cross-entropy
  uv run src/reveng/analysis/grid_inspector_cli.py search \\
      --dataset src/reveng/experiments/datasets/baseline_grids.pkl \\
      --metadata-dir /path/to/metadata --metric ce

  # Search for grids with high JSD (bounded [0,1], better for comparison)
  uv run src/reveng/analysis/grid_inspector_cli.py search \\
      --metadata-dir /path/to/metadata --metric jsd

  # Find grids with cells having 3+ optimal actions
  uv run src/reveng/analysis/grid_inspector_cli.py find-branching \\
      --dataset src/reveng/experiments/datasets/baseline_grids.pkl \\
      --min-branching 3 --verbose

  # Visualize a grid with JSD (bounded metric)
  uv run src/reveng/analysis/grid_inspector_cli.py visualize \\
      --grid-id grid_size5_complexity0.30_0001 \\
      --metadata-dir /path/to/metadata --metric jsd

  # Visualize a random grid with full distributions
  uv run src/reveng/analysis/grid_inspector_cli.py visualize --random \\
      --dataset src/reveng/experiments/datasets/baseline_grids.pkl \\
      --metadata-dir /path/to/metadata --metric full
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search for grids with high divergence from optimal (CE or JSD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    search_parser.add_argument(
        "--dataset",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Path to the grids pickle file",
    )
    search_parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Directory containing LLM policy metadata JSON files",
    )
    search_parser.add_argument(
        "--metric",
        type=str,
        choices=["ce", "jsd"],
        default="ce",
        help="Metric to use: 'ce' (cross-entropy) or 'jsd' (Jensen-Shannon divergence, bounded [0,1])",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top grids to display",
    )
    search_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of metadata files to load per batch (to limit RAM usage)",
    )
    search_parser.add_argument(
        "--output",
        type=str,
        help="Optional CSV file to save full rankings",
    )

    # Find-branching subcommand
    branching_parser = subparsers.add_parser(
        "find-branching",
        help="Find grids with cells having many optimal actions (high branching factor)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    branching_parser.add_argument(
        "--dataset",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Path to the grids pickle file",
    )
    branching_parser.add_argument(
        "--min-branching",
        type=int,
        default=3,
        help="Minimum number of optimal actions to search for",
    )
    branching_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top grids to display",
    )
    branching_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show cell positions for high-branching cells",
    )
    branching_parser.add_argument(
        "--output",
        type=str,
        help="Optional CSV file to save results",
    )

    # Visualize subcommand
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Visualize a grid with optimal and LLM distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    viz_parser.add_argument(
        "--dataset",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Path to the grids pickle file",
    )
    viz_parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Directory containing LLM policy metadata JSON files",
    )

    grid_group = viz_parser.add_mutually_exclusive_group(required=True)
    grid_group.add_argument(
        "--grid-id",
        type=str,
        help="Specific grid ID to visualize",
    )
    grid_group.add_argument(
        "--random",
        action="store_true",
        help="Visualize a randomly selected grid",
    )

    viz_parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save the visualization image",
    )
    viz_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot interactively (useful for saving only)",
    )
    viz_parser.add_argument(
        "--metric",
        type=str,
        choices=["full", "ce", "jsd"],
        default="full",
        help="Visualization mode: 'full' (distributions + entropy), 'ce' (cross-entropy), 'jsd' (Jensen-Shannon divergence)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "search":
        cmd_search(args)
    elif args.command == "find-branching":
        cmd_find_branching(args)
    elif args.command == "visualize":
        cmd_visualize(args)


if __name__ == "__main__":
    main()
