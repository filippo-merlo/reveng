# -*- coding: utf-8 -*-
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Path to the statistics file
REPO_ROOT = Path(__file__).resolve().parents[4]
STATS_PATH = REPO_ROOT / "trajectories_no_door" / "analysis_statistics.json"


def load_statistics(file_path: str) -> dict:
    """
    Load statistics from a JSON file.

    Args:
        file_path: Path to the statistics JSON file

    Returns:
        Dictionary containing the statistics
    """
    with open(file_path, "r") as f:
        return json.load(f)


def plot_analysis_results(stats: dict) -> None:
    """
    Plot success rate, total accuracy, key pickup rate, and non-optimal steps towards key.

    Args:
        stats: Dictionary containing statistics with means and standard deviations
    """
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # --- Metrics to plot ---
    metrics_data = []

    # Success Rate (position 0)
    success_mean = stats["success_rate"]["mean"]
    success_std = stats["success_rate"]["std"]
    metrics_data.append(("Success Rate", success_mean, success_std, "#2E86AB"))

    # Total Accuracy (position 1)
    accuracy_mean = stats["total_accuracy"]["mean"]
    accuracy_std = stats["total_accuracy"]["std"]
    metrics_data.append(("Total Accuracy", accuracy_mean, accuracy_std, "#F18F01"))

    # Non-optimal steps towards key percentage (position 2)
    towards_key_mean = stats["non_optimal_towards_key"]["mean_per_trajectory"]
    towards_key_std = stats["non_optimal_towards_key"]["std_per_trajectory"]
    metrics_data.append(
        (
            "Non-optimal Actions\nMoving Towards Key",
            towards_key_mean,
            towards_key_std,
            "#6A994E",
        )
    )

    # Key Pickup Rate (position 3)
    key_pickup_mean = stats["key_pickup_rate"]["mean"]
    key_pickup_std = stats["key_pickup_rate"]["std"]
    metrics_data.append(("Key Pickup Rate", key_pickup_mean, key_pickup_std, "#C73E1D"))

    # Extract data for plotting
    labels = [item[0] for item in metrics_data]
    means = [item[1] for item in metrics_data]
    stds = [item[2] for item in metrics_data]
    colors = [item[3] for item in metrics_data]

    # Determine which bars should have error bars (only Total Accuracy)
    yerrs = [None, stds[1], None, None]  # Error bars for position 1 only

    # Plot bars with selective error bars
    x_positions = np.arange(len(labels))
    bars = []
    for i, (x, mean, yerr, color) in enumerate(zip(x_positions, means, yerrs, colors)):
        if yerr is not None:
            bar = ax.bar(
                x,
                mean,
                yerr=yerr,
                capsize=5,
                alpha=0.8,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                width=0.8,
            )
        else:
            bar = ax.bar(
                x,
                mean,
                alpha=0.8,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                width=0.8,
            )
        bars.append(bar)

    # Set up axes
    ax.set_ylabel("Percentage (%)", fontsize=18, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylim(0, 120)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Add vertical separator line between success rate and other metrics
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=2, alpha=0.5)

    # Add value labels on bars
    for i, (bar_container, mean, yerr) in enumerate(zip(bars, means, yerrs)):
        # Get the actual bar patch from the container
        bar = bar_container[0]
        height = bar.get_height()
        # Adjust label position based on whether error bar exists
        y_offset = yerr + 2 if yerr is not None else 2
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add section titles
    ax.text(
        0, 130, "Success Rate", ha="center", va="top", fontsize=20, fontweight="bold"
    )
    ax.text(
        2,
        130,
        "Accuracy and Key Influence",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save figure as PDF
    output_path = Path(STATS_PATH).parent / "analysis_results_no_door.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    print("Loading statistics...")
    stats_path = Path(STATS_PATH)

    if not stats_path.exists():
        print(f"Error: Statistics file not found at {STATS_PATH}")
        print(
            "Please run analyze_key_no_door.py first to generate the statistics file."
        )
        exit(1)

    stats = load_statistics(str(stats_path))

    # Print statistics for verification
    print("\nStatistics Summary:")
    print(
        f"Total Accuracy: {stats['total_accuracy']['mean']:.2f}% � {stats['total_accuracy']['std']:.2f}%"
    )
    print(
        f"Success Rate: {stats['success_rate']['mean']:.2f}% � {stats['success_rate']['std']:.2f}%"
    )
    print(
        f"Key Pickup Rate: {stats['key_pickup_rate']['mean']:.2f}% � {stats['key_pickup_rate']['std']:.2f}%"
    )

    towards_key_stats = stats["non_optimal_towards_key"]
    print(
        f"Non-optimal steps towards key: {towards_key_stats['total_towards_key']}/{towards_key_stats['total_non_optimal']} ({towards_key_stats['overall_percentage']:.2f}%)"
    )
    print(
        f"  Per-trajectory mean: {towards_key_stats['mean_per_trajectory']:.2f}% � {towards_key_stats['std_per_trajectory']:.2f}%"
    )

    print("\nPlotting results...")
    plot_analysis_results(stats)
    print("Done!")
