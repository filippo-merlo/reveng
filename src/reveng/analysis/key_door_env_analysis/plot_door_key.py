import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Path to the statistics file
REPO_ROOT = Path(__file__).resolve().parents[4]
STATS_PATH = REPO_ROOT / "trajectories_key_door" / "analysis_statistics.json"


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
    Plot success rate and stage optimality in a single plot with a separator.

    Args:
        stats: Dictionary containing statistics with means and standard deviations
    """
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # --- Success Rate (position 0) ---
    success_mean = stats["success_rate"]["mean"]

    ax.bar(
        0,
        success_mean,
        alpha=0.8,
        color="#2E86AB",
        edgecolor="black",
        linewidth=1.5,
        width=0.8,
    )

    # --- Stage Optimality (positions 1, 2, 3) ---
    stage_names_display = {
        "before_key": "Collecting Key",
        "after_key_before_door": "Opening Door",
        "after_door": "Reaching Goal",
    }

    stage_metrics = []
    stage_means = []
    stage_stds = []

    for stage_name, display_name in stage_names_display.items():
        stage_data = stats["stage_optimality"][stage_name]
        stage_metrics.append(display_name)
        stage_means.append(stage_data["mean"])
        stage_stds.append(stage_data["std"])

    # Position stage bars starting at position 1
    x_pos_stages = np.arange(1, 1 + len(stage_metrics))
    stage_colors = ["#F18F01", "#C73E1D", "#6A994E"]

    bars2 = ax.bar(
        x_pos_stages,
        stage_means,
        yerr=stage_stds,
        capsize=5,
        alpha=0.8,
        color=stage_colors,
        edgecolor="black",
        linewidth=1.5,
        width=0.8,
    )

    # Set up axes
    ax.set_ylabel("Percentage (%)", fontsize=18, fontweight="bold")
    all_labels = ["Success Rate"] + stage_metrics
    all_positions = [0] + list(x_pos_stages)
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=18)
    ax.set_ylim(0, 110)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Add vertical separator line between success rate and stages
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=2, alpha=0.5)

    # Add value label for success rate
    ax.text(
        0,
        success_mean + 2,
        f"{success_mean:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    # Add value labels for stages
    for bar, mean, std in zip(bars2, stage_means, stage_stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 2,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add section titles
    ax.text(
        0, 120, "Success Rate", ha="center", va="top", fontsize=20, fontweight="bold"
    )
    ax.text(
        2,
        120,
        "Accuracy per Stage",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save figure as PDF
    output_path = Path(STATS_PATH).parent / "analysis_results.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    print("Loading statistics...")
    stats_path = Path(STATS_PATH)

    if not stats_path.exists():
        print(f"Error: Statistics file not found at {STATS_PATH}")
        print("Please run analyze_door_key.py first to generate the statistics file.")
        exit(1)

    stats = load_statistics(str(stats_path))

    print("Plotting results...")
    plot_analysis_results(stats)
    print("Done!")
