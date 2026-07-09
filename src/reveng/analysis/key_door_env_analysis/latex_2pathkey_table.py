# -*- coding: utf-8 -*-
import json
from pathlib import Path

# Path to the statistics file
REPO_ROOT = Path(__file__).resolve().parents[4]
STATS_FILE = REPO_ROOT / "trajectories_key2path" / "paired_analysis_statistics.json"
OUTPUT_FILE = REPO_ROOT / "trajectories_key2path" / "2path_key_results_table.tex"


def load_statistics(file_path: str) -> dict:
    """
    Load statistics from JSON file.

    Args:
        file_path: Path to the statistics JSON file

    Returns:
        Dictionary containing statistics
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["statistics"]


def generate_latex_table(stats: dict) -> str:
    """
    Generate a LaTeX table from the paired trajectory statistics.

    Args:
        stats: Dictionary containing statistics from compute_paired_statistics

    Returns:
        LaTeX table as a string
    """
    latex = []

    # Begin table
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Comparison of Agent Performance: With Key vs Without Key (2-Path Environment)}"
    )
    latex.append(r"\label{tab:2path_key_comparison}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Metric} & \textbf{With Key} & \textbf{Without Key} \\")
    latex.append(r"\hline")

    # Number of trajectories
    n_pairs = stats["n_pairs"]
    latex.append(f"Number of Trajectories & \\multicolumn{{2}}{{c}}{{{n_pairs}}} \\\\")
    latex.append(r"\hline")

    # Accuracy (mean � std)
    with_key_acc = stats["with_key"]["accuracy_mean"]
    with_key_acc_std = stats["with_key"]["accuracy_std"]
    without_key_acc = stats["without_key"]["accuracy_mean"]
    without_key_acc_std = stats["without_key"]["accuracy_std"]

    latex.append(
        f"Accuracy (\\%) & ${with_key_acc:.2f} \\pm {with_key_acc_std:.2f}$ & "
        f"${without_key_acc:.2f} \\pm {without_key_acc_std:.2f}$ \\\\"
    )

    # Success Rate (mean � std)
    with_key_success = stats["with_key"]["success_rate"]
    with_key_success_std = stats["with_key"]["success_rate_std"]
    without_key_success = stats["without_key"]["success_rate"]
    without_key_success_std = stats["without_key"]["success_rate_std"]

    latex.append(
        f"Success Rate (\\%) & ${with_key_success:.2f} \\pm {with_key_success_std:.2f}$ & "
        f"${without_key_success:.2f} \\pm {without_key_success_std:.2f}$ \\\\"
    )

    # Key Pickup Rate (only for with_key)
    key_pickup = stats["with_key"]["key_pickup_rate"]
    key_pickup_std = stats["with_key"]["key_pickup_rate_std"]
    latex.append(
        f"Key Pickup Rate (\\%) & ${key_pickup:.2f} \\pm {key_pickup_std:.2f}$ & N/A \\\\"
    )

    latex.append(r"\hline")

    # Accuracy difference
    acc_diff_mean = stats["accuracy_difference"]["mean"]
    acc_diff_std = stats["accuracy_difference"]["std"]
    acc_diff_median = stats["accuracy_difference"]["median"]

    latex.append(
        f"Accuracy Difference (Mean \\%) & \\multicolumn{{2}}{{c}}{{${acc_diff_mean:.2f} \\pm {acc_diff_std:.2f}$}} \\\\"
    )
    latex.append(
        f"Accuracy Difference (Median \\%) & \\multicolumn{{2}}{{c}}{{${acc_diff_median:.2f}$}} \\\\"
    )

    latex.append(r"\hline")

    # Jaccard Similarity (if available)
    if "jaccard_similarity" in stats:
        jaccard_mean = stats["jaccard_similarity"]["mean"]
        jaccard_std = stats["jaccard_similarity"]["std"]
        jaccard_median = stats["jaccard_similarity"]["median"]

        latex.append(
            f"Jaccard Similarity (Mean) & \\multicolumn{{2}}{{c}}{{${jaccard_mean:.4f} \\pm {jaccard_std:.4f}$}} \\\\"
        )
        latex.append(
            f"Jaccard Similarity (Median) & \\multicolumn{{2}}{{c}}{{${jaccard_median:.4f}$}} \\\\"
        )

        # Add separate Jaccard statistics for key picked vs not picked
        if "jaccard_similarity_key_picked" in stats:
            jaccard_picked_mean = stats["jaccard_similarity_key_picked"]["mean"]
            jaccard_picked_std = stats["jaccard_similarity_key_picked"]["std"]
            jaccard_picked_n = stats["jaccard_similarity_key_picked"]["n"]

            latex.append(
                f"Jaccard (Key Picked, n={jaccard_picked_n}) & \\multicolumn{{2}}{{c}}{{${jaccard_picked_mean:.4f} \\pm {jaccard_picked_std:.4f}$}} \\\\"
            )

        if "jaccard_similarity_key_not_picked" in stats:
            jaccard_not_picked_mean = stats["jaccard_similarity_key_not_picked"]["mean"]
            jaccard_not_picked_std = stats["jaccard_similarity_key_not_picked"]["std"]
            jaccard_not_picked_n = stats["jaccard_similarity_key_not_picked"]["n"]

            latex.append(
                f"Jaccard (Key Not Picked, n={jaccard_not_picked_n}) & \\multicolumn{{2}}{{c}}{{${jaccard_not_picked_mean:.4f} \\pm {jaccard_not_picked_std:.4f}$}} \\\\"
            )

    # End table
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def save_latex_table(latex_table: str, output_path: str) -> None:
    """
    Save the LaTeX table to a file.

    Args:
        latex_table: LaTeX table string
        output_path: Path to save the .tex file
    """
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {output_path}")


if __name__ == "__main__":
    # Load statistics
    print(f"Loading statistics from: {STATS_FILE}")
    stats = load_statistics(STATS_FILE)

    # Generate LaTeX table
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(stats)

    # Print to console
    print("\n" + "=" * 80)
    print("GENERATED LATEX TABLE")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    # Save to file
    save_latex_table(latex_table, OUTPUT_FILE)

    print("\nYou can now include this table in your LaTeX document using:")
    print(f"\\input{{{OUTPUT_FILE}}}")
