import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
KEY_DOOR_STATS = REPO_ROOT / "trajectories_key_door" / "analysis_statistics.json"
NO_DOOR_STATS = REPO_ROOT / "trajectories_no_door" / "analysis_statistics.json"


def load_statistics(file_path: str) -> dict:
    """Load statistics from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def generate_latex_table():
    """Generate a LaTeX table comparing key-door and no-door environments."""

    # Load both statistics files
    key_door_stats = load_statistics(KEY_DOOR_STATS)
    no_door_stats = load_statistics(NO_DOOR_STATS)

    # Extract values and format to 1 decimal place
    kd_success = f"{key_door_stats['success_rate']['mean']:.1f} \\pm {key_door_stats['success_rate']['std']:.1f}"
    kd_total_acc = f"{key_door_stats['total_accuracy']['mean']:.1f} \\pm {key_door_stats['total_accuracy']['std']:.1f}"
    kd_collecting = f"{key_door_stats['stage_optimality']['before_key']['mean']:.1f} \\pm {key_door_stats['stage_optimality']['before_key']['std']:.1f}"
    kd_opening = f"{key_door_stats['stage_optimality']['after_key_before_door']['mean']:.1f} \\pm {key_door_stats['stage_optimality']['after_key_before_door']['std']:.1f}"
    kd_reaching = f"{key_door_stats['stage_optimality']['after_door']['mean']:.1f} \\pm {key_door_stats['stage_optimality']['after_door']['std']:.1f}"
    kd_n_traj = key_door_stats["n_trajectories"]

    nd_success = f"{no_door_stats['success_rate']['mean']:.1f} \\pm {no_door_stats['success_rate']['std']:.1f}"
    nd_total_acc = f"{no_door_stats['total_accuracy']['mean']:.1f} \\pm {no_door_stats['total_accuracy']['std']:.1f}"
    nd_key_pickup = f"{no_door_stats['key_pickup_rate']['mean']:.1f} \\pm {no_door_stats['key_pickup_rate']['std']:.1f}"
    nd_non_opt = f"{no_door_stats['non_optimal_towards_key']['mean_per_trajectory']:.1f} \\pm {no_door_stats['non_optimal_towards_key']['std_per_trajectory']:.1f}"
    nd_n_traj = no_door_stats["n_trajectories"]

    latex_table = rf"""\begin{{table}}[h]
\centering
\caption{{Performance Comparison: Key-Door vs No-Door Environments}}
\label{{tab:key_door_comparison}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Metric}} & \textbf{{Key-Door Env}} & \textbf{{No-Door Env}} \\
\midrule
Success Rate (\%) & ${kd_success}$ & ${nd_success}$ \\
Total Accuracy (\%) & ${kd_total_acc}$ & ${nd_total_acc}$ \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Stage-specific Accuracy (\%):}}}} \\
\quad Collecting Key & ${kd_collecting}$ & --- \\
\quad Opening Door & ${kd_opening}$ & --- \\
\quad Reaching Goal & ${kd_reaching}$ & --- \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Key-related Metrics:}}}} \\
\quad Key Pickup Rate (\%) & --- & ${nd_key_pickup}$ \\
\quad Non-optimal Towards Key (\%) & --- & ${nd_non_opt}$ \\
\midrule
Number of Trajectories & {kd_n_traj} & {nd_n_traj} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""

    return latex_table


def generate_latex_table_transposed():
    """Generate a transposed LaTeX table with environments as rows."""

    # Load both statistics files
    key_door_stats = load_statistics(KEY_DOOR_STATS)
    no_door_stats = load_statistics(NO_DOOR_STATS)

    # Extract values and format to 1 decimal place
    kd_success = f"{key_door_stats['success_rate']['mean']:.1f} \\pm {key_door_stats['success_rate']['std']:.1f}"
    kd_total_acc = f"{key_door_stats['total_accuracy']['mean']:.1f} \\pm {key_door_stats['total_accuracy']['std']:.1f}"
    kd_collecting = f"{key_door_stats['stage_optimality']['before_key']['mean']:.1f} \\pm {key_door_stats['stage_optimality']['before_key']['std']:.1f}"
    kd_opening = f"{key_door_stats['stage_optimality']['after_key_before_door']['mean']:.1f} \\pm {key_door_stats['stage_optimality']['after_key_before_door']['std']:.1f}"
    kd_reaching = f"{key_door_stats['stage_optimality']['after_door']['mean']:.1f} \\pm {key_door_stats['stage_optimality']['after_door']['std']:.1f}"
    kd_n_traj = key_door_stats["n_trajectories"]

    nd_success = f"{no_door_stats['success_rate']['mean']:.1f} \\pm {no_door_stats['success_rate']['std']:.1f}"
    nd_total_acc = f"{no_door_stats['total_accuracy']['mean']:.1f} \\pm {no_door_stats['total_accuracy']['std']:.1f}"
    nd_key_pickup = f"{no_door_stats['key_pickup_rate']['mean']:.1f} \\pm {no_door_stats['key_pickup_rate']['std']:.1f}"
    nd_non_opt = f"{no_door_stats['non_optimal_towards_key']['mean_per_trajectory']:.1f} \\pm {no_door_stats['non_optimal_towards_key']['std_per_trajectory']:.1f}"
    nd_n_traj = no_door_stats["n_trajectories"]

    latex_table_transposed = rf"""\begin{{table}}[h]
\centering
\caption{{Performance Comparison: Key-Door vs No-Door Environments (Transposed)}}
\label{{tab:key_door_comparison_transposed}}
\begin{{tabular}}{{lccccccc}}
\toprule
\textbf{{Environment}} & \textbf{{Success}} & \textbf{{Total}} & \textbf{{Collecting}} & \textbf{{Opening}} & \textbf{{Reaching}} & \textbf{{Key Pickup}} & \textbf{{Non-opt.}} \\
 & \textbf{{Rate (\%)}} & \textbf{{Acc. (\%)}} & \textbf{{Key (\%)}} & \textbf{{Door (\%)}} & \textbf{{Goal (\%)}} & \textbf{{Rate (\%)}} & \textbf{{Towards Key (\%)}} \\
\midrule
Key-Door & ${kd_success}$ & ${kd_total_acc}$ & ${kd_collecting}$ & ${kd_opening}$ & ${kd_reaching}$ & --- & --- \\
No-Door & ${nd_success}$ & ${nd_total_acc}$ & --- & --- & --- & ${nd_key_pickup}$ & ${nd_non_opt}$ \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Number of Trajectories: Key-Door = {kd_n_traj}, No-Door = {nd_n_traj}}}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""

    return latex_table_transposed


if __name__ == "__main__":
    print("Generating LaTeX tables from analysis results...\n")

    # Check if statistics files exist
    if not KEY_DOOR_STATS.exists():
        print(f"Error: Key-door statistics file not found at {KEY_DOOR_STATS}")
        exit(1)

    if not NO_DOOR_STATS.exists():
        print(f"Error: No-door statistics file not found at {NO_DOOR_STATS}")
        exit(1)

    # Generate and print the original LaTeX table
    print("=" * 80)
    print("ORIGINAL TABLE (Metrics as Rows)")
    print("=" * 80)
    latex_table = generate_latex_table()
    print(latex_table)

    print("\n\n")

    # Generate and print the transposed LaTeX table
    print("=" * 80)
    print("TRANSPOSED TABLE (Environments as Rows)")
    print("=" * 80)
    latex_table_transposed = generate_latex_table_transposed()
    print(latex_table_transposed)

    print("\n" + "=" * 80)
    print("Both LaTeX tables generated successfully!")
    print("=" * 80)
