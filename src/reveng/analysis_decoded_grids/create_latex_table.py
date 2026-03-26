import pandas as pd
import os

def create_latex_table():
    # Read the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'outputs', 'summary_by_size_complexity.csv')
    df = pd.read_csv(csv_path)

    # Round numerical columns to 3 decimal places for better readability
    numeric_cols = df.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        df[col] = df[col].round(3)

    # Select key metrics
    key_metrics = ['mean_taken_optimal_gt', 'mean_taken_optimal_decoded',
                   'mean_gt_decoded_agreement', 'error_recovery_rate',
                   'avg_agent_distance', 'avg_goal_distance',
                   'min_agent_distance', 'min_goal_distance',
                   'max_agent_distance', 'max_goal_distance']

    # Filter for recovery calculation: only rows where GT wasn't optimal
    df_for_recovery = df[df['mean_taken_optimal_gt'] < 1.0].copy()

    # Create table aggregated by grid size (averaged across complexities)
    size_table = df.groupby('grid_size')[['mean_taken_optimal_gt', 'mean_taken_optimal_decoded',
                                           'mean_gt_decoded_agreement', 'avg_agent_distance',
                                           'avg_goal_distance', 'min_agent_distance',
                                           'min_goal_distance', 'max_agent_distance',
                                           'max_goal_distance']].mean().reset_index()
    # Calculate recovery only from filtered data
    size_recovery = df_for_recovery.groupby('grid_size')['error_recovery_rate'].mean().reset_index()
    size_table = size_table.merge(size_recovery, on='grid_size', how='left')
    size_table = size_table.rename(columns={'grid_size': 'Size'})

    # Create table aggregated by complexity (averaged across grid sizes)
    complexity_table = df.groupby('complexity')[['mean_taken_optimal_gt', 'mean_taken_optimal_decoded',
                                                  'mean_gt_decoded_agreement', 'avg_agent_distance',
                                                  'avg_goal_distance', 'min_agent_distance',
                                                  'min_goal_distance', 'max_agent_distance',
                                                  'max_goal_distance']].mean().reset_index()
    # Calculate recovery only from filtered data
    complexity_recovery = df_for_recovery.groupby('complexity')['error_recovery_rate'].mean().reset_index()
    complexity_table = complexity_table.merge(complexity_recovery, on='complexity', how='left')
    complexity_table = complexity_table.rename(columns={'complexity': 'Value'})

    # Rename metrics columns
    size_table = size_table.rename(columns={
        'mean_taken_optimal_gt': 'Opt. GT',
        'mean_taken_optimal_decoded': 'Opt. Dec.',
        'mean_gt_decoded_agreement': 'Agreement',
        'error_recovery_rate': 'Recovery',
        'avg_agent_distance': 'Avg Agent Dist',
        'avg_goal_distance': 'Avg Goal Dist',
        'min_agent_distance': 'Min Agent Dist',
        'min_goal_distance': 'Min Goal Dist',
        'max_agent_distance': 'Max Agent Dist',
        'max_goal_distance': 'Max Goal Dist'
    })

    complexity_table = complexity_table.rename(columns={
        'mean_taken_optimal_gt': 'Opt. GT',
        'mean_taken_optimal_decoded': 'Opt. Dec.',
        'mean_gt_decoded_agreement': 'Agreement',
        'error_recovery_rate': 'Recovery',
        'avg_agent_distance': 'Avg Agent Dist',
        'avg_goal_distance': 'Avg Goal Dist',
        'min_agent_distance': 'Min Agent Dist',
        'min_goal_distance': 'Min Goal Dist',
        'max_agent_distance': 'Max Agent Dist',
        'max_goal_distance': 'Max Goal Dist'
    })

    # Generate LaTeX manually for better control
    latex_combined = r"""\begin{table}[t!]
\centering
\caption{Results by Grid Size and Complexity}
\label{tab:results_combined}
\resizebox{\columnwidth}{!}{
\begin{tabular}{ccccccccccc}
\toprule
\multicolumn{11}{c}{\textbf{Grid Size (averaged across complexities)}} \\
\midrule
\textbf{Size} & \textbf{Opt. GT} & \textbf{Opt. Dec.} & \textbf{Agreement} & \textbf{Recovery} & \textbf{Avg Agent} & \textbf{Avg Goal} & \textbf{Min Agent} & \textbf{Min Goal} & \textbf{Max Agent} & \textbf{Max Goal} \\
\midrule
"""

    # Add grid size rows
    for _, row in size_table.iterrows():
        latex_combined += f"{int(row['Size'])} & ${row['Opt. GT']*100:.1f}$ & ${row['Opt. Dec.']*100:.1f}$ & ${row['Agreement']*100:.1f}$ & ${row['Recovery']*100:.1f}$ & ${row['Avg Agent Dist']:.2f}$ & ${row['Avg Goal Dist']:.2f}$ & ${row['Min Agent Dist']:.2f}$ & ${row['Min Goal Dist']:.2f}$ & ${row['Max Agent Dist']:.2f}$ & ${row['Max Goal Dist']:.2f}$ \\\\\n"

    latex_combined += r"""\midrule
\multicolumn{11}{c}{\textbf{Complexity (averaged across grid sizes)}} \\
\midrule
\textbf{Value} & \textbf{Opt. GT} & \textbf{Opt. Dec.} & \textbf{Agreement} & \textbf{Recovery} & \textbf{Avg Agent} & \textbf{Avg Goal} & \textbf{Min Agent} & \textbf{Min Goal} & \textbf{Max Agent} & \textbf{Max Goal} \\
\midrule
"""

    # Add complexity rows
    for _, row in complexity_table.iterrows():
        latex_combined += f"${row['Value']:.1f}$ & ${row['Opt. GT']*100:.1f}$ & ${row['Opt. Dec.']*100:.1f}$ & ${row['Agreement']*100:.1f}$ & ${row['Recovery']*100:.1f}$ & ${row['Avg Agent Dist']:.2f}$ & ${row['Avg Goal Dist']:.2f}$ & ${row['Min Agent Dist']:.2f}$ & ${row['Min Goal Dist']:.2f}$ & ${row['Max Agent Dist']:.2f}$ & ${row['Max Goal Dist']:.2f}$ \\\\\n"

    latex_combined += r"""\bottomrule
\end{tabular}}
\end{table}
"""

    # Save table
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    with open(os.path.join(output_dir, 'latex_table_combined.tex'), 'w') as f:
        f.write(latex_combined)

    print("LaTeX table generated successfully!")
    print("\n" + "="*80)
    print("COMBINED TABLE")
    print("="*80)
    print(latex_combined)

    return latex_combined


if __name__ == '__main__':
    create_latex_table()
