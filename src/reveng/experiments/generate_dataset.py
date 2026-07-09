"""Dataset generator for baseline grid environments.

This module creates a set of `Simple2DNavigationEnv` instances covering a
cartesian product of grid sizes and maze complexities. The generated environments
are saved to a pickle file so downstream experiments can load the same grids
consistently.

The main entrypoint `generate_baseline` returns the mapping of grid_id -> env and
also persists it to disk at the given output path.
"""

import pickle
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from papers.papers_code.reveng.src.reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv


def generate_baseline(
    num_grids_per_cfg: int,
    grid_sizes: list[int],
    complexities: list[float],
    output_file: str = "baseline_grids.pkl",
) -> Dict[str, Simple2DNavigationEnv]:
    """
    Generate a dataset of Simple2DNavigationEnv grids in fully observable setting.

    Generates num_grids_per_cfg environments for each combination of grid_size and complexity.

    Args:
        num_grids_per_cfg: Number of grids to generate per (grid_size, complexity) configuration
        grid_sizes: List of grid sizes to use (each grid will be size x size)
        complexities: List of maze complexities from 0.0 (empty room) to 1.0 (perfect maze)
        output_file: Path to save the pickle file

    Returns:
        A dictionary mapping grid_id -> Simple2DNavigationEnv

    Example:
        If num_grids_per_cfg=2, grid_sizes=[5, 7], complexities=[0.3, 0.7],
        this will generate 2*2*2=8 grids total (2 grids for each of 4 configurations).
    """
    dataset = {}

    # Calculate total number of grids
    total_grids = num_grids_per_cfg * len(grid_sizes) * len(complexities)

    print(
        f"Generating {total_grids} grids across {len(grid_sizes)} grid sizes "
        f"and {len(complexities)} complexity levels..."
    )
    print(f"  Grid sizes: {grid_sizes}")
    print(f"  Complexities: {complexities}")
    print(f"  Grids per configuration: {num_grids_per_cfg}")

    grid_counter = 0

    # Iterate over all combinations of grid_size and complexity
    with tqdm(total=total_grids, desc="Generating grids") as pbar:
        for grid_size in grid_sizes:
            for complexity in complexities:
                # Generate num_grids_per_cfg for this configuration
                for i in range(num_grids_per_cfg):
                    # Create environment with specified parameters
                    env = Simple2DNavigationEnv(
                        size=grid_size,
                        complexity=complexity,
                        render_mode=None,  # No rendering needed for dataset generation
                    )

                    # Reset to generate the grid
                    env.reset()

                    # Create unique ID for this grid
                    grid_id = f"grid_size{grid_size}_complexity{complexity:.2f}_{i:04d}"

                    # Store the environment directly
                    dataset[grid_id] = env
                    grid_counter += 1

                    pbar.update(1)

    # Save to pickle file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {output_path}")
    print(f"Total grids: {len(dataset)}")
    print(
        f"Configurations: {len(grid_sizes)} sizes × {len(complexities)} complexities "
        f"× {num_grids_per_cfg} grids = {total_grids} total"
    )

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate baseline grid dataset with multiple configurations"
    )
    parser.add_argument(
        "--num-grids-per-cfg",
        type=int,
        default=10,
        help="Number of grids to generate per configuration (default: 1)",
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[7, 9, 11, 13, 15],
        help="List of grid sizes (e.g., --grid-sizes 5 7 9) (default: [7, 9, 11, 13, 15])",
    )
    parser.add_argument(
        "--complexities",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="List of maze complexities from 0.0 to 1.0 (e.g., --complexities 0.3 0.5 0.7) (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/reveng/experiments/datasets/baseline_grids.pkl",
        help="Output pickle file path (default: src/reveng/experiments/datasets/baseline_grids.pkl)",
    )

    args = parser.parse_args()

    # Validate complexities
    for c in args.complexities:
        if not 0.0 <= c <= 1.0:
            parser.error(f"Complexity must be between 0.0 and 1.0, got {c}")

    # Generate the dataset
    generate_baseline(
        num_grids_per_cfg=args.num_grids_per_cfg,
        grid_sizes=args.grid_sizes,
        complexities=args.complexities,
        output_file=args.output,
    )
