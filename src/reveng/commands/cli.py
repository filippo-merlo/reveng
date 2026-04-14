"""Command-line interface for the reveng package."""

import logging

import tyro

from reveng.commands.generate_counterfactual_grids import (
    generate_counterfactual_grids,
    generate_counterfactual_grids_all_pairs,
    get_counterfactual_trajectories_all_pairs,
)
from reveng.commands.get_trajectory import (
    get_trajectories,
    get_trajectories_key_door_env,
    get_trajectories_multiple_per_grid,
    get_trajectory,
    get_trajectory_key_door_env,
    upload_trajectories_dir,
)


def main():
    """Main entry point for the reveng-cli command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - get_trajectory: Generate and save agent trajectories in navigation environments
    - get_trajectories: Generate multiple agent trajectories across parameter combinations in parallel
    - get_trajectories_multiple_per_grid: Generate multiple trajectories on the same grid layout
    - upload_trajectories_dir: Upload a directory of trajectory/grid JSON files to Hugging Face
    - get_trajectory_key_door_env: Generate and save agent trajectories in rooms environments with key-door mechanics
    - get_trajectories_key_door_env: Generate multiple agent trajectories in rooms environments with key-door mechanics across parameter combinations in parallel
    - generate_counterfactual_grids: Build agent-moved and goal-moved counterfactual grids from stored trajectory step-0 states
    - generate_counterfactual_grids_all_pairs: Run counterfactual grid generation across all discovered size/complexity pairs
    - get_counterfactual_trajectories_all_pairs: Generate one-step trajectories for all selected counterfactual grids
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict(
        {
            "get_trajectory": get_trajectory,
            "get_trajectories": get_trajectories,
            "get_trajectories_multiple_per_grid": get_trajectories_multiple_per_grid,
            "upload_trajectories_dir": upload_trajectories_dir,
            "get_trajectory_key_door_env": get_trajectory_key_door_env,
            "get_trajectories_key_door_env": get_trajectories_key_door_env,
            "generate_counterfactual_grids": generate_counterfactual_grids,
            "generate_counterfactual_grids_all_pairs": generate_counterfactual_grids_all_pairs,
            "get_counterfactual_trajectories_all_pairs": get_counterfactual_trajectories_all_pairs,
        }
    )


if __name__ == "__main__":
    main()
