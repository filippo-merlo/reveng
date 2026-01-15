"""Command-line interface for the reveng package."""

import tyro
import logging
from reveng.commands.get_trajectory import get_trajectory, get_trajectories


def main():
    """Main entry point for the reveng-cli command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - get_trajectory: Generate and save agent trajectories in navigation environments
    - get_trajectories: Generate multiple agent trajectories across parameter combinations in parallel
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    tyro.extras.subcommand_cli_from_dict(
        {
            "get_trajectory": get_trajectory,
            "get_trajectories": get_trajectories
        }
    )

if __name__ == "__main__":
    main()
