"""Command-line interface for the reveng package."""

import tyro
import logging
from reveng.commands.get_trajectory import get_trajectory


def main():
    """Main entry point for the reveng-cli command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - get_trajectory: Generate and save agent trajectories in navigation environments
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    tyro.extras.subcommand_cli_from_dict(
        {
            "get_trajectory": get_trajectory
        }
    )

if __name__ == "__main__":
    main()
