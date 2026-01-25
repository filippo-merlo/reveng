import os
import matplotlib.pyplot as plt
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.key_minigrid import CoinMinigridEnv
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv
from reveng.environment_generator.utils import remove_door
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)


def plot_complexities(num_instances=5):
    """Initialize a MiniGrid environment and plot its starting frame for different complexities.

    Args:
        num_instances: Number of instances to generate for each complexity level (default: 5)
    """
    # Create output directory
    output_dir = "mini_grid_plots_complexities"
    os.makedirs(output_dir, exist_ok=True)

    complexities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for complexity in complexities:
        print(
            f"\nGenerating {num_instances} instances for complexity {complexity:.1f}..."
        )
        # Create environment with specific complexity
        env = Simple2DNavigationEnv(size=11, complexity=complexity)

        for i in range(num_instances):
            # Reset to get a new maze instance
            env.reset()

            # Get the RGB frame using the same method as in rgb_obs_wrappers.py:49-50
            frame = env.get_frame(highlight=False)

            # Plot the frame
            plt.figure()
            plt.imshow(frame)
            plt.axis("off")
            plt.tight_layout()

            # Save as high-resolution PDF with cropped margins
            filename = os.path.join(
                output_dir, f"minigrid_complexity_{complexity:.1f}_instance_{i + 1}.pdf"
            )
            plt.savefig(
                filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0
            )
            print(f"Figure saved as '{filename}'")

            plt.close()

        env.close()

    print(f"\nAll figures saved to '{output_dir}/' directory")


def plot_multiple_mazes():
    """Generate and save multiple mazes with complexity 1.0."""
    # Create output directory
    output_dir = "mini_grid_plots_complexity1"
    os.makedirs(output_dir, exist_ok=True)

    complexity = 1.0
    num_iterations = 6

    # Create environment once
    env = Simple2DNavigationEnv(size=11, complexity=complexity)

    for i in range(num_iterations):
        # Reset to generate a new maze
        env.reset()

        # Get the RGB frame
        frame = env.get_frame(highlight=False)

        # Plot the frame
        plt.figure()
        plt.imshow(frame)
        plt.axis("off")
        plt.tight_layout()

        # Save as high-resolution PDF with cropped margins
        filename = os.path.join(
            output_dir, f"minigrid_complexity_1.0_instance_{i + 1}.pdf"
        )
        plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Figure saved as '{filename}'")

        plt.close()

    env.close()


def plot_multiple_environments():
    """Generate and save 5 figures for each environment type."""
    # Create output directory
    output_dir = "environment_plots"
    os.makedirs(output_dir, exist_ok=True)

    num_instances = 5

    # 1. Key/Coin environment (CoinMinigridEnv)
    print("\nGenerating Key environment figures...")
    key_env = CoinMinigridEnv(size=9, render_mode=None)
    for i in range(num_instances):
        key_env.reset()
        frame = key_env.get_frame(highlight=False)

        plt.figure()
        plt.imshow(frame)
        plt.axis("off")
        plt.tight_layout()

        filename = os.path.join(output_dir, f"key_env_instance_{i + 1}.pdf")
        plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Figure saved as '{filename}'")
        plt.close()
    key_env.close()

    # 2. Rooms environment with door (RoomsMinigridEnv with door)
    print("\nGenerating Rooms environment (with door) figures...")
    rooms_env = RoomsMinigridEnv(render_mode=None, add_door_key=True, rooms_per_side=2)
    for i in range(num_instances):
        rooms_env.reset()
        frame = rooms_env.get_frame(highlight=False)

        plt.figure()
        plt.imshow(frame)
        plt.axis("off")
        plt.tight_layout()

        filename = os.path.join(output_dir, f"rooms_with_door_instance_{i + 1}.pdf")
        plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Figure saved as '{filename}'")
        plt.close()
    rooms_env.close()

    # 3. Rooms environment without door (RoomsMinigridEnv with door removed)
    print("\nGenerating Rooms environment (without door) figures...")
    rooms_no_door_env = RoomsMinigridEnv(
        render_mode=None, add_door_key=True, rooms_per_side=2
    )
    for i in range(num_instances):
        rooms_no_door_env.reset()
        # Remove the door
        rooms_no_door_env.grid = remove_door(rooms_no_door_env).grid
        frame = rooms_no_door_env.get_frame(highlight=False)

        plt.figure()
        plt.imshow(frame)
        plt.axis("off")
        plt.tight_layout()

        filename = os.path.join(output_dir, f"rooms_no_door_instance_{i + 1}.pdf")
        plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Figure saved as '{filename}'")
        plt.close()
    rooms_no_door_env.close()

    print(f"\nAll figures saved to '{output_dir}/' directory")


def plot_complexity_0_4_with_text():
    """Generate and save 5 instances of environment with complexity 0.4 and size 11,
    showing both RGB visualization and text observation."""
    # Create output directory
    output_dir = "mini_grid_plots_complexity0.4_with_text"
    os.makedirs(output_dir, exist_ok=True)

    complexity = 0.4
    size = 11
    num_instances = 5

    # Create environment
    base_env = Simple2DNavigationEnv(size=size, complexity=complexity)

    # Wrap with text wrapper
    env = FullObservabilityTextWrapper(base_env)

    for i in range(num_instances):
        print(f"\nGenerating instance {i + 1}/{num_instances}...")
        # Reset to generate a new maze
        text_obs, _ = env.reset()

        # Get the RGB frame from the underlying environment
        frame = env.unwrapped.get_frame(highlight=False)

        # Save RGB visualization
        plt.figure()
        plt.imshow(frame)
        plt.axis("off")
        plt.tight_layout()

        rgb_filename = os.path.join(
            output_dir, f"minigrid_complexity_0.4_size_{size}_instance_{i + 1}_rgb.pdf"
        )
        plt.savefig(
            rgb_filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0
        )
        print(f"RGB figure saved as '{rgb_filename}'")
        plt.close()

        # Save text observation as a grid visualization
        # Parse the text observation into a grid
        lines = text_obs.strip().split("\n")
        grid_data = []

        # First line should be column headers, rest are rows with row numbers
        if lines:
            # Extract column numbers from first line to get the count
            col_headers = [cell for cell in lines[0].split() if cell]
            num_cols = len(col_headers)

            # Extract grid data (skip first line which has column headers)
            for line in lines[1:]:
                cells = [cell for cell in line.split() if cell]
                if cells:
                    # First cell is row number, rest are grid cells
                    grid_data.append(cells[1:] if len(cells) > 1 else cells)

        # Get dimensions
        num_rows = len(grid_data)

        # Ensure all rows have the same number of columns
        for row in grid_data:
            while len(row) < num_cols:
                row.append("")

        # Create figure
        fig, ax = plt.subplots(figsize=(num_cols * 0.6, num_rows * 0.6))
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(-0.5, num_rows - 0.5)
        ax.set_aspect("equal")

        # Draw grid lines
        for row_idx in range(num_rows + 1):
            ax.axhline(row_idx - 0.5, color="gray", linewidth=0.5)
        for col_idx in range(num_cols + 1):
            ax.axvline(col_idx - 0.5, color="gray", linewidth=0.5)

        # Add grid data (no labels), flipped vertically
        for row_idx, row in enumerate(grid_data):
            for col_idx, cell in enumerate(row):
                if cell:
                    # Flip vertically: use (num_rows - 1 - row_idx) for y position
                    ax.text(
                        col_idx,
                        num_rows - 1 - row_idx,
                        cell,
                        ha="center",
                        va="center",
                        fontsize=30,
                        fontfamily="monospace",
                        fontweight="bold",
                    )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.tight_layout()

        text_filename = os.path.join(
            output_dir, f"minigrid_complexity_0.4_size_{size}_instance_{i + 1}_text.pdf"
        )
        plt.savefig(
            text_filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0
        )
        print(f"Text observation saved as '{text_filename}'")
        plt.close()

    env.close()

    print(f"\nAll figures saved to '{output_dir}/' directory")


if __name__ == "__main__":
    # plot_complexities()
    # plot_multiple_mazes()
    # plot_multiple_environments()
    # plot_complexity_0_4_with_text()
    pass
