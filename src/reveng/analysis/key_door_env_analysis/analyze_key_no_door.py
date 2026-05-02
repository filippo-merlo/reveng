import json
from pathlib import Path
import numpy as np
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv
from reveng.environment_generator.utils import (
    compute_optimal_actions_from_position,
    replace_key_with_goal,
    remove_key,
)

DEBUG = False
REPO_ROOT = Path(__file__).resolve().parents[4]
DIR_PATH = REPO_ROOT / "trajectories_no_door"


def load_steps_from_json(file_path: str) -> list:
    """
    Load the 'steps' key from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List containing the steps data

    Raises:
        KeyError: If 'steps' key is not found in the JSON
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    if "steps" not in data:
        raise KeyError("'steps' key not found in JSON file")

    return data["steps"]


def parse_grid_state(grid_state: list[str]) -> list[list[str]]:
    """
    Parse a grid_state from the JSON format to a grid list.

    The input grid_state has coordinate labels on the first row and first column.
    This function removes those labels and returns just the grid content.

    Args:
        grid_state: List of strings representing the grid with coordinate labels

    Returns:
        2D list of strings representing the grid without labels

    Example:
        Input:
            [
                "  0 1 2 3 4 5 6 7 8 ",
                "0 # # # # # # # # # ",
                "1 # _ _ _ # _ _ _ # ",
                ...
            ]
        Output:
            [
                ['#', '#', '#', '#', '#', '#', '#', '#', '#'],
                ['#', '_', '_', '_', '#', '_', '_', '_', '#'],
                ...
            ]
    """
    # Skip the first row (coordinate header)
    grid_rows = grid_state[1:]

    parsed_grid = []
    for row in grid_rows:
        # Split by spaces and skip the first element (row coordinate)
        cells = row.split()
        if len(cells) > 1:  # Ensure there are cells after the row number
            parsed_grid.append(cells[1:])  # Skip the row number

    return parsed_grid


def load_env_from_step(
    step: dict, rooms_per_side: int = 2, render_mode: str | None = None
) -> RoomsMinigridEnv:
    """
    Load a RoomsMinigridEnv from a step's grid_state.

    This function supports all symbols including keys:
    - '#' (wall)
    - '_' (empty)
    - 'A' (agent)
    - 'G' (goal)
    - 'K' (key)

    Args:
        step: Dictionary containing a 'grid_state' key with the grid representation
        rooms_per_side: Number of rooms per side (2 for 2x2, 3 for 3x3). Default is 2.
        render_mode: Rendering mode ('human', 'rgb_array', or None). Default is None.

    Returns:
        Configured RoomsMinigridEnv instance

    Raises:
        KeyError: If 'grid_state' key is not found in the step
    """
    if "grid_state" not in step:
        raise KeyError("'grid_state' key not found in step")

    grid_state = step["grid_state"]
    parsed_grid = parse_grid_state(grid_state)

    # Create environment with appropriate room configuration
    # The add_door_key parameter doesn't matter since we're overriding with set_env_from_list
    env = RoomsMinigridEnv(
        rooms_per_side=rooms_per_side, add_door_key=False, render_mode=render_mode
    )

    # Reset to initialize the environment (this will print a warning)
    env.reset()

    # Set the environment from the parsed grid
    env.set_env_from_list(parsed_grid)

    return env


def step_analysis(
    step: dict, rooms_per_side: int = 2
) -> tuple[bool, RoomsMinigridEnv, bool]:
    """
    Analyze a step to check if the agent's action is optimal.

    This function:
    1. Loads the environment from the step's grid_state
    2. REMOVES THE KEY from the environment
    3. Computes optimal actions towards the GOAL (ignoring the key)
    4. Extracts the agent's action from the step
    5. Checks if the agent's action is among the optimal actions towards the goal
    6. If not optimal and key not picked, checks if action is optimal towards the KEY
    7. Returns the optimality result, the environment, and whether action is optimal towards key

    Args:
        step: Dictionary containing 'grid_state', 'agent_action', and 'carrying_key' keys
        rooms_per_side: Number of rooms per side (2 for 2x2, 3 for 3x3). Default is 2.

    Returns:
        Tuple of (is_optimal_to_goal, env, is_optimal_towards_key) where:
        - is_optimal_to_goal: True if the agent's action is optimal towards the goal (ignoring key), False otherwise
        - env: RoomsMinigridEnv instance (original environment)
        - is_optimal_towards_key: True if action is optimal towards key (when key not picked and action not optimal to goal), False otherwise

    Raises:
        KeyError: If required keys are not found in the step
        ValueError: If the action string is not recognized
    """
    # Mapping from action strings to action indices
    # Based on Simple2DNavigationEnv: LEFT=0, RIGHT=1, UP=2, DOWN=3
    action_str_to_int = {
        "LEFT": 0,
        "RIGHT": 1,
        "UP": 2,
        "DOWN": 3,
    }

    # Check if required keys exist in the step
    if "agent_action" not in step:
        raise KeyError("'agent_action' key not found in step")
    if "carrying_key" not in step:
        raise KeyError("'carrying_key' key not found in step")

    # Load the environment from the step (preserving original state)
    env = load_env_from_step(step, rooms_per_side=rooms_per_side, render_mode=None)

    # Get the agent's position
    agent_pos = tuple(env.agent_pos)

    # Get the agent's action from the step
    agent_action_str = step["agent_action"]

    # Convert action string to integer
    if agent_action_str not in action_str_to_int:
        raise ValueError(
            f"Unrecognized action: {agent_action_str}. Expected one of {list(action_str_to_int.keys())}"
        )

    agent_action_int = action_str_to_int[agent_action_str]

    # Get carrying_key status
    carrying_key = step["carrying_key"]

    # Handle both string and boolean values
    if carrying_key not in ("false", False, "true", True):
        raise ValueError(
            f"Invalid carrying_key value: {carrying_key}. Expected 'true', 'false', True, or False"
        )

    # Compute optimal actions towards the GOAL (with key removed from environment)
    env_without_key = remove_key(env)
    optimal_actions_to_goal = compute_optimal_actions_from_position(
        env_without_key, agent_pos
    )

    # Check if the agent's action is optimal towards the goal
    is_optimal_to_goal = agent_action_int in optimal_actions_to_goal

    # Check if action is optimal towards the key (when key not picked and action not optimal to goal)
    is_optimal_towards_key = False
    if not is_optimal_to_goal and carrying_key in ("false", False):
        # Create a version of the environment where we treat the key as the goal
        env_with_key_as_goal = load_env_from_step(
            step, rooms_per_side=rooms_per_side, render_mode=None
        )
        env_with_key_as_goal = replace_key_with_goal(env_with_key_as_goal)
        optimal_actions_towards_key = compute_optimal_actions_from_position(
            env_with_key_as_goal, agent_pos
        )
        is_optimal_towards_key = agent_action_int in optimal_actions_towards_key

    # Debug output
    if DEBUG:
        print("\n" + "=" * 60)
        print("GRID STATE:")
        for row in step["grid_state"]:
            print(row)
        print(f"\nCarrying key: {carrying_key}")
        print(f"Agent action: {agent_action_str} (action={agent_action_int})")

        # Convert optimal action integers back to strings for readability
        action_int_to_str = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}
        optimal_action_strs = [action_int_to_str[a] for a in optimal_actions_to_goal]
        print(
            f"Optimal actions to goal: {optimal_action_strs} (actions={optimal_actions_to_goal})"
        )
        print(f"Is optimal to goal: {is_optimal_to_goal}")

        if not is_optimal_to_goal and carrying_key in ("false", False):
            optimal_towards_key_strs = [
                action_int_to_str[a] for a in optimal_actions_towards_key
            ]
            print(
                f"Optimal actions towards key: {optimal_towards_key_strs} (actions={optimal_actions_towards_key})"
            )
            print(f"Is optimal towards key: {is_optimal_towards_key}")
        print("=" * 60)

    return is_optimal_to_goal, env, is_optimal_towards_key


def analyze_single_trajectory(file_path: str, rooms_per_side: int = 2) -> dict:
    """
    Analyze a single trajectory file for optimality, success, and key pickup.

    Args:
        file_path: Path to the JSON trajectory file
        rooms_per_side: Number of rooms per side (2 for 2x2, 3 for 3x3). Default is 2.

    Returns:
        Dictionary containing complete analysis of the trajectory:
        {
            'filename': str,
            'success': bool,
            'success_info': dict,
            'key_picked': bool,
            'total_steps': int,
            'optimal_steps': int,
            'non_optimal_steps': int,
            'accuracy': float,
            'errors': int
        }
    """
    # Mapping from action strings to action indices
    action_str_to_int = {
        "LEFT": 0,
        "RIGHT": 1,
        "UP": 2,
        "DOWN": 3,
    }

    file_path_obj = Path(file_path)
    filename = file_path_obj.name

    result = {
        "filename": filename,
        "success": False,
        "success_info": {},
        "key_picked": False,
        "total_steps": 0,
        "optimal_steps": 0,
        "non_optimal_steps": 0,
        "non_optimal_but_towards_key": 0,
        "accuracy": 0.0,
        "errors": 0,
    }

    try:
        # Load steps from the JSON file
        steps = load_steps_from_json(file_path)

        if not steps:
            result["success_info"] = {"error": "No steps in trajectory"}
            result["errors"] = 1
            return result

        # Analyze optimality for each step
        result["total_steps"] = len(steps)

        # Track if key was picked up at any point
        for i, single_step in enumerate(steps):
            try:
                # Check if carrying_key is true at any point
                carrying_key = single_step.get("carrying_key", False)
                if carrying_key in ("true", True):
                    result["key_picked"] = True

                # Analyze optimality
                is_optimal, _, is_optimal_towards_key = step_analysis(
                    single_step, rooms_per_side=rooms_per_side
                )

                # Update overall counters
                if is_optimal:
                    result["optimal_steps"] += 1
                else:
                    result["non_optimal_steps"] += 1

                    # Track if non-optimal action was optimal towards key
                    if is_optimal_towards_key:
                        if "non_optimal_but_towards_key" not in result:
                            result["non_optimal_but_towards_key"] = 0
                        result["non_optimal_but_towards_key"] += 1

            except Exception as e:
                if DEBUG:
                    print(f"  Error analyzing step {i} in {filename}: {e}")
                result["errors"] += 1
                continue

        # Calculate overall accuracy
        if result["total_steps"] > 0:
            result["accuracy"] = (result["optimal_steps"] / result["total_steps"]) * 100

        # Check trajectory success
        last_step = steps[-1]
        env = load_env_from_step(
            last_step, rooms_per_side=rooms_per_side, render_mode=None
        )

        if "agent_action" in last_step:
            agent_action_str = last_step["agent_action"]

            if agent_action_str in action_str_to_int:
                agent_action_int = action_str_to_int[agent_action_str]
                _, reward, terminated, truncated, info = env.step(agent_action_int)

                result["success"] = terminated and reward > 0
                result["success_info"] = {
                    "last_action": agent_action_str,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "agent_final_pos": tuple(env.agent_pos)
                    if hasattr(env, "agent_pos")
                    else None,
                }
            else:
                result["success_info"] = {
                    "error": f"Unrecognized action: {agent_action_str}"
                }
        else:
            result["success_info"] = {"error": "'agent_action' not found in last step"}

    except Exception as e:
        result["success_info"] = {"error": str(e)}
        result["errors"] = max(result["errors"], 1)

    return result


def analyze_directory(dir_path: str, rooms_per_side: int = 2) -> dict[str, dict]:
    """
    Analyze all JSON trajectory files in a directory.

    Args:
        dir_path: Path to the directory containing JSON trajectory files
        rooms_per_side: Number of rooms per side (2 for 2x2, 3 for 3x3). Default is 2.

    Returns:
        Dictionary mapping file names to their complete analysis results.
    """
    dir_path_obj = Path(dir_path)

    if not dir_path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path_obj.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    # Find all JSON files in the directory
    json_files = sorted(dir_path_obj.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return {}

    results = {}

    for json_file in json_files:
        trajectory_result = analyze_single_trajectory(
            str(json_file), rooms_per_side=rooms_per_side
        )
        results[trajectory_result["filename"]] = trajectory_result

    return results


def compute_aggregate_statistics(results: dict[str, dict]) -> dict:
    """
    Compute aggregate statistics including means and standard deviations.

    Args:
        results: Dictionary mapping file names to their analysis results

    Returns:
        Dictionary containing aggregate statistics with means and standard deviations
    """
    if not results:
        return {}

    # Collect per-trajectory metrics
    accuracies = []
    success_flags = []
    key_picked_flags = []
    towards_key_ratios = []

    # Aggregate counts for non-optimal steps towards key
    total_non_optimal_but_towards_key = 0
    total_non_optimal = 0

    for trajectory_result in results.values():
        accuracies.append(trajectory_result["accuracy"])
        success_flags.append(1.0 if trajectory_result["success"] else 0.0)
        key_picked_flags.append(1.0 if trajectory_result["key_picked"] else 0.0)

        # Aggregate non-optimal steps towards key
        non_optimal = trajectory_result["non_optimal_steps"]
        towards_key = trajectory_result.get("non_optimal_but_towards_key", 0)

        total_non_optimal += non_optimal
        total_non_optimal_but_towards_key += towards_key

        # Per-trajectory ratio (only if there are non-optimal steps)
        if non_optimal > 0:
            towards_key_ratios.append((towards_key / non_optimal) * 100)

    # Compute overall statistics
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0

    success_rate_mean = np.mean(success_flags) * 100
    success_rate_std = (
        np.std(success_flags, ddof=1) * 100 if len(success_flags) > 1 else 0.0
    )

    key_pickup_rate_mean = np.mean(key_picked_flags) * 100
    key_pickup_rate_std = (
        np.std(key_picked_flags, ddof=1) * 100 if len(key_picked_flags) > 1 else 0.0
    )

    # Compute towards key statistics
    towards_key_ratio_mean = np.mean(towards_key_ratios) if towards_key_ratios else 0.0
    towards_key_ratio_std = (
        np.std(towards_key_ratios, ddof=1) if len(towards_key_ratios) > 1 else 0.0
    )

    # Overall towards key percentage (across all trajectories)
    overall_towards_key_percentage = (
        (total_non_optimal_but_towards_key / total_non_optimal * 100)
        if total_non_optimal > 0
        else 0.0
    )

    return {
        "total_accuracy": {"mean": accuracy_mean, "std": accuracy_std},
        "success_rate": {"mean": success_rate_mean, "std": success_rate_std},
        "key_pickup_rate": {"mean": key_pickup_rate_mean, "std": key_pickup_rate_std},
        "non_optimal_towards_key": {
            "total_towards_key": total_non_optimal_but_towards_key,
            "total_non_optimal": total_non_optimal,
            "overall_percentage": overall_towards_key_percentage,
            "mean_per_trajectory": towards_key_ratio_mean,
            "std_per_trajectory": towards_key_ratio_std,
        },
        "n_trajectories": len(results),
    }


def save_statistics(statistics: dict, output_path: str) -> None:
    """
    Save aggregate statistics to a JSON file.

    Args:
        statistics: Dictionary containing aggregate statistics
        output_path: Path to save the JSON file
    """
    with open(output_path, "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"Statistics saved to: {output_path}")


if __name__ == "__main__":
    # Analyze all JSON files in the directory
    print("=" * 80)
    print("TRAJECTORY ANALYSIS (KEY-NO-DOOR ENVIRONMENT)")
    print("=" * 80)
    print(f"Analyzing directory: {DIR_PATH}")
    print()

    results = analyze_directory(DIR_PATH, rooms_per_side=2)

    if not results:
        print("No trajectories found to analyze.")
        exit(0)

    # Aggregate statistics
    total_files = len(results)
    total_steps = 0
    total_optimal = 0
    total_non_optimal_but_towards_key = 0
    total_errors = 0
    successful_trajectories = 0
    trajectories_with_key_picked = 0

    for trajectory_result in results.values():
        total_steps += trajectory_result["total_steps"]
        total_optimal += trajectory_result["optimal_steps"]
        total_non_optimal_but_towards_key += trajectory_result.get(
            "non_optimal_but_towards_key", 0
        )
        total_errors += trajectory_result["errors"]

        if trajectory_result["success"]:
            successful_trajectories += 1

        if trajectory_result["key_picked"]:
            trajectories_with_key_picked += 1

    # Calculate metrics
    overall_accuracy = (total_optimal / total_steps * 100) if total_steps > 0 else 0.0
    success_ratio = (
        (successful_trajectories / total_files * 100) if total_files > 0 else 0.0
    )
    key_pickup_ratio = (
        (trajectories_with_key_picked / total_files * 100) if total_files > 0 else 0.0
    )
    total_non_optimal = total_steps - total_optimal
    towards_key_ratio = (
        (total_non_optimal_but_towards_key / total_non_optimal * 100)
        if total_non_optimal > 0
        else 0.0
    )

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total trajectories analyzed: {total_files}")
    print(
        f"Successful trajectories: {successful_trajectories}/{total_files} ({success_ratio:.2f}%)"
    )
    print(
        f"Trajectories with key picked: {trajectories_with_key_picked}/{total_files} ({key_pickup_ratio:.2f}%)"
    )
    print(f"Total steps: {total_steps}")
    print(
        f"Overall optimality: {total_optimal}/{total_steps} ({overall_accuracy:.2f}%)"
    )
    print(
        f"Non-optimal steps towards key: {total_non_optimal_but_towards_key}/{total_non_optimal} ({towards_key_ratio:.2f}%)"
    )

    if total_errors > 0:
        print(f"Total errors encountered: {total_errors}")

    # Print detailed per-file results
    print("\n" + "=" * 80)
    print("PER-TRAJECTORY DETAILS")
    print("=" * 80)
    print(
        f"{'Filename':<40} {'Success':>8} {'Key':>5} {'Accuracy':>10} {'→Key':>8} {'Steps':>8}"
    )
    print("-" * 80)

    for filename, trajectory_result in sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    ):
        success_mark = "✓" if trajectory_result["success"] else "✗"
        key_mark = "✓" if trajectory_result["key_picked"] else "✗"
        towards_key = trajectory_result.get("non_optimal_but_towards_key", 0)
        print(
            f"{filename:<40} {success_mark:>8} {key_mark:>5} {trajectory_result['accuracy']:>9.2f}% {towards_key:>8} {trajectory_result['total_steps']:>8}"
        )

    # Compute aggregate statistics with standard deviations
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS (Mean ± Std)")
    print("=" * 80)

    aggregate_stats = compute_aggregate_statistics(results)

    if aggregate_stats:
        print(
            f"Total Accuracy: {aggregate_stats['total_accuracy']['mean']:.2f}% ± {aggregate_stats['total_accuracy']['std']:.2f}%"
        )
        print(
            f"Success Rate: {aggregate_stats['success_rate']['mean']:.2f}% ± {aggregate_stats['success_rate']['std']:.2f}%"
        )
        print(
            f"Key Pickup Rate: {aggregate_stats['key_pickup_rate']['mean']:.2f}% ± {aggregate_stats['key_pickup_rate']['std']:.2f}%"
        )

        # Print non-optimal towards key statistics
        towards_key_stats = aggregate_stats["non_optimal_towards_key"]
        print(
            f"Non-optimal steps towards key: {towards_key_stats['total_towards_key']}/{towards_key_stats['total_non_optimal']} ({towards_key_stats['overall_percentage']:.2f}%)"
        )
        print(
            f"  Per-trajectory mean: {towards_key_stats['mean_per_trajectory']:.2f}% ± {towards_key_stats['std_per_trajectory']:.2f}%"
        )

        # Save statistics to file
        stats_output_path = Path(DIR_PATH) / "analysis_statistics.json"
        save_statistics(aggregate_stats, str(stats_output_path))
        print()
