import json
from pathlib import Path
import numpy as np
from papers.papers_code.reveng.src.reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv
from papers.papers_code.reveng.src.reveng.environment_generator.utils import (
    compute_optimal_actions_from_position,
    remove_door,
    replace_key_with_goal,
)

DEBUG = False
REPO_ROOT = Path(__file__).resolve().parents[4]
DIR_PATH = REPO_ROOT / "trajectories_key_door"


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

    This function supports all symbols including keys and doors:
    - '#' (wall)
    - '_' (empty)
    - 'A' (agent)
    - 'G' (goal)
    - 'K' (key)
    - 'D' (door, locked)
    - 'O' (door, open)

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
) -> tuple[bool, RoomsMinigridEnv, str]:
    """
    Analyze a step to check if the agent's action is optimal.

    This function:
    1. Loads the environment from the step's grid_state
    2. If carrying_key is "false": replaces the key with a goal (removes original goal)
       If carrying_key is "true": removes the door from the environment
    3. Extracts the agent's action from the step
    4. Computes all optimal actions from the agent's position
    5. Checks if the agent's action is among the optimal actions
    6. Returns the optimality result, the modified environment, and the stage

    Args:
        step: Dictionary containing 'grid_state', 'agent_action', and 'carrying_key' keys
        rooms_per_side: Number of rooms per side (2 for 2x2, 3 for 3x3). Default is 2.

    Returns:
        Tuple of (is_optimal, env, stage) where:
        - is_optimal: True if the agent's action is optimal, False otherwise
        - env: RoomsMinigridEnv with the appropriate modification based on carrying_key
        - stage: String indicating the stage ("before_key", "after_key_before_door", or "after_door")

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

    # Load the environment from the step
    env = load_env_from_step(step, rooms_per_side=rooms_per_side, render_mode=None)

    # Determine the stage and apply appropriate transformation based on carrying_key
    carrying_key = step["carrying_key"]

    # Check if door exists in the grid_state
    door_exists = any("D" in row for row in step["grid_state"])

    # Handle both string and boolean values
    if carrying_key in ("false", False):
        stage = "before_key"
        # Agent is not carrying the key, so replace key with goal
        env = replace_key_with_goal(env)
    elif carrying_key in ("true", True):
        if door_exists:
            stage = "after_key_before_door"
        else:
            stage = "after_door"
        # Agent is carrying the key, so remove the door
        env = remove_door(env)
    else:
        raise ValueError(
            f"Invalid carrying_key value: {carrying_key}. Expected 'true', 'false', True, or False"
        )

    # Get the agent's action from the step
    agent_action_str = step["agent_action"]

    # Convert action string to integer
    if agent_action_str not in action_str_to_int:
        raise ValueError(
            f"Unrecognized action: {agent_action_str}. Expected one of {list(action_str_to_int.keys())}"
        )

    agent_action_int = action_str_to_int[agent_action_str]

    # Get the agent's position
    agent_pos = tuple(env.agent_pos)

    # Compute all optimal actions from the agent's position
    optimal_actions = compute_optimal_actions_from_position(env, agent_pos)

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
        optimal_action_strs = [action_int_to_str[a] for a in optimal_actions]
        print(f"Optimal actions: {optimal_action_strs} (actions={optimal_actions})")
        print(f"Is optimal: {agent_action_int in optimal_actions}")
        print("=" * 60)

    # Check if the agent's action is in the optimal actions
    is_optimal = agent_action_int in optimal_actions

    return is_optimal, env, stage


def analyze_single_trajectory(file_path: str, rooms_per_side: int = 2) -> dict:
    """
    Analyze a single trajectory file for both optimality and success.

    Args:
        file_path: Path to the JSON trajectory file
        rooms_per_side: Number of rooms per side (2 for 2x2, 3 for 3x3). Default is 2.

    Returns:
        Dictionary containing complete analysis of the trajectory:
        {
            'filename': str,
            'success': bool,
            'success_info': dict,
            'total_steps': int,
            'optimal_steps': int,
            'non_optimal_steps': int,
            'accuracy': float,
            'errors': int,
            'stages': {
                'before_key': {'total': int, 'optimal': int, 'accuracy': float},
                'after_key_before_door': {'total': int, 'optimal': int, 'accuracy': float},
                'after_door': {'total': int, 'optimal': int, 'accuracy': float}
            }
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
        "total_steps": 0,
        "optimal_steps": 0,
        "non_optimal_steps": 0,
        "accuracy": 0.0,
        "errors": 0,
        "stages": {
            "before_key": {"total": 0, "optimal": 0, "accuracy": 0.0},
            "after_key_before_door": {"total": 0, "optimal": 0, "accuracy": 0.0},
            "after_door": {"total": 0, "optimal": 0, "accuracy": 0.0},
        },
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

        for i, single_step in enumerate(steps):
            try:
                is_optimal, _, stage = step_analysis(
                    single_step, rooms_per_side=rooms_per_side
                )

                # Update overall counters
                if is_optimal:
                    result["optimal_steps"] += 1
                else:
                    result["non_optimal_steps"] += 1

                # Update stage-specific counters
                result["stages"][stage]["total"] += 1
                if is_optimal:
                    result["stages"][stage]["optimal"] += 1

            except Exception as e:
                if DEBUG:
                    print(f"  Error analyzing step {i} in {filename}: {e}")
                result["errors"] += 1
                continue

        # Calculate overall accuracy
        if result["total_steps"] > 0:
            result["accuracy"] = (result["optimal_steps"] / result["total_steps"]) * 100

        # Calculate stage-specific accuracies
        for stage in result["stages"]:
            stage_total = result["stages"][stage]["total"]
            stage_optimal = result["stages"][stage]["optimal"]
            if stage_total > 0:
                result["stages"][stage]["accuracy"] = (
                    stage_optimal / stage_total
                ) * 100

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
    stage_accuracies = {"before_key": [], "after_key_before_door": [], "after_door": []}

    for trajectory_result in results.values():
        accuracies.append(trajectory_result["accuracy"])
        success_flags.append(1.0 if trajectory_result["success"] else 0.0)

        for stage_name, stage_data in trajectory_result["stages"].items():
            stage_accuracies[stage_name].append(stage_data["accuracy"])

    # Compute overall statistics
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0

    success_rate_mean = np.mean(success_flags) * 100
    success_rate_std = (
        np.std(success_flags, ddof=1) * 100 if len(success_flags) > 1 else 0.0
    )

    # Compute stage-specific statistics
    stage_statistics = {}
    for stage_name, stage_acc_list in stage_accuracies.items():
        # Filter out trajectories with no steps in this stage (accuracy = 0.0 and total = 0)
        valid_accuracies = [
            acc
            for acc, result in zip(stage_acc_list, results.values())
            if result["stages"][stage_name]["total"] > 0
        ]

        if valid_accuracies:
            stage_statistics[stage_name] = {
                "mean": np.mean(valid_accuracies),
                "std": np.std(valid_accuracies, ddof=1)
                if len(valid_accuracies) > 1
                else 0.0,
                "n_trajectories": len(valid_accuracies),
            }
        else:
            stage_statistics[stage_name] = {
                "mean": 0.0,
                "std": 0.0,
                "n_trajectories": 0,
            }

    return {
        "total_accuracy": {"mean": accuracy_mean, "std": accuracy_std},
        "success_rate": {"mean": success_rate_mean, "std": success_rate_std},
        "stage_optimality": stage_statistics,
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
    print("TRAJECTORY ANALYSIS")
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
    total_errors = 0
    successful_trajectories = 0

    # Aggregate stage statistics
    stage_stats = {
        "before_key": {"total": 0, "optimal": 0},
        "after_key_before_door": {"total": 0, "optimal": 0},
        "after_door": {"total": 0, "optimal": 0},
    }

    for trajectory_result in results.values():
        total_steps += trajectory_result["total_steps"]
        total_optimal += trajectory_result["optimal_steps"]
        total_errors += trajectory_result["errors"]

        if trajectory_result["success"]:
            successful_trajectories += 1

        # Aggregate stage stats
        for stage_name, stage_data in trajectory_result["stages"].items():
            stage_stats[stage_name]["total"] += stage_data["total"]
            stage_stats[stage_name]["optimal"] += stage_data["optimal"]

    # Calculate metrics
    overall_accuracy = (total_optimal / total_steps * 100) if total_steps > 0 else 0.0
    success_ratio = (
        (successful_trajectories / total_files * 100) if total_files > 0 else 0.0
    )

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total trajectories analyzed: {total_files}")
    print(
        f"Successful trajectories: {successful_trajectories}/{total_files} ({success_ratio:.2f}%)"
    )
    print(f"Total steps: {total_steps}")
    print(f"Overall accuracy: {total_optimal}/{total_steps} ({overall_accuracy:.2f}%)")

    if total_errors > 0:
        print(f"Total errors encountered: {total_errors}")

    # Print stage-specific optimality
    print("\n" + "=" * 80)
    print("OPTIMALITY PER STAGE")
    print("=" * 80)

    for stage_name, stage_data in stage_stats.items():
        if stage_data["total"] > 0:
            stage_accuracy = (stage_data["optimal"] / stage_data["total"]) * 100
            print(
                f"{stage_name:25s}: {stage_data['optimal']:4d}/{stage_data['total']:4d} ({stage_accuracy:6.2f}%)"
            )
        else:
            print(f"{stage_name:25s}: No steps in this stage")

    # Print detailed per-file results
    print("\n" + "=" * 80)
    print("PER-TRAJECTORY DETAILS")
    print("=" * 80)
    print(f"{'Filename':<50} {'Success':>8} {'Accuracy':>10} {'Steps':>8}")
    print("-" * 80)

    for filename, trajectory_result in sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    ):
        success_mark = "✓" if trajectory_result["success"] else "✗"
        print(
            f"{filename:<50} {success_mark:>8} {trajectory_result['accuracy']:>9.2f}% {trajectory_result['total_steps']:>8}"
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
        print("\nStage Optimality:")
        for stage_name, stage_stats in aggregate_stats["stage_optimality"].items():
            print(
                f"  {stage_name:25s}: {stage_stats['mean']:6.2f}% ± {stage_stats['std']:5.2f}% (n={stage_stats['n_trajectories']})"
            )

        # Save statistics to file
        output_path = Path(DIR_PATH) / "analysis_statistics.json"
        save_statistics(aggregate_stats, str(output_path))
        print()
