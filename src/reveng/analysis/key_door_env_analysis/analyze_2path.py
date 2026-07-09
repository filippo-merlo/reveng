import json
from pathlib import Path
import numpy as np
from papers.papers_code.reveng.src.reveng.environment_generator.key_minigrid import Key2PathMinigridEnv
from papers.papers_code.reveng.src.reveng.environment_generator.utils import (
    compute_optimal_actions_from_position,
    replace_key_with_goal,
    remove_key,
)

DEBUG = False
REPO_ROOT = Path(__file__).resolve().parents[4]
DIR_PATH = REPO_ROOT / "trajectories_key2path"


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


def key_exists_in_grid(grid_state: list[str]) -> bool:
    """
    Check if the key ('K') exists in the grid state.

    Args:
        grid_state: List of strings representing the grid with coordinate labels

    Returns:
        True if 'K' is found in the grid, False otherwise
    """
    for row in grid_state:
        if "K" in row:
            return True
    return False


def load_env_from_step(
    step: dict, size: int = 11, render_mode: str | None = None
) -> Key2PathMinigridEnv:
    """
    Load a Key2PathMinigridEnv from a step's grid_state.

    Args:
        step: Dictionary containing a 'grid_state' key with the grid representation
        size: Size of the grid (default is 11).
        render_mode: Rendering mode ('human', 'rgb_array', or None). Default is None.

    Returns:
        Configured Key2PathMinigridEnv instance

    Raises:
        KeyError: If 'grid_state' key is not found in the step
    """
    if "grid_state" not in step:
        raise KeyError("'grid_state' key not found in step")

    grid_state = step["grid_state"]
    parsed_grid = parse_grid_state(grid_state)

    # Create environment
    env = Key2PathMinigridEnv(size=size, render_mode=render_mode)

    # Reset to initialize the environment
    env.reset()

    # Set the environment from the parsed grid
    env.set_env_from_list(parsed_grid)

    return env


def step_analysis(step: dict, size: int = 11) -> tuple[bool, Key2PathMinigridEnv, bool]:
    """
    Analyze a step to check if the agent's action is optimal.

    This function:
    1. Loads the environment from the step's grid_state
    2. REMOVES THE KEY from the environment
    3. Computes optimal actions towards the GOAL (ignoring the key)
    4. Extracts the agent's action from the step
    5. Checks if the agent's action is among the optimal actions towards the goal
    6. If not optimal and key exists in grid, checks if action is optimal towards the KEY
    7. Returns the optimality result, the environment, and whether action is optimal towards key

    Args:
        step: Dictionary containing 'grid_state' and 'agent_action' keys
        size: Size of the grid (default is 11).

    Returns:
        Tuple of (is_optimal_to_goal, env, is_optimal_towards_key)
    """
    # Mapping from action strings to action indices
    action_str_to_int = {
        "LEFT": 0,
        "RIGHT": 1,
        "UP": 2,
        "DOWN": 3,
    }

    # Check if required keys exist in the step
    if "agent_action" not in step:
        raise KeyError("'agent_action' key not found in step")
    if "grid_state" not in step:
        raise KeyError("'grid_state' key not found in step")

    # Load the environment from the step (preserving original state)
    env = load_env_from_step(step, size=size, render_mode=None)

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

    # Check if key exists in the grid state
    key_exists = key_exists_in_grid(step["grid_state"])

    # Compute optimal actions towards the GOAL (with key removed from environment)
    env_without_key = remove_key(env)
    optimal_actions_to_goal = compute_optimal_actions_from_position(
        env_without_key, agent_pos
    )

    # Check if the agent's action is optimal towards the goal
    is_optimal_to_goal = agent_action_int in optimal_actions_to_goal

    # Check if action is optimal towards the key (when key exists in grid and action not optimal to goal)
    is_optimal_towards_key = False
    if not is_optimal_to_goal and key_exists:
        # Create a version of the environment where we treat the key as the goal
        env_with_key_as_goal = load_env_from_step(step, size=size, render_mode=None)
        env_with_key_as_goal = replace_key_with_goal(env_with_key_as_goal)
        optimal_actions_towards_key = compute_optimal_actions_from_position(
            env_with_key_as_goal, agent_pos
        )
        is_optimal_towards_key = agent_action_int in optimal_actions_towards_key

    return is_optimal_to_goal, env, is_optimal_towards_key


def analyze_single_trajectory(file_path: str, size: int = 11) -> dict:
    """
    Analyze a single trajectory file for optimality, success, and key pickup.

    Args:
        file_path: Path to the JSON trajectory file
        size: Size of the grid (default is 11).

    Returns:
        Dictionary containing complete analysis of the trajectory
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

        # Check if key was picked up by checking if 'K' doesn't exist in the last step's grid
        last_step_grid_state = steps[-1].get("grid_state", [])
        if not key_exists_in_grid(last_step_grid_state):
            result["key_picked"] = True

        for i, single_step in enumerate(steps):
            try:
                # Analyze optimality
                is_optimal, _, is_optimal_towards_key = step_analysis(
                    single_step, size=size
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
        env = load_env_from_step(last_step, size=size, render_mode=None)

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


def get_visited_positions(file_path: str, size: int = 11) -> set[tuple[int, int]]:
    """
    Extract the set of all visited positions (agent_pos) from a trajectory.

    Args:
        file_path: Path to the JSON trajectory file
        size: Size of the grid (default is 11).

    Returns:
        Set of tuples representing visited positions (x, y)
    """
    visited = set()

    try:
        steps = load_steps_from_json(file_path)

        for step in steps:
            try:
                # Load environment from the step to get agent position
                env = load_env_from_step(step, size=size, render_mode=None)
                agent_pos = tuple(env.agent_pos)
                visited.add(agent_pos)
            except Exception as e:
                if DEBUG:
                    print(f"Error loading environment from step: {e}")
                continue
    except Exception as e:
        if DEBUG:
            print(f"Error extracting positions from {file_path}: {e}")

    return visited


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """
    Compute the Jaccard Similarity Index between two sets.

    The Jaccard Similarity is defined as:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity (0.0 to 1.0), or 0.0 if both sets are empty
    """
    if len(set1) == 0 and len(set2) == 0:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def compute_trajectory_pair_jaccard(
    key_file: str, no_key_file: str, size: int = 11
) -> dict:
    """
    Compute the Jaccard Similarity Index for a trajectory pair.

    Compares the sets of visited positions between the trajectory with key
    and the trajectory without key.

    Args:
        key_file: Path to the trajectory file with key
        no_key_file: Path to the trajectory file without key
        size: Size of the grid (default is 11).

    Returns:
        Dictionary containing:
        - 'jaccard_similarity': The Jaccard similarity index (0.0 to 1.0)
        - 'with_key_positions': Set of positions visited in trajectory with key
        - 'without_key_positions': Set of positions visited in trajectory without key
        - 'intersection_size': Number of common positions
        - 'union_size': Total number of unique positions
    """
    with_key_positions = get_visited_positions(key_file, size=size)
    without_key_positions = get_visited_positions(no_key_file, size=size)

    jaccard = compute_jaccard_similarity(with_key_positions, without_key_positions)

    intersection = with_key_positions & without_key_positions
    union = with_key_positions | without_key_positions

    return {
        "jaccard_similarity": jaccard,
        "with_key_positions": with_key_positions,
        "without_key_positions": without_key_positions,
        "intersection_size": len(intersection),
        "union_size": len(union),
        "with_key_count": len(with_key_positions),
        "without_key_count": len(without_key_positions),
    }


def find_trajectory_pairs(dir_path: str) -> list[tuple[str, str]]:
    """
    Find pairs of trajectory files (with key and without key).

    Args:
        dir_path: Path to the directory containing JSON trajectory files

    Returns:
        List of tuples (key_file_path, no_key_file_path)
    """
    dir_path_obj = Path(dir_path)

    if not dir_path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path_obj.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    # Find all JSON files without "_no_key" suffix
    all_files = sorted(dir_path_obj.glob("*.json"))
    key_files = [f for f in all_files if not f.stem.endswith("_no_key")]

    pairs = []
    for key_file in key_files:
        # Construct the no_key filename
        no_key_file = dir_path_obj / f"{key_file.stem}_no_key.json"

        if no_key_file.exists():
            pairs.append((str(key_file), str(no_key_file)))
        else:
            if DEBUG:
                print(f"Warning: No matching no_key file for {key_file.name}")

    return pairs


def analyze_paired_trajectories(
    dir_path: str, size: int = 11, include_jaccard: bool = True
) -> dict:
    """
    Analyze paired trajectories (with key vs without key) and compute comparison statistics.

    Args:
        dir_path: Path to the directory containing JSON trajectory files
        size: Size of the grid (default is 11).
        include_jaccard: Whether to compute Jaccard similarity for visited positions (default is True).

    Returns:
        Dictionary containing paired analysis results and statistics
    """
    pairs = find_trajectory_pairs(dir_path)

    if not pairs:
        print(f"No trajectory pairs found in {dir_path}")
        return {}

    results = {"pairs": [], "with_key": {}, "without_key": {}}

    for key_file, no_key_file in pairs:
        # Analyze both trajectories
        key_result = analyze_single_trajectory(key_file, size=size)
        no_key_result = analyze_single_trajectory(no_key_file, size=size)

        # Extract base ID from filename
        key_filename = Path(key_file).stem

        pair_result = {
            "id": key_filename,
            "with_key": key_result,
            "without_key": no_key_result,
            "accuracy_diff": key_result["accuracy"] - no_key_result["accuracy"],
        }

        # Compute Jaccard similarity if requested
        if include_jaccard:
            jaccard_result = compute_trajectory_pair_jaccard(
                key_file, no_key_file, size=size
            )
            pair_result["jaccard_similarity"] = jaccard_result["jaccard_similarity"]
            pair_result["intersection_size"] = jaccard_result["intersection_size"]
            pair_result["union_size"] = jaccard_result["union_size"]
            pair_result["with_key_positions_count"] = jaccard_result["with_key_count"]
            pair_result["without_key_positions_count"] = jaccard_result[
                "without_key_count"
            ]

        results["pairs"].append(pair_result)
        results["with_key"][key_filename] = key_result
        results["without_key"][key_filename + "_no_key"] = no_key_result

    return results


def compute_paired_statistics(paired_results: dict) -> dict:
    """
    Compute statistics comparing trajectories with key vs without key.

    Args:
        paired_results: Dictionary from analyze_paired_trajectories

    Returns:
        Dictionary containing comparison statistics
    """
    if not paired_results or "pairs" not in paired_results:
        return {}

    pairs = paired_results["pairs"]

    # Collect metrics
    with_key_accuracies = []
    without_key_accuracies = []
    accuracy_diffs = []

    with_key_success = []
    without_key_success = []

    with_key_picked = []

    jaccard_similarities = []
    jaccard_key_picked = []  # Jaccard for trajectories where key was picked
    jaccard_key_not_picked = []  # Jaccard for trajectories where key was not picked

    for pair in pairs:
        with_key_accuracies.append(pair["with_key"]["accuracy"])
        without_key_accuracies.append(pair["without_key"]["accuracy"])
        accuracy_diffs.append(pair["accuracy_diff"])

        with_key_success.append(1.0 if pair["with_key"]["success"] else 0.0)
        without_key_success.append(1.0 if pair["without_key"]["success"] else 0.0)

        key_picked = pair["with_key"]["key_picked"]
        with_key_picked.append(1.0 if key_picked else 0.0)

        # Collect Jaccard similarity if available
        if "jaccard_similarity" in pair:
            jaccard_similarities.append(pair["jaccard_similarity"])

            # Split by whether key was picked or not
            if key_picked:
                jaccard_key_picked.append(pair["jaccard_similarity"])
            else:
                jaccard_key_not_picked.append(pair["jaccard_similarity"])

    # Compute statistics
    stats = {
        "n_pairs": len(pairs),
        "with_key": {
            "accuracy_mean": np.mean(with_key_accuracies),
            "accuracy_std": np.std(with_key_accuracies, ddof=1)
            if len(with_key_accuracies) > 1
            else 0.0,
            "success_rate": np.mean(with_key_success) * 100,
            "success_rate_std": np.std(with_key_success, ddof=1) * 100
            if len(with_key_success) > 1
            else 0.0,
            "key_pickup_rate": np.mean(with_key_picked) * 100,
            "key_pickup_rate_std": np.std(with_key_picked, ddof=1) * 100
            if len(with_key_picked) > 1
            else 0.0,
        },
        "without_key": {
            "accuracy_mean": np.mean(without_key_accuracies),
            "accuracy_std": np.std(without_key_accuracies, ddof=1)
            if len(without_key_accuracies) > 1
            else 0.0,
            "success_rate": np.mean(without_key_success) * 100,
            "success_rate_std": np.std(without_key_success, ddof=1) * 100
            if len(without_key_success) > 1
            else 0.0,
        },
        "accuracy_difference": {
            "mean": np.mean(accuracy_diffs),
            "std": np.std(accuracy_diffs, ddof=1) if len(accuracy_diffs) > 1 else 0.0,
            "median": np.median(accuracy_diffs),
        },
    }

    # Add Jaccard similarity statistics if available
    if jaccard_similarities:
        stats["jaccard_similarity"] = {
            "mean": np.mean(jaccard_similarities),
            "std": np.std(jaccard_similarities, ddof=1)
            if len(jaccard_similarities) > 1
            else 0.0,
            "median": np.median(jaccard_similarities),
            "min": np.min(jaccard_similarities),
            "max": np.max(jaccard_similarities),
        }

    # Add Jaccard similarity for key picked cases
    if jaccard_key_picked:
        stats["jaccard_similarity_key_picked"] = {
            "n": len(jaccard_key_picked),
            "mean": np.mean(jaccard_key_picked),
            "std": np.std(jaccard_key_picked, ddof=1)
            if len(jaccard_key_picked) > 1
            else 0.0,
            "median": np.median(jaccard_key_picked),
            "min": np.min(jaccard_key_picked),
            "max": np.max(jaccard_key_picked),
        }

    # Add Jaccard similarity for key not picked cases
    if jaccard_key_not_picked:
        stats["jaccard_similarity_key_not_picked"] = {
            "n": len(jaccard_key_not_picked),
            "mean": np.mean(jaccard_key_not_picked),
            "std": np.std(jaccard_key_not_picked, ddof=1)
            if len(jaccard_key_not_picked) > 1
            else 0.0,
            "median": np.median(jaccard_key_not_picked),
            "min": np.min(jaccard_key_not_picked),
            "max": np.max(jaccard_key_not_picked),
        }

    return stats


def print_paired_comparison(paired_results: dict, stats: dict) -> None:
    """
    Print a detailed comparison of paired trajectories.

    Args:
        paired_results: Dictionary from analyze_paired_trajectories
        stats: Dictionary from compute_paired_statistics
    """
    if not paired_results or "pairs" not in paired_results:
        print("No paired results to display")
        return

    print("\n" + "=" * 100)
    print("PAIRED TRAJECTORY COMPARISON (KEY vs NO-KEY)")
    print("=" * 100)

    # Check if Jaccard similarity is included
    has_jaccard = (
        len(paired_results["pairs"]) > 0
        and "jaccard_similarity" in paired_results["pairs"][0]
    )

    # Print per-pair details
    if has_jaccard:
        print(
            f"{'ID':<35} {'With Key':>12} {'Without Key':>12} {'Difference':>12} {'Key Pick':>10} {'Jaccard':>10}"
        )
        print("-" * 105)
        for pair in sorted(
            paired_results["pairs"], key=lambda x: x["accuracy_diff"], reverse=True
        ):
            key_picked = "Yes" if pair["with_key"]["key_picked"] else "No"
            print(
                f"{pair['id']:<35} {pair['with_key']['accuracy']:>11.2f}% "
                f"{pair['without_key']['accuracy']:>11.2f}% "
                f"{pair['accuracy_diff']:>11.2f}% "
                f"{key_picked:>10} "
                f"{pair['jaccard_similarity']:>9.4f}"
            )
    else:
        print(
            f"{'ID':<45} {'With Key':>12} {'Without Key':>12} {'Difference':>12} {'Key Pick':>10}"
        )
        print("-" * 105)
        for pair in sorted(
            paired_results["pairs"], key=lambda x: x["accuracy_diff"], reverse=True
        ):
            key_picked = "Yes" if pair["with_key"]["key_picked"] else "No"
            print(
                f"{pair['id']:<45} {pair['with_key']['accuracy']:>11.2f}% "
                f"{pair['without_key']['accuracy']:>11.2f}% "
                f"{pair['accuracy_diff']:>11.2f}% "
                f"{key_picked:>10}"
            )

    # Print aggregate statistics
    print("\n" + "=" * 100)
    print("AGGREGATE STATISTICS (Mean +/- Std)")
    print("=" * 100)
    print(f"Number of trajectory pairs: {stats['n_pairs']}")
    print()

    print("WITH KEY:")
    print(
        f"  Accuracy: {stats['with_key']['accuracy_mean']:.2f}% +/- {stats['with_key']['accuracy_std']:.2f}%"
    )
    print(
        f"  Success Rate: {stats['with_key']['success_rate']:.2f}% +/- {stats['with_key']['success_rate_std']:.2f}%"
    )
    print(
        f"  Key Pickup Rate: {stats['with_key']['key_pickup_rate']:.2f}% +/- {stats['with_key']['key_pickup_rate_std']:.2f}%"
    )
    print()

    print("WITHOUT KEY:")
    print(
        f"  Accuracy: {stats['without_key']['accuracy_mean']:.2f}% +/- {stats['without_key']['accuracy_std']:.2f}%"
    )
    print(
        f"  Success Rate: {stats['without_key']['success_rate']:.2f}% +/- {stats['without_key']['success_rate_std']:.2f}%"
    )
    print()

    print("ACCURACY DIFFERENCE (With Key - Without Key):")
    print(
        f"  Mean: {stats['accuracy_difference']['mean']:.2f}% +/- {stats['accuracy_difference']['std']:.2f}%"
    )
    print(f"  Median: {stats['accuracy_difference']['median']:.2f}%")
    print()

    # Print Jaccard similarity statistics if available
    if "jaccard_similarity" in stats:
        print("JACCARD SIMILARITY (Visited Positions Overlap):")
        print(
            f"  Overall Mean: {stats['jaccard_similarity']['mean']:.4f} +/- {stats['jaccard_similarity']['std']:.4f}"
        )
        print(f"  Overall Median: {stats['jaccard_similarity']['median']:.4f}")
        print(
            f"  Overall Range: [{stats['jaccard_similarity']['min']:.4f}, {stats['jaccard_similarity']['max']:.4f}]"
        )
        print()

        # Print separate statistics for key picked vs not picked
        if "jaccard_similarity_key_picked" in stats:
            print(f"  Key Picked (n={stats['jaccard_similarity_key_picked']['n']}):")
            print(
                f"    Mean: {stats['jaccard_similarity_key_picked']['mean']:.4f} +/- {stats['jaccard_similarity_key_picked']['std']:.4f}"
            )
            print(f"    Median: {stats['jaccard_similarity_key_picked']['median']:.4f}")
            print(
                f"    Range: [{stats['jaccard_similarity_key_picked']['min']:.4f}, {stats['jaccard_similarity_key_picked']['max']:.4f}]"
            )
            print()

        if "jaccard_similarity_key_not_picked" in stats:
            print(
                f"  Key NOT Picked (n={stats['jaccard_similarity_key_not_picked']['n']}):"
            )
            print(
                f"    Mean: {stats['jaccard_similarity_key_not_picked']['mean']:.4f} +/- {stats['jaccard_similarity_key_not_picked']['std']:.4f}"
            )
            print(
                f"    Median: {stats['jaccard_similarity_key_not_picked']['median']:.4f}"
            )
            print(
                f"    Range: [{stats['jaccard_similarity_key_not_picked']['min']:.4f}, {stats['jaccard_similarity_key_not_picked']['max']:.4f}]"
            )
            print()


def convert_to_serializable(obj):
    """
    Recursively convert numpy types and other non-serializable types to Python native types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def save_paired_statistics(paired_results: dict, stats: dict, output_path: str) -> None:
    """
    Save paired analysis results and statistics to a JSON file.

    Args:
        paired_results: Dictionary from analyze_paired_trajectories
        stats: Dictionary from compute_paired_statistics
        output_path: Path to save the JSON file
    """
    output_data = {
        "statistics": convert_to_serializable(stats),
        "pairs": convert_to_serializable(paired_results["pairs"]),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Paired statistics saved to: {output_path}")


if __name__ == "__main__":
    # Analyze paired trajectories in the directory
    print("=" * 100)
    print("TRAJECTORY ANALYSIS (2PATH ENVIRONMENT: KEY vs NO-KEY)")
    print("=" * 100)
    print(f"Analyzing directory: {DIR_PATH}")
    print()

    paired_results = analyze_paired_trajectories(DIR_PATH, size=11)

    if not paired_results:
        print("No trajectory pairs found to analyze.")
        exit(0)

    # Compute statistics
    stats = compute_paired_statistics(paired_results)

    # Print comparison
    print_paired_comparison(paired_results, stats)

    # Save statistics to file
    stats_output_path = Path(DIR_PATH) / "paired_analysis_statistics.json"
    save_paired_statistics(paired_results, stats, str(stats_output_path))
