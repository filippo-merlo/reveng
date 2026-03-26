"""Analyze decoded grid accuracy against ground truth and agent behavior.

This module analyzes trajectories with probe-decoded grid representations,
computing metrics to understand how well the decoded environment explains
the agent's behavior.

Key metrics:
- Optimal vs Decoded-Optimal: Do optimal actions match between ground truth and decoded grid?
- Taken vs Decoded-Optimal: Was the taken action optimal according to the decoded grid?
- Error Recovery: When taken action is NOT optimal (ground truth), is decoded-optimal correct?
"""

import gc
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.utils import compute_optimal_actions_from_position

# =============================================================================
# Constants
# =============================================================================

# Mapping for special tokens to prompt_suffix_tokens indices
PROMPT_SUFFIX_TOKEN_MAPPING = {"<|end|>": 0, "<|start|>": 1, "assistant": 2}
DEFAULT_PROMPT_SUFFIX_TOKEN = "<|end|>"

# Action mappings
ACTION_NAME_TO_ENUM = {
    "LEFT": Simple2DNavigationEnv.Actions.LEFT,
    "RIGHT": Simple2DNavigationEnv.Actions.RIGHT,
    "UP": Simple2DNavigationEnv.Actions.UP,
    "DOWN": Simple2DNavigationEnv.Actions.DOWN,
}

ACTION_ENUM_TO_NAME = {v: k for k, v in ACTION_NAME_TO_ENUM.items()}

# Paper-quality plot settings
PAPER_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

MODEL_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
]


def setup_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(PAPER_RC)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StepAnalysis:
    """Analysis results for a single step."""

    step_id: int
    trajectory_id: str
    grid_size: int
    complexity: float

    # Actions
    taken_action: str
    optimal_actions_gt: set[str]  # Ground truth optimal actions
    decoded_optimal_actions: set[str]  # Optimal actions from decoded grid

    # Accuracy flags
    taken_is_optimal_gt: bool  # Was taken action optimal (ground truth)?
    taken_is_optimal_decoded: bool  # Was taken action optimal (decoded)?
    optimal_gt_matches_decoded: bool  # Do GT and decoded optimal sets overlap?

    # Decoded grid info
    decoded_agent_prob: float
    decoded_goal_prob: float
    decoded_valid: bool  # Was a valid environment decodable?

    # Manhattan distance metrics
    avg_agent_distance: float = float("nan")
    avg_goal_distance: float = float("nan")
    min_agent_distance: float = float("nan")
    min_goal_distance: float = float("nan")
    max_agent_distance: float = float("nan")
    max_goal_distance: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "step_id": self.step_id,
            "trajectory_id": self.trajectory_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "taken_action": self.taken_action,
            "optimal_actions_gt": ",".join(sorted(self.optimal_actions_gt)),
            "decoded_optimal_actions": ",".join(sorted(self.decoded_optimal_actions)),
            "taken_is_optimal_gt": self.taken_is_optimal_gt,
            "taken_is_optimal_decoded": self.taken_is_optimal_decoded,
            "optimal_gt_matches_decoded": self.optimal_gt_matches_decoded,
            "decoded_agent_prob": self.decoded_agent_prob,
            "decoded_goal_prob": self.decoded_goal_prob,
            "decoded_valid": self.decoded_valid,
            "avg_agent_distance": self.avg_agent_distance,
            "avg_goal_distance": self.avg_goal_distance,
            "min_agent_distance": self.min_agent_distance,
            "min_goal_distance": self.min_goal_distance,
            "max_agent_distance": self.max_agent_distance,
            "max_goal_distance": self.max_goal_distance,
        }


@dataclass
class TrajectoryAnalysis:
    """Analysis results for a single trajectory."""

    trajectory_id: str
    grid_size: int
    complexity: float
    num_steps: int
    n_valid_decoded: int  # Number of steps with valid decoded grids

    # Aggregated metrics
    taken_optimal_gt_rate: float  # Rate of taken actions being optimal (GT)
    taken_optimal_decoded_rate: float  # Rate of taken actions being optimal (decoded)
    gt_decoded_agreement_rate: float  # Rate of GT and decoded optimal agreement

    # Error recovery metric: when taken is NOT optimal (GT), is decoded-optimal correct?
    error_steps: int  # Steps where taken != optimal (GT)
    decoded_correct_on_errors: (
        int  # Of error steps, how many had decoded-optimal in GT-optimal
    )

    step_analyses: list[StepAnalysis] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        error_recovery_rate = (
            self.decoded_correct_on_errors / self.error_steps
            if self.error_steps > 0
            else float("nan")
        )
        return {
            "trajectory_id": self.trajectory_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "num_steps": self.num_steps,
            "n_valid_decoded": self.n_valid_decoded,
            "taken_optimal_gt_rate": self.taken_optimal_gt_rate,
            "taken_optimal_decoded_rate": self.taken_optimal_decoded_rate,
            "gt_decoded_agreement_rate": self.gt_decoded_agreement_rate,
            "error_steps": self.error_steps,
            "decoded_correct_on_errors": self.decoded_correct_on_errors,
            "error_recovery_rate": error_recovery_rate,
        }


@dataclass
class AnalysisResults:
    """Complete analysis results."""

    trajectory_df: pd.DataFrame
    step_df: pd.DataFrame
    summary_by_size_complexity: pd.DataFrame
    overall_summary: dict[str, Any]


# =============================================================================
# File Discovery
# =============================================================================


def parse_trajectory_filename(filename: str) -> Optional[dict[str, Any]]:
    """Parse trajectory filename to extract metadata.

    Expected format: {model}_size{N}_comp{X.X}_{instance}.json
    """
    # Skip eval results files
    if filename.startswith("_"):
        return None

    # Pattern: {model}_size{size}_comp{comp}_{instance}.json
    pattern = r"(.+)_size(\d+)_comp([\d.]+)_(\d+)\.json"
    match = re.match(pattern, filename)

    if not match:
        return None

    model, size, comp, instance = match.groups()
    return {
        "model": model,
        "grid_size": int(size),
        "complexity": float(comp),
        "instance_id": int(instance),
    }


def discover_trajectory_files(base_dir: Path) -> dict[str, list[Path]]:
    """Discover trajectory files grouped by size.

    Handles nested structure: base_dir/size{N}/*.json

    Returns:
        Dictionary mapping size key to list of trajectory file paths
    """
    size_trajectories: dict[str, list[Path]] = defaultdict(list)

    # Check for size subdirectories
    for size_dir in sorted(base_dir.iterdir()):
        if not size_dir.is_dir():
            continue
        if not size_dir.name.startswith("size"):
            continue

        # Find all JSON files in this size directory
        for filepath in sorted(size_dir.glob("*.json")):
            parsed = parse_trajectory_filename(filepath.name)
            if parsed:
                size_key = f"size{parsed['grid_size']}"
                size_trajectories[size_key].append(filepath)

    return dict(size_trajectories)


def batch_file_list(files: list[Path], batch_size: int) -> Iterator[list[Path]]:
    """Yield batches of files."""
    for i in range(0, len(files), batch_size):
        yield files[i : i + batch_size]


# =============================================================================
# Grid State Parsing
# =============================================================================


def get_true_positions(
    grid_state: list[str],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Extract true agent and goal positions from grid_state.

    Args:
        grid_state: List of strings representing the grid, where the first row
                   contains column indices and the first column contains row indices.

    Returns:
        Tuple of (agent_position, goal_position) where each position is (row, col)

    Raises:
        ValueError: If agent or goal cannot be found in grid_state
    """
    if not grid_state or len(grid_state) < 2:
        raise ValueError("grid_state must have at least 2 rows (header + data)")

    agent_pos = None
    goal_pos = None

    # Skip the first row (column indices) and parse the data rows
    for row_idx, row_str in enumerate(grid_state[1:]):  # Skip header row
        parts = row_str.strip().split()
        if len(parts) < 2:
            continue
        # First part is the row index, rest are cell values
        row_cells = parts[1:]

        for col_idx, cell in enumerate(row_cells):
            if cell == "A":
                agent_pos = (row_idx, col_idx)
            elif cell == "G":
                goal_pos = (row_idx, col_idx)

    if agent_pos is None:
        raise ValueError("Agent position 'A' not found in grid_state")
    if goal_pos is None:
        raise ValueError("Goal position 'G' not found in grid_state")

    return agent_pos, goal_pos


def gridstate2env(grid_state: list[str]) -> Simple2DNavigationEnv:
    """Convert a grid_state representation to a Simple2DNavigationEnv.

    Args:
        grid_state: List of strings representing the grid, where the first row
                   contains column indices and the first column contains row indices.
    """
    if not grid_state or len(grid_state) < 2:
        raise ValueError("grid_state must have at least 2 rows (header + data)")

    # Skip the first row (column indices) and parse the data rows
    grid_list = []

    for row_str in grid_state[1:]:  # Skip header row
        parts = row_str.strip().split()
        if len(parts) < 2:
            continue
        # First part is the row index, rest are cell values
        row_cells = parts[1:]
        grid_list.append(row_cells)

    if not grid_list:
        raise ValueError("No valid grid data found in grid_state")

    grid_height = len(grid_list)
    grid_width = max(len(row) for row in grid_list) if grid_list else 0
    size = max(grid_height, grid_width)

    env = Simple2DNavigationEnv(size=size)
    env.set_env_from_list(grid_list)

    return env


# =============================================================================
# Probe to Environment Conversion
# =============================================================================


def get_classified_cells(
    probes: dict,
    layer_key: str = "model.layers.15.output",
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Extract all cells classified as agent or goal with their probabilities.

    Args:
        probes: Dictionary of probe data with keys like
                "cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{row}_c{col}"
        layer_key: The layer key to extract predictions from

    Returns:
        Tuple of (agent_positions, goal_positions) where each is a list of
        (row, col, probability) tuples sorted by probability in descending order
    """
    # Parse probe keys to extract grid positions and predictions
    grid_data = {}

    for probe_key, probe_value in probes.items():
        if "_r" in probe_key and "_c" in probe_key:
            parts = probe_key.split("_")
            row_idx = None
            col_idx = None

            for part in parts:
                if part.startswith("r") and part[1:].lstrip("-").isdigit():
                    row_idx = int(part[1:])
                elif part.startswith("c") and part[1:].lstrip("-").isdigit():
                    col_idx = int(part[1:])

            if row_idx is not None and col_idx is not None:
                predictions = probe_value.get(layer_key, {})
                grid_data[(row_idx, col_idx)] = predictions

    # Find cells classified as agent or goal
    agent_positions = []
    goal_positions = []

    for (row, col), predictions in grid_data.items():
        if predictions:
            max_class = max(predictions.items(), key=lambda x: x[1])
            class_name, class_prob = max_class

            if class_name == "agent":
                agent_positions.append((row, col, class_prob))
            elif class_name == "goal":
                goal_positions.append((row, col, class_prob))

    # Sort by probability in descending order
    agent_positions.sort(key=lambda x: x[2], reverse=True)
    goal_positions.sort(key=lambda x: x[2], reverse=True)

    return agent_positions, goal_positions


def calculate_manhattan_distance_metrics(
    agent_cells: list[tuple[int, int, float]],
    goal_cells: list[tuple[int, int, float]],
    true_agent_pos: tuple[int, int],
    true_goal_pos: tuple[int, int],
) -> dict[str, float]:
    """Calculate average Manhattan distance metrics for classified cells.

    Args:
        agent_cells: List of (row, col, probability) for cells classified as agent
        goal_cells: List of (row, col, probability) for cells classified as goal
        true_agent_pos: Ground truth agent position (row, col)
        true_goal_pos: Ground truth goal position (row, col)

    Returns:
        Dictionary containing:
            - avg_agent_distance: Average Manhattan distance of agent cells to true agent
            - avg_goal_distance: Average Manhattan distance of goal cells to true goal
            - min_agent_distance: Minimum Manhattan distance of agent cells to true agent
            - min_goal_distance: Minimum Manhattan distance of goal cells to true goal
            - max_agent_distance: Maximum Manhattan distance of agent cells to true agent
            - max_goal_distance: Maximum Manhattan distance of goal cells to true goal
    """
    metrics = {
        "avg_agent_distance": float("nan"),
        "avg_goal_distance": float("nan"),
        "min_agent_distance": float("nan"),
        "min_goal_distance": float("nan"),
        "max_agent_distance": float("nan"),
        "max_goal_distance": float("nan"),
    }

    # Calculate agent metrics
    if agent_cells:
        agent_distances = [
            abs(row - true_agent_pos[0]) + abs(col - true_agent_pos[1])
            for row, col, _ in agent_cells
        ]
        metrics["avg_agent_distance"] = sum(agent_distances) / len(agent_distances)
        metrics["min_agent_distance"] = float(min(agent_distances))
        metrics["max_agent_distance"] = float(max(agent_distances))

    # Calculate goal metrics
    if goal_cells:
        goal_distances = [
            abs(row - true_goal_pos[0]) + abs(col - true_goal_pos[1])
            for row, col, _ in goal_cells
        ]
        metrics["avg_goal_distance"] = sum(goal_distances) / len(goal_distances)
        metrics["min_goal_distance"] = float(min(goal_distances))
        metrics["max_goal_distance"] = float(max(goal_distances))

    return metrics


def probe2env(
    probes: dict,
    layer_key: str = "model.layers.15.output",
    k: int = 1,
) -> Optional[Simple2DNavigationEnv]:
    """Convert probe data to the most likely Simple2DNavigationEnv.

    Uses cells classified as agent and goal with highest probability.
    Returns only the top-k=1 most likely environment for efficiency.

    Args:
        probes: Dictionary of probe data with keys like
                "cognitive_map_probe_l15_s0_suffix_-3--1_mlp_1024_full_upsample_normalize_r{row}_c{col}"
        layer_key: The layer key to extract predictions from
        k: Number of top combinations to consider (default: 1 for most likely)

    Returns:
        Simple2DNavigationEnv or None if no valid environment can be constructed
    """
    # Parse probe keys to extract grid positions and predictions
    grid_data = {}
    max_row = -1
    max_col = -1

    for probe_key, probe_value in probes.items():
        if "_r" in probe_key and "_c" in probe_key:
            parts = probe_key.split("_")
            row_idx = None
            col_idx = None

            for part in parts:
                if part.startswith("r") and part[1:].lstrip("-").isdigit():
                    row_idx = int(part[1:])
                elif part.startswith("c") and part[1:].lstrip("-").isdigit():
                    col_idx = int(part[1:])

            if row_idx is not None and col_idx is not None:
                max_row = max(max_row, row_idx)
                max_col = max(max_col, col_idx)
                predictions = probe_value.get(layer_key, {})
                grid_data[(row_idx, col_idx)] = predictions

    if not grid_data:
        raise ValueError("No valid probe data found")

    height = max_row + 1
    width = max_col + 1

    # Find cells classified as agent or goal
    agent_positions = []
    goal_positions = []

    for (row, col), predictions in grid_data.items():
        if predictions:
            max_class = max(predictions.items(), key=lambda x: x[1])
            class_name, class_prob = max_class

            if class_name == "agent":
                agent_positions.append((row, col, class_prob))
            elif class_name == "goal":
                goal_positions.append((row, col, class_prob))

    # Sort by probability and take top
    agent_positions.sort(key=lambda x: x[2], reverse=True)
    goal_positions.sort(key=lambda x: x[2], reverse=True)

    if not agent_positions or not goal_positions:
        return None

    # Take the most likely agent and goal
    agent_row, agent_col, agent_prob = agent_positions[0]
    goal_row, goal_col, goal_prob = goal_positions[0]

    # Skip if agent and goal are at the same position
    if agent_row == goal_row and agent_col == goal_col:
        if len(goal_positions) > 1:
            goal_row, goal_col, goal_prob = goal_positions[1]
        else:
            return None

    # Build grid
    grid_list = []
    for row in range(height):
        grid_row = []
        for col in range(width):
            predictions = grid_data.get((row, col), {})

            if row == agent_row and col == agent_col:
                grid_row.append("A")
            elif row == goal_row and col == goal_col:
                grid_row.append("G")
            else:
                cell_mapping = {
                    "wall": "#",
                    "empty": "_",
                    "padding": "#",
                    "agent": "_",
                    "goal": "_",
                }
                if predictions:
                    max_class = max(predictions.items(), key=lambda x: x[1])
                    class_name, _ = max_class
                    cell_symbol = cell_mapping.get(class_name, "_")
                else:
                    cell_symbol = "_"
                grid_row.append(cell_symbol)
        grid_list.append(grid_row)

    # Create environment
    env = Simple2DNavigationEnv(size=max(width, height))
    env.set_env_from_list(grid_list)
    env.agent_prob = agent_prob
    env.goal_prob = goal_prob

    return env


# =============================================================================
# Lightweight Data Loading
# =============================================================================


def load_step_data(
    filepath: Path,
    token_name: str = DEFAULT_PROMPT_SUFFIX_TOKEN,
) -> Iterator[dict[str, Any]]:
    """Load step data from a trajectory file, yielding one step at a time.

    This is memory-efficient as it doesn't keep all steps in memory.

    Yields:
        Dictionary with keys: grid_state, probes, agent_action, step_id
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        token_index = PROMPT_SUFFIX_TOKEN_MAPPING.get(token_name, 0)
        steps = data.get("steps", [])

        for step_id, step in enumerate(steps):
            prompt_suffix_tokens = step.get("prompt_suffix_tokens", [])
            if not prompt_suffix_tokens or len(prompt_suffix_tokens) <= token_index:
                continue

            probes = prompt_suffix_tokens[token_index].get("probes", {})
            if not probes:
                continue

            yield {
                "step_id": step_id,
                "grid_state": step.get("grid_state"),
                "probes": probes,
                "agent_action": step.get("agent_action"),
            }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Error loading {filepath.name}: {e}")


def load_trajectory_metadata(filepath: Path) -> Optional[dict[str, Any]]:
    """Load only metadata from a trajectory file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        gp = data.get("grid_params", {})
        return {
            "grid_size": gp.get("grid_width", 0),
            "complexity": gp.get("grid_complexity", 0.0),
            "num_steps": len(data.get("steps", [])),
        }
    except (json.JSONDecodeError, KeyError):
        return None


# =============================================================================
# Optimal Action Computation
# =============================================================================


def get_optimal_actions_gt(env: Simple2DNavigationEnv) -> set[str]:
    """Get optimal actions from ground truth environment."""
    agent_pos = tuple(env.agent_pos)
    try:
        optimal_actions = compute_optimal_actions_from_position(env, agent_pos)
        return {
            ACTION_ENUM_TO_NAME[a] for a in optimal_actions if a in ACTION_ENUM_TO_NAME
        }
    except Exception:
        return set()


def get_optimal_actions_decoded(env: Optional[Simple2DNavigationEnv]) -> set[str]:
    """Get optimal actions from decoded environment."""
    if env is None:
        return set()
    agent_pos = tuple(env.agent_pos)
    try:
        optimal_actions = compute_optimal_actions_from_position(env, agent_pos)
        return {
            ACTION_ENUM_TO_NAME[a] for a in optimal_actions if a in ACTION_ENUM_TO_NAME
        }
    except Exception:
        return set()


# =============================================================================
# Step Analysis
# =============================================================================


def analyze_step(
    step_data: dict[str, Any],
    trajectory_id: str,
    grid_size: int,
    complexity: float,
    layer_key: str = "model.layers.15.output",
) -> Optional[StepAnalysis]:
    """Analyze a single step."""
    grid_state = step_data.get("grid_state")
    probes = step_data.get("probes")
    taken_action = step_data.get("agent_action", "").upper()

    if not grid_state or not probes or not taken_action:
        return None

    try:
        # Create ground truth environment
        gt_env = gridstate2env(grid_state)
        optimal_gt = get_optimal_actions_gt(gt_env)

        # Create decoded environment
        decoded_env = probe2env(probes, layer_key=layer_key)
        optimal_decoded = get_optimal_actions_decoded(decoded_env)

        # Compute metrics
        taken_is_optimal_gt = taken_action in optimal_gt
        taken_is_optimal_decoded = (
            taken_action in optimal_decoded if optimal_decoded else False
        )
        optimal_gt_matches_decoded = (
            bool(optimal_gt & optimal_decoded) if optimal_decoded else False
        )

        decoded_agent_prob = (
            getattr(decoded_env, "agent_prob", 0.0) if decoded_env else 0.0
        )
        decoded_goal_prob = (
            getattr(decoded_env, "goal_prob", 0.0) if decoded_env else 0.0
        )
        decoded_valid = decoded_env is not None and bool(optimal_decoded)

        # Compute Manhattan distance metrics
        manhattan_metrics = {
            "avg_agent_distance": float("nan"),
            "avg_goal_distance": float("nan"),
            "min_agent_distance": float("nan"),
            "min_goal_distance": float("nan"),
            "max_agent_distance": float("nan"),
            "max_goal_distance": float("nan"),
        }

        try:
            # Get true positions
            true_agent_pos, true_goal_pos = get_true_positions(grid_state)
            # Get classified cells
            agent_cells, goal_cells = get_classified_cells(probes, layer_key=layer_key)
            # Compute Manhattan distance metrics
            manhattan_metrics = calculate_manhattan_distance_metrics(
                agent_cells, goal_cells, true_agent_pos, true_goal_pos
            )
        except Exception:
            # If Manhattan distance computation fails, keep NaN values
            pass

        return StepAnalysis(
            step_id=step_data["step_id"],
            trajectory_id=trajectory_id,
            grid_size=grid_size,
            complexity=complexity,
            taken_action=taken_action,
            optimal_actions_gt=optimal_gt,
            decoded_optimal_actions=optimal_decoded,
            taken_is_optimal_gt=taken_is_optimal_gt,
            taken_is_optimal_decoded=taken_is_optimal_decoded,
            optimal_gt_matches_decoded=optimal_gt_matches_decoded,
            decoded_agent_prob=decoded_agent_prob,
            decoded_goal_prob=decoded_goal_prob,
            decoded_valid=decoded_valid,
            avg_agent_distance=manhattan_metrics["avg_agent_distance"],
            avg_goal_distance=manhattan_metrics["avg_goal_distance"],
            min_agent_distance=manhattan_metrics["min_agent_distance"],
            min_goal_distance=manhattan_metrics["min_goal_distance"],
            max_agent_distance=manhattan_metrics["max_agent_distance"],
            max_goal_distance=manhattan_metrics["max_goal_distance"],
        )

    except Exception as e:
        print(f"Warning: Error analyzing step {step_data.get('step_id')}: {e}")
        return None


# =============================================================================
# Trajectory Analysis
# =============================================================================


def analyze_trajectory(
    filepath: Path,
    layer_key: str = "model.layers.15.output",
) -> Optional[TrajectoryAnalysis]:
    """Analyze a complete trajectory."""
    parsed = parse_trajectory_filename(filepath.name)
    if not parsed:
        return None

    metadata = load_trajectory_metadata(filepath)
    if not metadata:
        return None

    grid_size = metadata["grid_size"]
    complexity = metadata["complexity"]

    # Compute trajectory_id early so it can be passed to step analysis
    trajectory_id = (
        f"{parsed['model']}_size{grid_size}_comp{complexity}_{parsed['instance_id']}"
    )

    step_analyses: list[StepAnalysis] = []

    for step_data in load_step_data(filepath):
        analysis = analyze_step(
            step_data, trajectory_id, grid_size, complexity, layer_key
        )
        if analysis:
            step_analyses.append(analysis)

    if not step_analyses:
        return None

    # Aggregate metrics
    n_steps = len(step_analyses)
    n_valid = sum(1 for s in step_analyses if s.decoded_valid)

    taken_optimal_gt_rate = (
        sum(1 for s in step_analyses if s.taken_is_optimal_gt) / n_steps
    )
    taken_optimal_decoded_rate = (
        sum(1 for s in step_analyses if s.decoded_valid and s.taken_is_optimal_decoded)
        / n_valid
        if n_valid > 0
        else 0.0
    )
    gt_decoded_agreement_rate = (
        sum(
            1 for s in step_analyses if s.decoded_valid and s.optimal_gt_matches_decoded
        )
        / n_valid
        if n_valid > 0
        else 0.0
    )

    # Error recovery analysis: when taken is NOT optimal (GT), is decoded-optimal correct?
    error_steps = [
        s for s in step_analyses if not s.taken_is_optimal_gt and s.decoded_valid
    ]
    decoded_correct_on_errors = sum(
        1 for s in error_steps if s.decoded_optimal_actions & s.optimal_actions_gt
    )

    return TrajectoryAnalysis(
        trajectory_id=trajectory_id,
        grid_size=grid_size,
        complexity=complexity,
        num_steps=n_steps,
        n_valid_decoded=n_valid,
        taken_optimal_gt_rate=taken_optimal_gt_rate,
        taken_optimal_decoded_rate=taken_optimal_decoded_rate,
        gt_decoded_agreement_rate=gt_decoded_agreement_rate,
        error_steps=len(error_steps),
        decoded_correct_on_errors=decoded_correct_on_errors,
        step_analyses=step_analyses,
    )


# =============================================================================
# Main Processing Pipeline
# =============================================================================


def process_trajectories(
    base_dir: Path,
    layer_key: str = "model.layers.15.output",
    batch_size: int = 20,
) -> AnalysisResults:
    """Process all trajectories in the directory structure.

    Args:
        base_dir: Base directory containing size subdirectories
        layer_key: Layer key for probe predictions
        batch_size: Number of files to process per batch

    Returns:
        AnalysisResults with all computed metrics
    """
    print(f"\nProcessing trajectories from: {base_dir}")

    # Discover files
    size_trajectories = discover_trajectory_files(base_dir)
    total_files = sum(len(files) for files in size_trajectories.values())
    print(
        f"Found {total_files} trajectory files across {len(size_trajectories)} size groups"
    )

    if not size_trajectories:
        raise ValueError(f"No trajectory files found in {base_dir}")

    all_trajectory_results: list[TrajectoryAnalysis] = []
    all_step_results: list[StepAnalysis] = []

    for size_key, files in sorted(size_trajectories.items()):
        print(f"\n  Processing {size_key}: {len(files)} files")

        total_batches = (len(files) + batch_size - 1) // batch_size

        for batch_idx, batch_files in enumerate(batch_file_list(files, batch_size)):
            for filepath in tqdm(
                batch_files,
                desc=f"{size_key} batch {batch_idx + 1}/{total_batches}",
                leave=False,
            ):
                result = analyze_trajectory(filepath, layer_key)
                if result:
                    all_trajectory_results.append(result)
                    all_step_results.extend(result.step_analyses)

            gc.collect()

    # Build DataFrames
    trajectory_df = pd.DataFrame([t.to_dict() for t in all_trajectory_results])
    step_df = pd.DataFrame([s.to_dict() for s in all_step_results])

    # Compute summaries
    summary_df = compute_summary_by_size_complexity(trajectory_df, step_df)
    overall = compute_overall_summary(trajectory_df, step_df)

    return AnalysisResults(
        trajectory_df=trajectory_df,
        step_df=step_df,
        summary_by_size_complexity=summary_df,
        overall_summary=overall,
    )


def compute_summary_by_size_complexity(
    trajectory_df: pd.DataFrame, step_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute summary statistics grouped by grid_size and complexity."""
    if trajectory_df.empty:
        return pd.DataFrame()

    summary = (
        trajectory_df.groupby(["grid_size", "complexity"])
        .agg(
            n_trajectories=("trajectory_id", "count"),
            mean_taken_optimal_gt=("taken_optimal_gt_rate", "mean"),
            se_taken_optimal_gt=("taken_optimal_gt_rate", "sem"),
            mean_taken_optimal_decoded=("taken_optimal_decoded_rate", "mean"),
            se_taken_optimal_decoded=("taken_optimal_decoded_rate", "sem"),
            mean_gt_decoded_agreement=("gt_decoded_agreement_rate", "mean"),
            se_gt_decoded_agreement=("gt_decoded_agreement_rate", "sem"),
            total_error_steps=("error_steps", "sum"),
            total_decoded_correct=("decoded_correct_on_errors", "sum"),
        )
        .reset_index()
    )

    # Compute error recovery rate
    summary["error_recovery_rate"] = (
        summary["total_decoded_correct"] / summary["total_error_steps"]
    ).fillna(0)

    # Add Manhattan distance metrics by grouping step_df
    if not step_df.empty:
        manhattan_summary = (
            step_df.groupby(["grid_size", "complexity"])
            .agg(
                avg_agent_distance=("avg_agent_distance", "mean"),
                avg_goal_distance=("avg_goal_distance", "mean"),
                min_agent_distance=("min_agent_distance", "mean"),
                min_goal_distance=("min_goal_distance", "mean"),
                max_agent_distance=("max_agent_distance", "mean"),
                max_goal_distance=("max_goal_distance", "mean"),
            )
            .reset_index()
        )
        # Merge with summary
        summary = summary.merge(
            manhattan_summary, on=["grid_size", "complexity"], how="left"
        )

    return summary


def compute_overall_summary(
    trajectory_df: pd.DataFrame,
    step_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute overall summary statistics."""
    if trajectory_df.empty:
        return {}

    total_error_steps = trajectory_df["error_steps"].sum()
    total_decoded_correct = trajectory_df["decoded_correct_on_errors"].sum()

    # Compute Manhattan distance metrics (averaging over all valid steps)
    manhattan_metrics = {
        "avg_agent_distance": float(step_df["avg_agent_distance"].mean()),
        "avg_goal_distance": float(step_df["avg_goal_distance"].mean()),
        "min_agent_distance": float(step_df["min_agent_distance"].mean()),
        "min_goal_distance": float(step_df["min_goal_distance"].mean()),
        "max_agent_distance": float(step_df["max_agent_distance"].mean()),
        "max_goal_distance": float(step_df["max_goal_distance"].mean()),
    }

    return {
        "n_trajectories": int(len(trajectory_df)),
        "total_steps": int(len(step_df)),
        "total_valid_decoded_steps": int(step_df["decoded_valid"].sum()),
        "overall_taken_optimal_gt": float(
            trajectory_df["taken_optimal_gt_rate"].mean()
        ),
        "overall_taken_optimal_decoded": float(
            trajectory_df["taken_optimal_decoded_rate"].mean()
        ),
        "overall_gt_decoded_agreement": float(
            trajectory_df["gt_decoded_agreement_rate"].mean()
        ),
        "total_error_steps": int(total_error_steps),
        "total_decoded_correct_on_errors": int(total_decoded_correct),
        "overall_error_recovery_rate": (
            float(total_decoded_correct / total_error_steps)
            if total_error_steps > 0
            else 0.0
        ),
        "manhattan_distance_metrics": manhattan_metrics,
    }


# =============================================================================
# Visualizations
# =============================================================================


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save figure to both PNG and PDF subfolders."""
    png_dir = output_dir / "png"
    pdf_dir = output_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f"{filename}.png"
    pdf_path = pdf_dir / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    return png_path


def plot_accuracy_comparison(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot comparison of different accuracy metrics."""
    setup_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # By grid size (line plot to match complexity plot style)
    size_summary = df.groupby("grid_size").agg(
        {
            "taken_optimal_gt_rate": ["mean", "sem"],
            "taken_optimal_decoded_rate": ["mean", "sem"],
            "gt_decoded_agreement_rate": ["mean", "sem"],
        }
    )

    for metric, label, color in [
        ("taken_optimal_gt_rate", "Taken optimal (GT)", MODEL_COLORS[0]),
        ("taken_optimal_decoded_rate", "Taken optimal (Decoded)", MODEL_COLORS[1]),
        ("gt_decoded_agreement_rate", "GT-Decoded Agreement", MODEL_COLORS[2]),
    ]:
        axes[0].errorbar(
            size_summary.index,
            size_summary[metric]["mean"],
            yerr=size_summary[metric]["sem"],
            marker="o",
            label=label,
            color=color,
            capsize=3,
        )
    axes[0].set_xlabel("Grid Size")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Metrics by Grid Size")
    axes[0].set_xticks(sorted(size_summary.index))
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    # By complexity
    comp_summary = df.groupby("complexity").agg(
        {
            "taken_optimal_gt_rate": ["mean", "sem"],
            "taken_optimal_decoded_rate": ["mean", "sem"],
            "gt_decoded_agreement_rate": ["mean", "sem"],
        }
    )

    for i, (metric, label, color) in enumerate(
        [
            ("taken_optimal_gt_rate", "Taken optimal (GT)", MODEL_COLORS[0]),
            ("taken_optimal_decoded_rate", "Taken optimal (Decoded)", MODEL_COLORS[1]),
            ("gt_decoded_agreement_rate", "GT-Decoded Agreement", MODEL_COLORS[2]),
        ]
    ):
        axes[1].errorbar(
            comp_summary.index,
            comp_summary[metric]["mean"],
            yerr=comp_summary[metric]["sem"],
            marker="o",
            label=label,
            color=color,
            capsize=3,
        )
    axes[1].set_xlabel("Complexity")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Metrics by Complexity")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    # Error recovery by size (with error bars, only for trajectories with errors)
    # Filter to trajectories that have error steps to compute meaningful per-trajectory rates
    df_with_errors = df[df["error_steps"] > 0].copy()
    if len(df_with_errors) > 0:
        error_summary = df_with_errors.groupby("grid_size").agg(
            {"error_recovery_rate": ["mean", "sem"]}
        )
        axes[2].errorbar(
            error_summary.index,
            error_summary["error_recovery_rate"]["mean"],
            yerr=error_summary["error_recovery_rate"]["sem"],
            marker="o",
            color=MODEL_COLORS[3],
            linewidth=2,
            capsize=3,
        )
        axes[2].set_xticks(sorted(error_summary.index))
    axes[2].set_xlabel("Grid Size")
    axes[2].set_ylabel("Error Recovery Rate")
    axes[2].set_title("Error Recovery Rate by Grid Size")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Decoded Grid Accuracy Analysis", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "accuracy_comparison")
    plt.close(fig)

    return output_path


def plot_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot heatmaps of metrics by grid_size x complexity."""
    setup_paper_style()

    metrics = [
        ("mean_taken_optimal_gt", "Taken Optimal (GT)"),
        ("mean_taken_optimal_decoded", "Taken Optimal (Decoded)"),
        ("mean_gt_decoded_agreement", "GT-Decoded Agreement"),
        ("error_recovery_rate", "Error Recovery Rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (metric_col, metric_label) in enumerate(metrics):
        if metric_col not in summary_df.columns:
            continue

        pivot = summary_df.pivot(
            index="complexity", columns="grid_size", values=metric_col
        )

        im = axes[idx].imshow(
            pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
        )
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels(pivot.columns)
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels([f"{c:.1f}" for c in pivot.index])
        axes[idx].set_xlabel("Grid Size")
        axes[idx].set_ylabel("Complexity")
        axes[idx].set_title(metric_label)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    axes[idx].text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=8
                    )

        plt.colorbar(im, ax=axes[idx])

    plt.suptitle("Decoded Grid Metrics Heatmaps", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "metrics_heatmaps")
    plt.close(fig)

    return output_path


def plot_additional_analysis(
    trajectory_df: pd.DataFrame,
    step_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot additional analysis: error recovery by complexity, decoded validity, error breakdown."""
    setup_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Error recovery by complexity (top-left)
    df_with_errors = trajectory_df[trajectory_df["error_steps"] > 0].copy()
    if len(df_with_errors) > 0:
        comp_error = df_with_errors.groupby("complexity").agg(
            {"error_recovery_rate": ["mean", "sem"]}
        )
        axes[0, 0].errorbar(
            comp_error.index,
            comp_error["error_recovery_rate"]["mean"],
            yerr=comp_error["error_recovery_rate"]["sem"],
            marker="o",
            color=MODEL_COLORS[3],
            linewidth=2,
            capsize=3,
        )
    axes[0, 0].set_xlabel("Complexity")
    axes[0, 0].set_ylabel("Error Recovery Rate")
    axes[0, 0].set_title("Error Recovery by Complexity")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Decoded validity rate by size and complexity (top-right)
    validity_by_size = trajectory_df.groupby("grid_size").agg(
        {"n_valid_decoded": "sum", "num_steps": "sum"}
    )
    validity_by_size["validity_rate"] = (
        validity_by_size["n_valid_decoded"] / validity_by_size["num_steps"]
    )

    validity_by_comp = trajectory_df.groupby("complexity").agg(
        {"n_valid_decoded": "sum", "num_steps": "sum"}
    )
    validity_by_comp["validity_rate"] = (
        validity_by_comp["n_valid_decoded"] / validity_by_comp["num_steps"]
    )

    ax2 = axes[0, 1]
    ax2.plot(
        validity_by_size.index,
        validity_by_size["validity_rate"],
        marker="o",
        label="By Grid Size",
        color=MODEL_COLORS[0],
    )
    ax2_twin = ax2.twiny()
    ax2_twin.plot(
        validity_by_comp.index,
        validity_by_comp["validity_rate"],
        marker="s",
        label="By Complexity",
        color=MODEL_COLORS[1],
        linestyle="--",
    )
    ax2.set_xlabel("Grid Size")
    ax2_twin.set_xlabel("Complexity", color=MODEL_COLORS[1])
    ax2.set_ylabel("Decoded Validity Rate")
    ax2.set_title("Decoded Grid Validity Rate")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=7)
    ax2_twin.legend(loc="lower right", fontsize=7)

    # 3. Error breakdown: When agent errs, is decoded grid correct? (bottom-left)
    # Error steps are steps where taken_is_optimal_gt is False
    error_steps_df = step_df[~step_df["taken_is_optimal_gt"]].copy()
    if len(error_steps_df) > 0:
        # Breakdown by whether decoded grid agrees with GT
        error_breakdown = error_steps_df.groupby("grid_size").agg(
            {
                "optimal_gt_matches_decoded": ["sum", "count"],
            }
        )
        error_breakdown["decoded_correct_pct"] = (
            error_breakdown["optimal_gt_matches_decoded"]["sum"]
            / error_breakdown["optimal_gt_matches_decoded"]["count"]
        )
        error_breakdown["decoded_wrong_pct"] = (
            1 - error_breakdown["decoded_correct_pct"]
        )

        x = error_breakdown.index
        width = 0.35
        axes[1, 0].bar(
            x - width / 2,
            error_breakdown["decoded_correct_pct"],
            width,
            label="Decoded = GT Optimal",
            color=MODEL_COLORS[2],
        )
        axes[1, 0].bar(
            x + width / 2,
            error_breakdown["decoded_wrong_pct"],
            width,
            label="Decoded ≠ GT Optimal",
            color=MODEL_COLORS[4] if len(MODEL_COLORS) > 4 else "gray",
        )
        axes[1, 0].set_xticks(x)
    axes[1, 0].set_xlabel("Grid Size")
    axes[1, 0].set_ylabel("Proportion of Errors")
    axes[1, 0].set_title("Error Breakdown: Is Decoded Grid Correct?")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Step position analysis: accuracy over trajectory (bottom-right)
    # Bin steps by relative position in trajectory
    if "step_id" in step_df.columns:
        step_analysis = step_df.copy()
        # Get max step per trajectory to compute relative position
        max_steps = step_analysis.groupby("trajectory_id")["step_id"].transform("max")
        step_analysis["relative_pos"] = step_analysis["step_id"] / (max_steps + 1)
        step_analysis["pos_bin"] = pd.cut(
            step_analysis["relative_pos"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"],
        )

        pos_summary = step_analysis.groupby("pos_bin", observed=True).agg(
            {
                "taken_is_optimal_gt": "mean",
                "optimal_gt_matches_decoded": "mean",
            }
        )

        x_pos = range(len(pos_summary))
        axes[1, 1].plot(
            x_pos,
            pos_summary["taken_is_optimal_gt"],
            marker="o",
            label="Taken Optimal (GT)",
            color=MODEL_COLORS[0],
        )
        axes[1, 1].plot(
            x_pos,
            pos_summary["optimal_gt_matches_decoded"],
            marker="s",
            label="GT-Decoded Agreement",
            color=MODEL_COLORS[2],
        )
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(pos_summary.index)
    axes[1, 1].set_xlabel("Position in Trajectory")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_title("Accuracy by Trajectory Position")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Additional Decoded Grid Analysis", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "additional_analysis")
    plt.close(fig)

    return output_path


# =============================================================================
# Output Saving
# =============================================================================


def save_results(results: AnalysisResults, output_dir: Path) -> dict[str, Path]:
    """Save results to CSV files and generate visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    # Save trajectory-level metrics
    traj_path = output_dir / "trajectory_metrics.csv"
    results.trajectory_df.to_csv(traj_path, index=False)
    output_paths["trajectory_metrics"] = traj_path
    print(f"  Saved: {traj_path}")

    # Save step-level metrics
    step_path = output_dir / "step_metrics.csv"
    results.step_df.to_csv(step_path, index=False)
    output_paths["step_metrics"] = step_path
    print(f"  Saved: {step_path}")

    # Save summary
    summary_path = output_dir / "summary_by_size_complexity.csv"
    results.summary_by_size_complexity.to_csv(summary_path, index=False)
    output_paths["summary"] = summary_path
    print(f"  Saved: {summary_path}")

    # Save overall summary
    overall_path = output_dir / "overall_summary.json"
    with open(overall_path, "w") as f:
        json.dump(results.overall_summary, f, indent=2)
    output_paths["overall"] = overall_path
    print(f"  Saved: {overall_path}")

    # Generate visualizations
    print("  Generating visualizations...")
    if not results.trajectory_df.empty:
        plot_accuracy_comparison(results.trajectory_df, output_dir)
        if not results.step_df.empty:
            plot_additional_analysis(results.trajectory_df, results.step_df, output_dir)
    if not results.summary_by_size_complexity.empty:
        plot_heatmaps(results.summary_by_size_complexity, output_dir)

    return output_paths


def print_summary(results: AnalysisResults) -> None:
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("DECODED GRID ANALYSIS SUMMARY")
    print("=" * 60)

    overall = results.overall_summary
    print(f"\nTotal trajectories analyzed: {overall.get('n_trajectories', 0)}")
    print(f"Total steps: {overall.get('total_steps', 0)}")
    print(f"Valid decoded steps: {overall.get('total_valid_decoded_steps', 0)}")

    print("\n--- Accuracy Metrics ---")
    print(f"Taken optimal (GT): {overall.get('overall_taken_optimal_gt', 0):.4f}")
    print(
        f"Taken optimal (Decoded): {overall.get('overall_taken_optimal_decoded', 0):.4f}"
    )
    print(f"GT-Decoded Agreement: {overall.get('overall_gt_decoded_agreement', 0):.4f}")

    print("\n--- Error Recovery Analysis ---")
    print(
        f"Total error steps (taken != optimal GT): {overall.get('total_error_steps', 0)}"
    )
    print(
        f"Decoded correct on errors: {overall.get('total_decoded_correct_on_errors', 0)}"
    )
    print(f"Error recovery rate: {overall.get('overall_error_recovery_rate', 0):.4f}")

    print("\n--- By Grid Size ---")
    if not results.trajectory_df.empty:
        size_summary = results.trajectory_df.groupby("grid_size").agg(
            {
                "taken_optimal_gt_rate": "mean",
                "taken_optimal_decoded_rate": "mean",
                "gt_decoded_agreement_rate": "mean",
            }
        )
        print(size_summary.to_string())

    print("\n" + "=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Command-line interface entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze decoded grid accuracy against ground truth and agent behavior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/pre_reasoning",
        help="Base directory containing size subdirectories with trajectory files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save analysis outputs",
    )

    parser.add_argument(
        "--layer-key",
        type=str,
        default="model.layers.15.output",
        help="Layer key for probe predictions",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of files to process per batch (limits RAM usage)",
    )

    args = parser.parse_args()

    base_path = Path(args.base_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = process_trajectories(
        base_path,
        layer_key=args.layer_key,
        batch_size=args.batch_size,
    )

    save_results(results, output_path)
    print_summary(results)


if __name__ == "__main__":
    main()
