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
from itertools import product
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from papers.papers_code.reveng.src.reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from papers.papers_code.reveng.src.reveng.environment_generator.utils import compute_optimal_actions_from_position

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
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
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

    # Top-k decoded-candidate analysis fields (defaults preserve top-1 behavior)
    decode_mode: str = "top1"
    topk_k: int = 1
    topk_n_candidates: int = 1
    topk_total_joint_mass: float = float("nan")
    topk_any_valid: bool = False
    topk_any_optimal_gt_match: bool = False
    topk_any_taken_optimal_decoded: bool = False
    topk_best_candidate_rank: int = -1
    topk_best_joint_prob: float = float("nan")
    topk_best_agent_prob: float = float("nan")
    topk_best_goal_prob: float = float("nan")
    topk_weighted_taken_optimal_decoded: float = float("nan")
    topk_weighted_optimal_gt_match: float = float("nan")
    topk_fraction_taken_optimal: float = float("nan")

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
            "decode_mode": self.decode_mode,
            "topk_k": self.topk_k,
            "topk_n_candidates": self.topk_n_candidates,
            "topk_total_joint_mass": self.topk_total_joint_mass,
            "topk_any_valid": self.topk_any_valid,
            "topk_any_optimal_gt_match": self.topk_any_optimal_gt_match,
            "topk_any_taken_optimal_decoded": self.topk_any_taken_optimal_decoded,
            "topk_best_candidate_rank": self.topk_best_candidate_rank,
            "topk_best_joint_prob": self.topk_best_joint_prob,
            "topk_best_agent_prob": self.topk_best_agent_prob,
            "topk_best_goal_prob": self.topk_best_goal_prob,
            "topk_weighted_taken_optimal_decoded": self.topk_weighted_taken_optimal_decoded,
            "topk_weighted_optimal_gt_match": self.topk_weighted_optimal_gt_match,
            "topk_fraction_taken_optimal": self.topk_fraction_taken_optimal,
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

    # Reverse recovery: taken is NOT optimal (decoded), but IS optimal (GT)
    reverse_error_steps: int  # Steps where taken != optimal (decoded) but decoded valid
    reverse_recovered: int  # Of those, how many had taken in GT-optimal

    step_analyses: list[StepAnalysis] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        error_recovery_rate = (
            self.decoded_correct_on_errors / self.error_steps
            if self.error_steps > 0
            else float("nan")
        )
        reverse_recovery_rate = (
            self.reverse_recovered / self.reverse_error_steps
            if self.reverse_error_steps > 0
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
            "reverse_error_steps": self.reverse_error_steps,
            "reverse_recovered": self.reverse_recovered,
            "reverse_recovery_rate": reverse_recovery_rate,
        }


@dataclass
class AnalysisResults:
    """Complete analysis results."""

    trajectory_df: pd.DataFrame
    step_df: pd.DataFrame
    summary_by_size_complexity: pd.DataFrame
    overall_summary: dict[str, Any]


@dataclass
class DecodedEnvCandidate:
    """Single decoded environment candidate from top-k agent/goal positions."""

    env: Simple2DNavigationEnv
    agent_row: int
    agent_col: int
    goal_row: int
    goal_col: int
    agent_prob: float
    goal_prob: float
    agent_rank: int
    goal_rank: int
    joint_prob: float
    rank: int


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


def probe2env_topk_candidates(
    probes: dict,
    layer_key: str = "model.layers.15.output",
    k: int = 3,
) -> list[DecodedEnvCandidate]:
    """Build top-k decoded environment candidates from agent/goal probe probabilities.

    Args:
        probes: Dictionary of probe data with keys containing _r{row}_c{col}
        layer_key: Layer key used for class probabilities
        k: Number of top agent and top goal locations to combine

    Returns:
        Ranked list of decoded candidates (highest joint prob first).
    """
    # Parse probe keys to extract grid positions and predictions
    grid_data: dict[tuple[int, int], dict[str, float]] = {}
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

    # Collect top-k cells classified as agent or goal
    agent_positions: list[tuple[int, int, float]] = []
    goal_positions: list[tuple[int, int, float]] = []

    for (row, col), predictions in grid_data.items():
        if not predictions:
            continue
        class_name, class_prob = max(predictions.items(), key=lambda x: x[1])
        if class_name == "agent":
            agent_positions.append((row, col, class_prob))
        elif class_name == "goal":
            goal_positions.append((row, col, class_prob))

    agent_positions.sort(key=lambda x: x[2], reverse=True)
    goal_positions.sort(key=lambda x: x[2], reverse=True)

    top_agents = agent_positions[: max(1, k)]
    top_goals = goal_positions[: max(1, k)]

    agent_ranks = {(row, col): idx + 1 for idx, (row, col, _) in enumerate(top_agents)}
    goal_ranks = {(row, col): idx + 1 for idx, (row, col, _) in enumerate(top_goals)}

    if not top_agents or not top_goals:
        return []

    # Build valid combinations (agent and goal cannot overlap)
    combinations: list[tuple[int, int, float, int, int, int, float, int, float]] = []
    for (a_row, a_col, a_prob), (g_row, g_col, g_prob) in product(
        top_agents, top_goals
    ):
        if a_row == g_row and a_col == g_col:
            continue
        a_rank = agent_ranks[(a_row, a_col)]
        g_rank = goal_ranks[(g_row, g_col)]
        combinations.append(
            (
                a_row,
                a_col,
                a_prob,
                a_rank,
                g_row,
                g_col,
                g_prob,
                g_rank,
                a_prob * g_prob,
            )
        )

    # If all combos collide, no valid candidate env
    if not combinations:
        return []

    combinations.sort(key=lambda x: x[6], reverse=True)

    candidates: list[DecodedEnvCandidate] = []
    for rank, (
        agent_row,
        agent_col,
        agent_prob,
        agent_rank,
        goal_row,
        goal_col,
        goal_prob,
        goal_rank,
        joint_prob,
    ) in enumerate(combinations, start=1):
        # Build grid for this candidate
        grid_list: list[list[str]] = []
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
                        class_name, _ = max(predictions.items(), key=lambda x: x[1])
                        cell_symbol = cell_mapping.get(class_name, "_")
                    else:
                        cell_symbol = "_"
                    grid_row.append(cell_symbol)
            grid_list.append(grid_row)

        env = Simple2DNavigationEnv(size=max(width, height))
        env.set_env_from_list(grid_list)
        env.agent_prob = agent_prob
        env.goal_prob = goal_prob
        env.joint_prob = joint_prob

        candidates.append(
            DecodedEnvCandidate(
                env=env,
                agent_row=agent_row,
                agent_col=agent_col,
                goal_row=goal_row,
                goal_col=goal_col,
                agent_prob=agent_prob,
                goal_prob=goal_prob,
                agent_rank=agent_rank,
                goal_rank=goal_rank,
                joint_prob=joint_prob,
                rank=rank,
            )
        )

    return candidates


def probe2env(
    probes: dict,
    layer_key: str = "model.layers.15.output",
    k: int = 1,
) -> Optional[Simple2DNavigationEnv]:
    """Convert probe data to the most likely Simple2DNavigationEnv.

    Backward-compatible top-1 decode wrapper around top-k candidates.
    """
    candidates = probe2env_topk_candidates(
        probes=probes, layer_key=layer_key, k=max(1, k)
    )
    if not candidates:
        return None
    return candidates[0].env


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


def analyze_step_topk(
    step_data: dict[str, Any],
    trajectory_id: str,
    grid_size: int,
    complexity: float,
    layer_key: str = "model.layers.15.output",
    k: int = 3,
) -> Optional[StepAnalysis]:
    """Analyze a single step using top-k agent/goal decoded candidates."""
    grid_state = step_data.get("grid_state")
    probes = step_data.get("probes")
    taken_action = step_data.get("agent_action", "").upper()

    if not grid_state or not probes or not taken_action:
        return None

    try:
        gt_env = gridstate2env(grid_state)
        optimal_gt = get_optimal_actions_gt(gt_env)

        manhattan_metrics = {
            "avg_agent_distance": float("nan"),
            "avg_goal_distance": float("nan"),
            "min_agent_distance": float("nan"),
            "min_goal_distance": float("nan"),
            "max_agent_distance": float("nan"),
            "max_goal_distance": float("nan"),
        }
        try:
            true_agent_pos, true_goal_pos = get_true_positions(grid_state)
            agent_cells, goal_cells = get_classified_cells(probes, layer_key=layer_key)
            manhattan_metrics = calculate_manhattan_distance_metrics(
                agent_cells, goal_cells, true_agent_pos, true_goal_pos
            )
        except Exception:
            pass

        candidates = probe2env_topk_candidates(probes, layer_key=layer_key, k=k)
        candidates_sorted = sorted(candidates, key=lambda c: c.joint_prob, reverse=True)

        (
            candidate_decoded_optimal,
            candidate_valid,
            candidate_taken_optimal,
            candidate_match_gt,
            joint_probs,
        ) = _compute_candidate_metrics(candidates_sorted, taken_action, optimal_gt)

        return _build_step_analysis_from_metrics(
            step_id=step_data["step_id"],
            trajectory_id=trajectory_id,
            grid_size=grid_size,
            complexity=complexity,
            taken_action=taken_action,
            optimal_gt=optimal_gt,
            manhattan_metrics=manhattan_metrics,
            candidates_sorted=candidates_sorted,
            candidate_decoded_optimal=candidate_decoded_optimal,
            candidate_valid=candidate_valid,
            candidate_taken_optimal=candidate_taken_optimal,
            candidate_match_gt=candidate_match_gt,
            joint_probs=joint_probs,
            k=k,
        )

    except Exception as e:
        print(f"Warning: Error analyzing top-k step {step_data.get('step_id')}: {e}")
        return None


def analyze_step_topk_from_candidates(
    step_id: int,
    trajectory_id: str,
    grid_size: int,
    complexity: float,
    taken_action: str,
    optimal_gt: set[str],
    manhattan_metrics: dict[str, float],
    candidates: list[DecodedEnvCandidate],
    k: int,
) -> Optional[StepAnalysis]:
    """Analyze a single step using precomputed decoded candidates."""
    candidates_sorted = sorted(candidates, key=lambda c: c.joint_prob, reverse=True)
    (
        candidate_decoded_optimal,
        candidate_valid,
        candidate_taken_optimal,
        candidate_match_gt,
        joint_probs,
    ) = _compute_candidate_metrics(candidates_sorted, taken_action, optimal_gt)

    return _build_step_analysis_from_metrics(
        step_id=step_id,
        trajectory_id=trajectory_id,
        grid_size=grid_size,
        complexity=complexity,
        taken_action=taken_action,
        optimal_gt=optimal_gt,
        manhattan_metrics=manhattan_metrics,
        candidates_sorted=candidates_sorted,
        candidate_decoded_optimal=candidate_decoded_optimal,
        candidate_valid=candidate_valid,
        candidate_taken_optimal=candidate_taken_optimal,
        candidate_match_gt=candidate_match_gt,
        joint_probs=joint_probs,
        k=k,
    )


def _compute_candidate_metrics(
    candidates_sorted: list[DecodedEnvCandidate],
    taken_action: str,
    optimal_gt: set[str],
) -> tuple[list[set[str]], list[bool], list[bool], list[bool], list[float]]:
    candidate_decoded_optimal: list[set[str]] = []
    candidate_valid: list[bool] = []
    candidate_taken_optimal: list[bool] = []
    candidate_match_gt: list[bool] = []
    joint_probs: list[float] = []

    for cand in candidates_sorted:
        opt_dec = get_optimal_actions_decoded(cand.env)
        is_valid = bool(opt_dec)
        is_taken_opt = taken_action in opt_dec if is_valid else False
        matches_gt = bool(optimal_gt & opt_dec) if is_valid else False

        candidate_decoded_optimal.append(opt_dec)
        candidate_valid.append(is_valid)
        candidate_taken_optimal.append(is_taken_opt)
        candidate_match_gt.append(matches_gt)
        joint_probs.append(float(cand.joint_prob))

    return (
        candidate_decoded_optimal,
        candidate_valid,
        candidate_taken_optimal,
        candidate_match_gt,
        joint_probs,
    )


def _build_step_analysis_from_metrics(
    step_id: int,
    trajectory_id: str,
    grid_size: int,
    complexity: float,
    taken_action: str,
    optimal_gt: set[str],
    manhattan_metrics: dict[str, float],
    candidates_sorted: list[DecodedEnvCandidate],
    candidate_decoded_optimal: list[set[str]],
    candidate_valid: list[bool],
    candidate_taken_optimal: list[bool],
    candidate_match_gt: list[bool],
    joint_probs: list[float],
    k: int,
) -> Optional[StepAnalysis]:
    if not taken_action:
        return None

    metrics = {
        "avg_agent_distance": float("nan"),
        "avg_goal_distance": float("nan"),
        "min_agent_distance": float("nan"),
        "min_goal_distance": float("nan"),
        "max_agent_distance": float("nan"),
        "max_goal_distance": float("nan"),
    }
    metrics.update(manhattan_metrics or {})

    if not candidates_sorted:
        return StepAnalysis(
            step_id=step_id,
            trajectory_id=trajectory_id,
            grid_size=grid_size,
            complexity=complexity,
            taken_action=taken_action,
            optimal_actions_gt=optimal_gt,
            decoded_optimal_actions=set(),
            taken_is_optimal_gt=taken_action in optimal_gt,
            taken_is_optimal_decoded=False,
            optimal_gt_matches_decoded=False,
            decoded_agent_prob=0.0,
            decoded_goal_prob=0.0,
            decoded_valid=False,
            avg_agent_distance=metrics["avg_agent_distance"],
            avg_goal_distance=metrics["avg_goal_distance"],
            min_agent_distance=metrics["min_agent_distance"],
            min_goal_distance=metrics["min_goal_distance"],
            max_agent_distance=metrics["max_agent_distance"],
            max_goal_distance=metrics["max_goal_distance"],
            decode_mode="topk",
            topk_k=max(1, k),
            topk_n_candidates=0,
            topk_total_joint_mass=0.0,
            topk_any_valid=False,
            topk_any_optimal_gt_match=False,
            topk_any_taken_optimal_decoded=False,
            topk_best_candidate_rank=-1,
            topk_best_joint_prob=float("nan"),
            topk_best_agent_prob=float("nan"),
            topk_best_goal_prob=float("nan"),
            topk_weighted_taken_optimal_decoded=float("nan"),
            topk_weighted_optimal_gt_match=float("nan"),
            topk_fraction_taken_optimal=float("nan"),
        )

    total_joint_mass = float(sum(joint_probs))
    if total_joint_mass > 0:
        weights = [p / total_joint_mass for p in joint_probs]
        weighted_taken_opt = float(
            sum(
                w * (1.0 if flag else 0.0)
                for w, flag in zip(weights, candidate_taken_optimal)
            )
        )
        weighted_match_gt = float(
            sum(
                w * (1.0 if flag else 0.0)
                for w, flag in zip(weights, candidate_match_gt)
            )
        )
    else:
        weighted_taken_opt = float("nan")
        weighted_match_gt = float("nan")

    fraction_taken_optimal = (
        float(sum(1 for flag in candidate_taken_optimal if flag))
        / len(candidates_sorted)
        if candidates_sorted
        else float("nan")
    )

    best = candidates_sorted[0]
    best_optimal_decoded = candidate_decoded_optimal[0]
    best_decoded_valid = candidate_valid[0]
    best_taken_is_optimal_decoded = candidate_taken_optimal[0]
    best_optimal_gt_matches_decoded = candidate_match_gt[0]

    return StepAnalysis(
        step_id=step_id,
        trajectory_id=trajectory_id,
        grid_size=grid_size,
        complexity=complexity,
        taken_action=taken_action,
        optimal_actions_gt=optimal_gt,
        decoded_optimal_actions=best_optimal_decoded,
        taken_is_optimal_gt=taken_action in optimal_gt,
        taken_is_optimal_decoded=best_taken_is_optimal_decoded,
        optimal_gt_matches_decoded=best_optimal_gt_matches_decoded,
        decoded_agent_prob=float(best.agent_prob),
        decoded_goal_prob=float(best.goal_prob),
        decoded_valid=best_decoded_valid,
        avg_agent_distance=metrics["avg_agent_distance"],
        avg_goal_distance=metrics["avg_goal_distance"],
        min_agent_distance=metrics["min_agent_distance"],
        min_goal_distance=metrics["min_goal_distance"],
        max_agent_distance=metrics["max_agent_distance"],
        max_goal_distance=metrics["max_goal_distance"],
        decode_mode="topk",
        topk_k=max(1, k),
        topk_n_candidates=len(candidates_sorted),
        topk_total_joint_mass=total_joint_mass,
        topk_any_valid=any(candidate_valid),
        topk_any_optimal_gt_match=any(candidate_match_gt),
        topk_any_taken_optimal_decoded=any(candidate_taken_optimal),
        topk_best_candidate_rank=1,
        topk_best_joint_prob=float(best.joint_prob),
        topk_best_agent_prob=float(best.agent_prob),
        topk_best_goal_prob=float(best.goal_prob),
        topk_weighted_taken_optimal_decoded=weighted_taken_opt,
        topk_weighted_optimal_gt_match=weighted_match_gt,
        topk_fraction_taken_optimal=fraction_taken_optimal,
    )


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

    # Reverse recovery: taken is NOT optimal (decoded), but IS optimal (GT)
    reverse_error_steps = [
        s for s in step_analyses if not s.taken_is_optimal_decoded and s.decoded_valid
    ]
    reverse_recovered = sum(1 for s in reverse_error_steps if s.taken_is_optimal_gt)

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
        reverse_error_steps=len(reverse_error_steps),
        reverse_recovered=reverse_recovered,
        step_analyses=step_analyses,
    )


def analyze_trajectory_topk(
    filepath: Path,
    layer_key: str = "model.layers.15.output",
    k: int = 3,
    aggregation: str = "weighted",
) -> Optional[TrajectoryAnalysis]:
    """Analyze a complete trajectory using top-k decoded env candidates per step.

    Args:
        filepath: trajectory json file
        layer_key: probe layer key
        k: top-k agent and goal candidates used to form combinations
        aggregation: how trajectory-level decoded metrics are aggregated:
            - "top1": use best candidate metrics (same fields as legacy)
            - "any": step is success if any candidate satisfies the criterion
            - "weighted": probability-weighted step metric using candidate joint probs
    """
    parsed = parse_trajectory_filename(filepath.name)
    if not parsed:
        return None

    metadata = load_trajectory_metadata(filepath)
    if not metadata:
        return None

    grid_size = metadata["grid_size"]
    complexity = metadata["complexity"]
    trajectory_id = (
        f"{parsed['model']}_size{grid_size}_comp{complexity}_{parsed['instance_id']}"
    )

    step_analyses: list[StepAnalysis] = []
    for step_data in load_step_data(filepath):
        analysis = analyze_step_topk(
            step_data=step_data,
            trajectory_id=trajectory_id,
            grid_size=grid_size,
            complexity=complexity,
            layer_key=layer_key,
            k=k,
        )
        if analysis:
            step_analyses.append(analysis)

    if not step_analyses:
        return None

    return _aggregate_trajectory_from_steps(
        step_analyses=step_analyses,
        aggregation=aggregation,
        trajectory_id=trajectory_id,
        grid_size=grid_size,
        complexity=complexity,
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


def process_trajectories_topk(
    base_dir: Path,
    layer_key: str = "model.layers.15.output",
    batch_size: int = 20,
    k: int = 3,
    aggregation: str = "weighted",
) -> AnalysisResults:
    """Process all trajectories using top-k decoded candidate analysis."""
    print(f"\nProcessing trajectories (top-k) from: {base_dir}")
    print(f"Top-k config: k={k}, aggregation={aggregation}")

    results_map = process_trajectories_multi_k(
        base_dir=base_dir,
        k_values=[int(max(1, k))],
        aggregations=[aggregation],
        layer_key=layer_key,
        batch_size=batch_size,
    )
    results = results_map[(int(max(1, k)), aggregation)]

    if not results.step_df.empty:
        results.overall_summary["topk_step_any_valid_rate"] = float(
            results.step_df["topk_any_valid"].mean()
        )
        results.overall_summary["topk_step_any_optimal_gt_match_rate"] = float(
            results.step_df["topk_any_optimal_gt_match"].mean()
        )
        results.overall_summary["topk_step_any_taken_optimal_decoded_rate"] = float(
            results.step_df["topk_any_taken_optimal_decoded"].mean()
        )
        results.overall_summary["topk_step_weighted_taken_optimal_decoded_mean"] = (
            float(results.step_df["topk_weighted_taken_optimal_decoded"].mean())
        )
        results.overall_summary["topk_step_weighted_optimal_gt_match_mean"] = float(
            results.step_df["topk_weighted_optimal_gt_match"].mean()
        )

    return results


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
            total_reverse_error_steps=("reverse_error_steps", "sum"),
            total_reverse_recovered=("reverse_recovered", "sum"),
        )
        .reset_index()
    )

    # Compute error recovery rate
    summary["error_recovery_rate"] = (
        summary["total_decoded_correct"] / summary["total_error_steps"]
    ).fillna(0)

    # Compute reverse recovery rate
    summary["reverse_recovery_rate"] = (
        summary["total_reverse_recovered"] / summary["total_reverse_error_steps"]
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
    total_reverse_error_steps = trajectory_df["reverse_error_steps"].sum()
    total_reverse_recovered = trajectory_df["reverse_recovered"].sum()

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
        "total_reverse_error_steps": int(total_reverse_error_steps),
        "total_reverse_recovered": int(total_reverse_recovered),
        "overall_reverse_recovery_rate": (
            float(total_reverse_recovered / total_reverse_error_steps)
            if total_reverse_error_steps > 0
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
            label="Error Recovery",
        )
        axes[2].set_xticks(sorted(error_summary.index))

    df_with_rev_errors = df[df["reverse_error_steps"] > 0].copy()
    if len(df_with_rev_errors) > 0:
        rev_summary = df_with_rev_errors.groupby("grid_size").agg(
            {"reverse_recovery_rate": ["mean", "sem"]}
        )
        axes[2].errorbar(
            rev_summary.index,
            rev_summary["reverse_recovery_rate"]["mean"],
            yerr=rev_summary["reverse_recovery_rate"]["sem"],
            marker="s",
            color=MODEL_COLORS[5],
            linewidth=2,
            capsize=3,
            linestyle="--",
            label="Reverse Recovery",
        )
        axes[2].set_xticks(sorted(rev_summary.index))

    axes[2].set_xlabel("Grid Size")
    axes[2].set_ylabel("Recovery Rate")
    axes[2].set_title("Recovery Rates by Grid Size")
    axes[2].legend(fontsize=7)
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
            label="Error Recovery",
        )

    df_with_rev_errors = trajectory_df[trajectory_df["reverse_error_steps"] > 0].copy()
    if len(df_with_rev_errors) > 0:
        comp_rev = df_with_rev_errors.groupby("complexity").agg(
            {"reverse_recovery_rate": ["mean", "sem"]}
        )
        axes[0, 0].errorbar(
            comp_rev.index,
            comp_rev["reverse_recovery_rate"]["mean"],
            yerr=comp_rev["reverse_recovery_rate"]["sem"],
            marker="s",
            color=MODEL_COLORS[5],
            linewidth=2,
            capsize=3,
            linestyle="--",
            label="Reverse Recovery",
        )

    axes[0, 0].set_xlabel("Complexity")
    axes[0, 0].set_ylabel("Recovery Rate")
    axes[0, 0].set
    axes[0, 0].legend(fontsize=7)
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


def plot_topk_analysis(
    trajectory_df: pd.DataFrame,
    step_df: pd.DataFrame,
    output_dir: Path,
    topk_k: Optional[int] = None,
) -> Path:
    """Plot top-k specific analysis panels using existing paper style."""
    setup_paper_style()

    inferred_k = None
    if topk_k is None and not step_df.empty and "topk_k" in step_df.columns:
        k_values = step_df["topk_k"].dropna().unique()
        if len(k_values) > 0:
            inferred_k = int(k_values[0])
    k_value = int(topk_k) if topk_k is not None else inferred_k
    k_label = f"Top-{k_value}" if k_value is not None else "Top-k"
    k_suffix = f"_k{k_value}" if k_value is not None else ""

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # 1) Taken-optimal by grid size (GT vs top-1 vs top-k)
    if not step_df.empty:
        size_taken = step_df.groupby("grid_size").agg(
            taken_is_optimal_gt_mean=("taken_is_optimal_gt", "mean"),
            taken_is_optimal_gt_sem=("taken_is_optimal_gt", "sem"),
            taken_is_optimal_decoded_mean=("taken_is_optimal_decoded", "mean"),
            taken_is_optimal_decoded_sem=("taken_is_optimal_decoded", "sem"),
            topk_any_taken_optimal_decoded_mean=(
                "topk_any_taken_optimal_decoded",
                "mean",
            ),
            topk_any_taken_optimal_decoded_sem=(
                "topk_any_taken_optimal_decoded",
                "sem",
            ),
            topk_weighted_taken_optimal_decoded_mean=(
                "topk_weighted_taken_optimal_decoded",
                "mean",
            ),
            topk_weighted_taken_optimal_decoded_sem=(
                "topk_weighted_taken_optimal_decoded",
                "sem",
            ),
        )
        axes[0, 0].errorbar(
            size_taken.index,
            size_taken["taken_is_optimal_gt_mean"],
            yerr=size_taken["taken_is_optimal_gt_sem"],
            marker="o",
            label="Taken optimal (GT)",
            color=MODEL_COLORS[0],
            capsize=3,
        )
        axes[0, 0].errorbar(
            size_taken.index,
            size_taken["taken_is_optimal_decoded_mean"],
            yerr=size_taken["taken_is_optimal_decoded_sem"],
            marker="s",
            label="Top-1 decoded",
            color=MODEL_COLORS[1],
            capsize=3,
        )
        axes[0, 0].errorbar(
            size_taken.index,
            size_taken["topk_any_taken_optimal_decoded_mean"],
            yerr=size_taken["topk_any_taken_optimal_decoded_sem"],
            marker="^",
            label=f"{k_label} any",
            color=MODEL_COLORS[2],
            capsize=3,
        )
        axes[0, 0].errorbar(
            size_taken.index,
            size_taken["topk_weighted_taken_optimal_decoded_mean"],
            yerr=size_taken["topk_weighted_taken_optimal_decoded_sem"],
            marker="d",
            label=f"{k_label} weighted",
            color=MODEL_COLORS[3],
            linestyle="--",
            capsize=3,
        )
        axes[0, 0].set_xticks(sorted(size_taken.index))
    axes[0, 0].set_xlabel("Grid Size")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_title("Taken-Optimal by Grid Size")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    # 2) GT agreement by grid size (top-1 vs top-k)
    if not step_df.empty:
        size_agree = step_df.groupby("grid_size").agg(
            optimal_gt_matches_decoded_mean=("optimal_gt_matches_decoded", "mean"),
            optimal_gt_matches_decoded_sem=("optimal_gt_matches_decoded", "sem"),
            topk_any_optimal_gt_match_mean=("topk_any_optimal_gt_match", "mean"),
            topk_any_optimal_gt_match_sem=("topk_any_optimal_gt_match", "sem"),
            topk_weighted_optimal_gt_match_mean=(
                "topk_weighted_optimal_gt_match",
                "mean",
            ),
            topk_weighted_optimal_gt_match_sem=(
                "topk_weighted_optimal_gt_match",
                "sem",
            ),
        )
        axes[0, 1].errorbar(
            size_agree.index,
            size_agree["optimal_gt_matches_decoded_mean"],
            yerr=size_agree["optimal_gt_matches_decoded_sem"],
            marker="s",
            label="Top-1 decoded",
            color=MODEL_COLORS[1],
            capsize=3,
        )
        axes[0, 1].errorbar(
            size_agree.index,
            size_agree["topk_any_optimal_gt_match_mean"],
            yerr=size_agree["topk_any_optimal_gt_match_sem"],
            marker="^",
            label=f"{k_label} any",
            color=MODEL_COLORS[2],
            capsize=3,
        )
        axes[0, 1].errorbar(
            size_agree.index,
            size_agree["topk_weighted_optimal_gt_match_mean"],
            yerr=size_agree["topk_weighted_optimal_gt_match_sem"],
            marker="d",
            label=f"{k_label} weighted",
            color=MODEL_COLORS[3],
            linestyle="--",
            capsize=3,
        )
        axes[0, 1].set_xticks(sorted(size_agree.index))
    axes[0, 1].set_xlabel("Grid Size")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_title("GT Agreement by Grid Size")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7)

    # 3) Taken-optimal by complexity (GT vs top-1 vs top-k)
    if not step_df.empty:
        comp_taken = step_df.groupby("complexity").agg(
            taken_is_optimal_gt_mean=("taken_is_optimal_gt", "mean"),
            taken_is_optimal_gt_sem=("taken_is_optimal_gt", "sem"),
            taken_is_optimal_decoded_mean=("taken_is_optimal_decoded", "mean"),
            taken_is_optimal_decoded_sem=("taken_is_optimal_decoded", "sem"),
            topk_any_taken_optimal_decoded_mean=(
                "topk_any_taken_optimal_decoded",
                "mean",
            ),
            topk_any_taken_optimal_decoded_sem=(
                "topk_any_taken_optimal_decoded",
                "sem",
            ),
            topk_weighted_taken_optimal_decoded_mean=(
                "topk_weighted_taken_optimal_decoded",
                "mean",
            ),
            topk_weighted_taken_optimal_decoded_sem=(
                "topk_weighted_taken_optimal_decoded",
                "sem",
            ),
        )
        axes[1, 0].errorbar(
            comp_taken.index,
            comp_taken["taken_is_optimal_gt_mean"],
            yerr=comp_taken["taken_is_optimal_gt_sem"],
            marker="o",
            label="Taken optimal (GT)",
            color=MODEL_COLORS[0],
            capsize=3,
        )
        axes[1, 0].errorbar(
            comp_taken.index,
            comp_taken["taken_is_optimal_decoded_mean"],
            yerr=comp_taken["taken_is_optimal_decoded_sem"],
            marker="s",
            label="Top-1 decoded",
            color=MODEL_COLORS[1],
            capsize=3,
        )
        axes[1, 0].errorbar(
            comp_taken.index,
            comp_taken["topk_any_taken_optimal_decoded_mean"],
            yerr=comp_taken["topk_any_taken_optimal_decoded_sem"],
            marker="^",
            label=f"{k_label} any",
            color=MODEL_COLORS[2],
            capsize=3,
        )
        axes[1, 0].errorbar(
            comp_taken.index,
            comp_taken["topk_weighted_taken_optimal_decoded_mean"],
            yerr=comp_taken["topk_weighted_taken_optimal_decoded_sem"],
            marker="d",
            label=f"{k_label} weighted",
            color=MODEL_COLORS[3],
            linestyle="--",
            capsize=3,
        )
    axes[1, 0].set_xlabel("Complexity")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_title("Taken-Optimal by Complexity")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    # 4) Best-candidate probability and concentration by grid size
    if not step_df.empty and "topk_total_joint_mass" in step_df.columns:
        mass_df = step_df.copy()
        mass_df["topk_mass_concentration"] = (
            mass_df["topk_best_joint_prob"] / mass_df["topk_total_joint_mass"]
        )
        mass_df.loc[
            ~np.isfinite(mass_df["topk_mass_concentration"]), "topk_mass_concentration"
        ] = np.nan

        size_mass = mass_df.groupby("grid_size").agg(
            topk_best_joint_prob_mean=("topk_best_joint_prob", "mean"),
            topk_best_joint_prob_sem=("topk_best_joint_prob", "sem"),
            topk_mass_concentration_mean=("topk_mass_concentration", "mean"),
            topk_mass_concentration_sem=("topk_mass_concentration", "sem"),
        )
        axes[1, 1].errorbar(
            size_mass.index,
            size_mass["topk_best_joint_prob_mean"],
            yerr=size_mass["topk_best_joint_prob_sem"],
            marker="o",
            color=MODEL_COLORS[4] if len(MODEL_COLORS) > 4 else MODEL_COLORS[2],
            label="Best candidate joint prob",
            capsize=3,
        )
        axes[1, 1].errorbar(
            size_mass.index,
            size_mass["topk_mass_concentration_mean"],
            yerr=size_mass["topk_mass_concentration_sem"],
            marker="s",
            color=MODEL_COLORS[6] if len(MODEL_COLORS) > 6 else MODEL_COLORS[0],
            linestyle="--",
            label="Best / total joint mass",
            capsize=3,
        )
        axes[1, 1].set_xticks(sorted(size_mass.index))
    axes[1, 1].set_xlabel("Grid Size")
    axes[1, 1].set_ylabel("Probability / Concentration")
    axes[1, 1].set_title(f"{k_label} Confidence Concentration")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    # 5) Fraction of combinations where taken action is optimal (by grid size)
    if not step_df.empty and "topk_fraction_taken_optimal" in step_df.columns:
        frac_by_size = step_df.groupby("grid_size").agg(
            topk_fraction_taken_optimal_mean=("topk_fraction_taken_optimal", "mean"),
            topk_fraction_taken_optimal_sem=("topk_fraction_taken_optimal", "sem"),
        )
        axes[2, 0].errorbar(
            frac_by_size.index,
            frac_by_size["topk_fraction_taken_optimal_mean"],
            yerr=frac_by_size["topk_fraction_taken_optimal_sem"],
            marker="o",
            color=MODEL_COLORS[5] if len(MODEL_COLORS) > 5 else MODEL_COLORS[1],
            label=f"{k_label} fraction optimal",
            capsize=3,
        )
        axes[2, 0].set_xticks(sorted(frac_by_size.index))
    axes[2, 0].set_xlabel("Grid Size")
    axes[2, 0].set_ylabel("Fraction")
    axes[2, 0].set_title(f"{k_label} Combination Optimality Fraction")
    axes[2, 0].set_ylim(0, 1.05)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(fontsize=7)

    axes[2, 1].axis("off")

    plt.suptitle(f"{k_label} Decoded Grid Analysis", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, f"topk_analysis{k_suffix}")
    plt.close(fig)
    return output_path


def plot_all_modes_comparison(
    top1_df: pd.DataFrame,
    topk_any_df: pd.DataFrame,
    topk_weighted_df: pd.DataFrame,
    output_dir: Path,
    topk_k: Optional[int] = None,
) -> Path:
    """Compare GT vs Top-1 vs Top-k(any) vs Top-k(weighted) across size/complexity."""
    setup_paper_style()

    k_value = int(topk_k) if topk_k is not None else None
    k_label = f"Top-{k_value}" if k_value is not None else "Top-k"
    k_suffix = f"_k{k_value}" if k_value is not None else ""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Taken-optimal by grid size
    size_top1 = top1_df.groupby("grid_size").agg(
        gt_mean=("taken_optimal_gt_rate", "mean"),
        gt_sem=("taken_optimal_gt_rate", "sem"),
        top1_mean=("taken_optimal_decoded_rate", "mean"),
        top1_sem=("taken_optimal_decoded_rate", "sem"),
    )
    size_any = topk_any_df.groupby("grid_size").agg(
        topk_any_mean=("taken_optimal_decoded_rate", "mean"),
        topk_any_sem=("taken_optimal_decoded_rate", "sem"),
    )
    size_weighted = topk_weighted_df.groupby("grid_size").agg(
        topk_weighted_mean=("taken_optimal_decoded_rate", "mean"),
        topk_weighted_sem=("taken_optimal_decoded_rate", "sem"),
    )
    size_join = (
        size_top1.join(size_any, how="outer")
        .join(size_weighted, how="outer")
        .sort_index()
    )

    axes[0, 0].errorbar(
        size_join.index,
        size_join["gt_mean"],
        yerr=size_join["gt_sem"],
        marker="o",
        color=MODEL_COLORS[0],
        label="Taken optimal (GT)",
        capsize=3,
    )
    axes[0, 0].errorbar(
        size_join.index,
        size_join["top1_mean"],
        yerr=size_join["top1_sem"],
        marker="s",
        color=MODEL_COLORS[1],
        label="Top-1 decoded",
        capsize=3,
    )
    axes[0, 0].errorbar(
        size_join.index,
        size_join["topk_any_mean"],
        yerr=size_join["topk_any_sem"],
        marker="^",
        color=MODEL_COLORS[2],
        label=f"{k_label} any",
        capsize=3,
    )
    axes[0, 0].errorbar(
        size_join.index,
        size_join["topk_weighted_mean"],
        yerr=size_join["topk_weighted_sem"],
        marker="d",
        color=MODEL_COLORS[3],
        label=f"{k_label} weighted",
        capsize=3,
    )
    axes[0, 0].set_title("Taken-Optimal by Grid Size")
    axes[0, 0].set_xlabel("Grid Size")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    # Taken-optimal by complexity
    comp_top1 = top1_df.groupby("complexity").agg(
        gt_mean=("taken_optimal_gt_rate", "mean"),
        gt_sem=("taken_optimal_gt_rate", "sem"),
        top1_mean=("taken_optimal_decoded_rate", "mean"),
        top1_sem=("taken_optimal_decoded_rate", "sem"),
    )
    comp_any = topk_any_df.groupby("complexity").agg(
        topk_any_mean=("taken_optimal_decoded_rate", "mean"),
        topk_any_sem=("taken_optimal_decoded_rate", "sem"),
    )
    comp_weighted = topk_weighted_df.groupby("complexity").agg(
        topk_weighted_mean=("taken_optimal_decoded_rate", "mean"),
        topk_weighted_sem=("taken_optimal_decoded_rate", "sem"),
    )
    comp_join = (
        comp_top1.join(comp_any, how="outer")
        .join(comp_weighted, how="outer")
        .sort_index()
    )

    axes[0, 1].errorbar(
        comp_join.index,
        comp_join["gt_mean"],
        yerr=comp_join["gt_sem"],
        marker="o",
        color=MODEL_COLORS[0],
        label="Taken optimal (GT)",
        capsize=3,
    )
    axes[0, 1].errorbar(
        comp_join.index,
        comp_join["top1_mean"],
        yerr=comp_join["top1_sem"],
        marker="s",
        color=MODEL_COLORS[1],
        label="Top-1 decoded",
        capsize=3,
    )
    axes[0, 1].errorbar(
        comp_join.index,
        comp_join["topk_any_mean"],
        yerr=comp_join["topk_any_sem"],
        marker="^",
        color=MODEL_COLORS[2],
        label=f"{k_label} any",
        capsize=3,
    )
    axes[0, 1].errorbar(
        comp_join.index,
        comp_join["topk_weighted_mean"],
        yerr=comp_join["topk_weighted_sem"],
        marker="d",
        color=MODEL_COLORS[3],
        label=f"{k_label} weighted",
        capsize=3,
    )
    axes[0, 1].set_title("Taken-Optimal by Complexity")
    axes[0, 1].set_xlabel("Complexity")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7)

    # GT-decoded agreement by grid size
    agree_top1 = top1_df.groupby("grid_size").agg(
        top1_mean=("gt_decoded_agreement_rate", "mean"),
        top1_sem=("gt_decoded_agreement_rate", "sem"),
    )
    agree_any = topk_any_df.groupby("grid_size").agg(
        topk_any_mean=("gt_decoded_agreement_rate", "mean"),
        topk_any_sem=("gt_decoded_agreement_rate", "sem"),
    )
    agree_weighted = topk_weighted_df.groupby("grid_size").agg(
        topk_weighted_mean=("gt_decoded_agreement_rate", "mean"),
        topk_weighted_sem=("gt_decoded_agreement_rate", "sem"),
    )
    agree_size = (
        agree_top1.join(agree_any, how="outer")
        .join(agree_weighted, how="outer")
        .sort_index()
    )

    axes[1, 0].errorbar(
        agree_size.index,
        agree_size["top1_mean"],
        yerr=agree_size["top1_sem"],
        marker="s",
        color=MODEL_COLORS[1],
        label="Top-1 decoded",
        capsize=3,
    )
    axes[1, 0].errorbar(
        agree_size.index,
        agree_size["topk_any_mean"],
        yerr=agree_size["topk_any_sem"],
        marker="^",
        color=MODEL_COLORS[2],
        label=f"{k_label} any",
        capsize=3,
    )
    axes[1, 0].errorbar(
        agree_size.index,
        agree_size["topk_weighted_mean"],
        yerr=agree_size["topk_weighted_sem"],
        marker="d",
        color=MODEL_COLORS[3],
        label=f"{k_label} weighted",
        capsize=3,
    )
    axes[1, 0].set_title("GT Agreement by Grid Size")
    axes[1, 0].set_xlabel("Grid Size")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    # GT-decoded agreement by complexity
    agree_top1_c = top1_df.groupby("complexity").agg(
        top1_mean=("gt_decoded_agreement_rate", "mean"),
        top1_sem=("gt_decoded_agreement_rate", "sem"),
    )
    agree_any_c = topk_any_df.groupby("complexity").agg(
        topk_any_mean=("gt_decoded_agreement_rate", "mean"),
        topk_any_sem=("gt_decoded_agreement_rate", "sem"),
    )
    agree_weighted_c = topk_weighted_df.groupby("complexity").agg(
        topk_weighted_mean=("gt_decoded_agreement_rate", "mean"),
        topk_weighted_sem=("gt_decoded_agreement_rate", "sem"),
    )
    agree_comp = (
        agree_top1_c.join(agree_any_c, how="outer")
        .join(agree_weighted_c, how="outer")
        .sort_index()
    )

    axes[1, 1].errorbar(
        agree_comp.index,
        agree_comp["top1_mean"],
        yerr=agree_comp["top1_sem"],
        marker="s",
        color=MODEL_COLORS[1],
        label="Top-1 decoded",
        capsize=3,
    )
    axes[1, 1].errorbar(
        agree_comp.index,
        agree_comp["topk_any_mean"],
        yerr=agree_comp["topk_any_sem"],
        marker="^",
        color=MODEL_COLORS[2],
        label=f"{k_label} any",
        capsize=3,
    )
    axes[1, 1].errorbar(
        agree_comp.index,
        agree_comp["topk_weighted_mean"],
        yerr=agree_comp["topk_weighted_sem"],
        marker="d",
        color=MODEL_COLORS[3],
        label=f"{k_label} weighted",
        capsize=3,
    )
    axes[1, 1].set_title("GT Agreement by Complexity")
    axes[1, 1].set_xlabel("Complexity")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    plt.suptitle(
        "Comprehensive Comparison: GT vs Top-1 vs {}".format(k_label),
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = save_figure(
        fig, output_dir, "all_modes_comparison{}".format(k_suffix)
    )
    plt.close(fig)
    return output_path


def plot_topk_sweep(
    sweep_df: pd.DataFrame,
    output_dir: Path,
    aggregation: str,
) -> Path:
    """Plot comparison of key metrics across k values."""
    setup_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    k_vals = sweep_df["k"].tolist()

    if aggregation == "both":
        axes[0, 0].plot(
            k_vals,
            sweep_df["overall_taken_optimal_decoded_any"],
            marker="o",
            color=MODEL_COLORS[2],
            label="Decoded taken-optimal (any)",
        )
        axes[0, 0].plot(
            k_vals,
            sweep_df["overall_taken_optimal_decoded_weighted"],
            marker="s",
            color=MODEL_COLORS[3],
            linestyle="--",
            label="Decoded taken-optimal (weighted)",
        )
        axes[0, 0].plot(
            k_vals,
            sweep_df["overall_taken_optimal_gt"],
            marker="^",
            color=MODEL_COLORS[0],
            linestyle=":",
            label="GT taken-optimal",
        )
    else:
        axes[0, 0].plot(
            k_vals,
            sweep_df["overall_taken_optimal_decoded"],
            marker="o",
            color=MODEL_COLORS[1],
            label="Decoded taken-optimal",
        )
        axes[0, 0].plot(
            k_vals,
            sweep_df["overall_taken_optimal_gt"],
            marker="s",
            color=MODEL_COLORS[0],
            linestyle="--",
            label="GT taken-optimal",
        )
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_title("Taken-Optimal vs k")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    if aggregation == "both":
        axes[0, 1].plot(
            k_vals,
            sweep_df["overall_gt_decoded_agreement_any"],
            marker="o",
            color=MODEL_COLORS[2],
            label="GT-decoded agreement (any)",
        )
        axes[0, 1].plot(
            k_vals,
            sweep_df["overall_gt_decoded_agreement_weighted"],
            marker="s",
            color=MODEL_COLORS[3],
            linestyle="--",
            label="GT-decoded agreement (weighted)",
        )
    else:
        axes[0, 1].plot(
            k_vals,
            sweep_df["overall_gt_decoded_agreement"],
            marker="o",
            color=MODEL_COLORS[2],
            label="GT-decoded agreement",
        )
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_title("GT Agreement vs k")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7)

    if aggregation == "both":
        axes[1, 0].plot(
            k_vals,
            sweep_df["overall_error_recovery_rate_any"],
            marker="o",
            color=MODEL_COLORS[2],
            label="Error recovery (any)",
        )
        axes[1, 0].plot(
            k_vals,
            sweep_df["overall_error_recovery_rate_weighted"],
            marker="s",
            color=MODEL_COLORS[3],
            linestyle="--",
            label="Error recovery (weighted)",
        )
    else:
        axes[1, 0].plot(
            k_vals,
            sweep_df["overall_error_recovery_rate"],
            marker="o",
            color=MODEL_COLORS[3],
            label="Error recovery rate",
        )
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_title("Error Recovery vs k")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    if aggregation == "both":
        axes[1, 1].plot(
            k_vals,
            sweep_df["mean_topk_fraction_taken_optimal_any"],
            marker="o",
            color=MODEL_COLORS[2],
            label="Combo optimal fraction (any)",
        )
        axes[1, 1].plot(
            k_vals,
            sweep_df["mean_topk_fraction_taken_optimal_weighted"],
            marker="s",
            color=MODEL_COLORS[3],
            linestyle="--",
            label="Combo optimal fraction (weighted)",
        )
    else:
        axes[1, 1].plot(
            k_vals,
            sweep_df["mean_topk_fraction_taken_optimal"],
            marker="o",
            color=MODEL_COLORS[5] if len(MODEL_COLORS) > 5 else MODEL_COLORS[1],
            label="Fraction of combinations optimal",
        )
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_title("Combination Optimality vs k")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    plt.suptitle("Top-k Sweep Comparison ({})".format(aggregation), fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(
        fig, output_dir, "topk_sweep_k1-{}_{}".format(int(max(k_vals)), aggregation)
    )
    plt.close(fig)
    return output_path


def plot_taken_optimal_k_compare(
    top1_df: pd.DataFrame,
    series: list[dict[str, Any]],
    output_dir: Path,
    k_values: list[int],
) -> Path:
    """Plot taken-optimal by size/complexity for k values (any/weighted) with baselines."""
    setup_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    k_suffix = "_k{}".format("_k".join(str(k) for k in k_values))

    # Paper-style consistent colors
    baseline_styles = {
        "Taken optimal (GT)": {
            "color": MODEL_COLORS[0],  # blue
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 1.0,
            "zorder": 5,
        },
        "Top-1 decoded": {
            "color": MODEL_COLORS[1],  # orange
            "marker": "s",
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 1.0,
            "zorder": 4,
        },
    }

    any_palette = [MODEL_COLORS[2], MODEL_COLORS[3], MODEL_COLORS[4], MODEL_COLORS[5]]
    weighted_palette = [
        MODEL_COLORS[2],
        MODEL_COLORS[3],
        MODEL_COLORS[4],
        MODEL_COLORS[5],
    ]

    # Assign style per series label so lines are easy to distinguish
    series_styles: dict[str, dict[str, Any]] = {}
    any_idx = 0
    weighted_idx = 0
    for item in series:
        label = item.get("label", "")
        if " any" in label:
            color = any_palette[any_idx % len(any_palette)]
            any_idx += 1
            series_styles[label] = {
                "color": color,
                "marker": "^",
                "linestyle": "-",
                "linewidth": 1.8,
                "alpha": 0.95,
                "zorder": 3,
            }
        else:
            color = weighted_palette[weighted_idx % len(weighted_palette)]
            weighted_idx += 1
            series_styles[label] = {
                "color": color,
                "marker": "d",
                "linestyle": "--",
                "linewidth": 1.8,
                "alpha": 0.95,
                "zorder": 3,
            }

    # Track ranges including error bars for data-driven y-limits
    size_ymins: list[float] = []
    size_ymaxs: list[float] = []
    comp_ymins: list[float] = []
    comp_ymaxs: list[float] = []

    # Grid size
    size_gt = top1_df.groupby("grid_size").agg(
        mean=("taken_optimal_gt_rate", "mean"),
        sem=("taken_optimal_gt_rate", "sem"),
    )
    size_top1 = top1_df.groupby("grid_size").agg(
        mean=("taken_optimal_decoded_rate", "mean"),
        sem=("taken_optimal_decoded_rate", "sem"),
    )

    size_gt_err = size_gt["sem"].fillna(0.0)
    size_top1_err = size_top1["sem"].fillna(0.0)

    axes[0].errorbar(
        size_gt.index,
        size_gt["mean"],
        yerr=size_gt_err,
        marker=baseline_styles["Taken optimal (GT)"]["marker"],
        color=baseline_styles["Taken optimal (GT)"]["color"],
        linestyle=baseline_styles["Taken optimal (GT)"]["linestyle"],
        linewidth=baseline_styles["Taken optimal (GT)"]["linewidth"],
        alpha=baseline_styles["Taken optimal (GT)"]["alpha"],
        zorder=baseline_styles["Taken optimal (GT)"]["zorder"],
        label="Taken optimal (GT)",
        capsize=3,
    )
    axes[0].errorbar(
        size_top1.index,
        size_top1["mean"],
        yerr=size_top1_err,
        marker=baseline_styles["Top-1 decoded"]["marker"],
        color=baseline_styles["Top-1 decoded"]["color"],
        linestyle=baseline_styles["Top-1 decoded"]["linestyle"],
        linewidth=baseline_styles["Top-1 decoded"]["linewidth"],
        alpha=baseline_styles["Top-1 decoded"]["alpha"],
        zorder=baseline_styles["Top-1 decoded"]["zorder"],
        label="Top-1 decoded",
        capsize=3,
    )

    size_ymins.extend((size_gt["mean"] - size_gt_err).tolist())
    size_ymaxs.extend((size_gt["mean"] + size_gt_err).tolist())
    size_ymins.extend((size_top1["mean"] - size_top1_err).tolist())
    size_ymaxs.extend((size_top1["mean"] + size_top1_err).tolist())

    for item in series:
        size_k = (
            item["df"]
            .groupby("grid_size")
            .agg(
                mean=("taken_optimal_decoded_rate", "mean"),
                sem=("taken_optimal_decoded_rate", "sem"),
            )
        )
        style = series_styles.get(
            item["label"],
            {
                "color": item["color"],
                "marker": item["marker"],
                "linestyle": item["linestyle"],
                "linewidth": 1.8,
                "alpha": 0.95,
                "zorder": 3,
            },
        )
        size_k_err = size_k["sem"].fillna(0.0)
        axes[0].errorbar(
            size_k.index,
            size_k["mean"],
            yerr=size_k_err,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=item["label"],
            capsize=3,
        )
        size_ymins.extend((size_k["mean"] - size_k_err).tolist())
        size_ymaxs.extend((size_k["mean"] + size_k_err).tolist())

    axes[0].set_title("Taken-Optimal by Grid Size")
    axes[0].set_xlabel("Grid Size")
    axes[0].set_ylabel("Rate")
    axes[0].grid(True, alpha=0.3)

    # Complexity
    comp_gt = top1_df.groupby("complexity").agg(
        mean=("taken_optimal_gt_rate", "mean"),
        sem=("taken_optimal_gt_rate", "sem"),
    )
    comp_top1 = top1_df.groupby("complexity").agg(
        mean=("taken_optimal_decoded_rate", "mean"),
        sem=("taken_optimal_decoded_rate", "sem"),
    )

    comp_gt_err = comp_gt["sem"].fillna(0.0)
    comp_top1_err = comp_top1["sem"].fillna(0.0)

    axes[1].errorbar(
        comp_gt.index,
        comp_gt["mean"],
        yerr=comp_gt_err,
        marker=baseline_styles["Taken optimal (GT)"]["marker"],
        color=baseline_styles["Taken optimal (GT)"]["color"],
        linestyle=baseline_styles["Taken optimal (GT)"]["linestyle"],
        linewidth=baseline_styles["Taken optimal (GT)"]["linewidth"],
        alpha=baseline_styles["Taken optimal (GT)"]["alpha"],
        zorder=baseline_styles["Taken optimal (GT)"]["zorder"],
        label="Taken optimal (GT)",
        capsize=3,
    )
    axes[1].errorbar(
        comp_top1.index,
        comp_top1["mean"],
        yerr=comp_top1_err,
        marker=baseline_styles["Top-1 decoded"]["marker"],
        color=baseline_styles["Top-1 decoded"]["color"],
        linestyle=baseline_styles["Top-1 decoded"]["linestyle"],
        linewidth=baseline_styles["Top-1 decoded"]["linewidth"],
        alpha=baseline_styles["Top-1 decoded"]["alpha"],
        zorder=baseline_styles["Top-1 decoded"]["zorder"],
        label="Top-1 decoded",
        capsize=3,
    )

    comp_ymins.extend((comp_gt["mean"] - comp_gt_err).tolist())
    comp_ymaxs.extend((comp_gt["mean"] + comp_gt_err).tolist())
    comp_ymins.extend((comp_top1["mean"] - comp_top1_err).tolist())
    comp_ymaxs.extend((comp_top1["mean"] + comp_top1_err).tolist())

    for item in series:
        comp_k = (
            item["df"]
            .groupby("complexity")
            .agg(
                mean=("taken_optimal_decoded_rate", "mean"),
                sem=("taken_optimal_decoded_rate", "sem"),
            )
        )
        style = series_styles.get(
            item["label"],
            {
                "color": item["color"],
                "marker": item["marker"],
                "linestyle": item["linestyle"],
                "linewidth": 1.8,
                "alpha": 0.95,
                "zorder": 3,
            },
        )
        comp_k_err = comp_k["sem"].fillna(0.0)
        axes[1].errorbar(
            comp_k.index,
            comp_k["mean"],
            yerr=comp_k_err,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=item["label"],
            capsize=3,
        )
        comp_ymins.extend((comp_k["mean"] - comp_k_err).tolist())
        comp_ymaxs.extend((comp_k["mean"] + comp_k_err).tolist())

    axes[1].set_title("Taken-Optimal by Complexity")
    axes[1].set_xlabel("Complexity")
    axes[1].set_ylabel("Rate")
    axes[1].grid(True, alpha=0.3)

    # Data-driven y-limits with padding; clamp to valid rate range [0, 1]
    def _compute_dynamic_ymin(
        ymins_a: list[float],
        ymaxs_a: list[float],
        ymins_b: list[float],
        ymaxs_b: list[float],
    ) -> float:
        valid_vals = [
            float(v) for v in (ymins_a + ymaxs_a + ymins_b + ymaxs_b) if np.isfinite(v)
        ]
        if not valid_vals:
            return 0.0

        data_min = min(valid_vals)
        data_max = max(valid_vals)
        span = max(data_max - data_min, 1e-6)
        pad = max(0.05 * span, 0.01)

        y_lower = max(0.0, data_min - pad)

        # Keep enough vertical span while enforcing upper bound at 1.0
        if 1.0 - y_lower < 0.05:
            y_lower = max(0.0, 0.95)

        return y_lower

    shared_ymin = _compute_dynamic_ymin(size_ymins, size_ymaxs, comp_ymins, comp_ymaxs)
    axes[0].set_ylim(shared_ymin, 1.0)
    axes[1].set_ylim(shared_ymin, 1.0)

    # Single combined legend above plots spanning full figure width
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        markerscale=1.4,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.4,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = save_figure(fig, output_dir, "decoded_optimality{}".format(k_suffix))
    plt.close(fig)
    return output_path


def run_taken_optimal_k_compare(
    base_dir: Path,
    output_dir: Path,
    layer_key: str = "model.layers.15.output",
    batch_size: int = 40,
    k_values: Optional[list[int]] = None,
    compare_aggregation: str = "any",
) -> Path:
    """Run analysis for specific k values and generate taken-optimal comparison PDF."""
    k_list = sorted({int(k) for k in (k_values or [5, 10]) if int(k) > 0})
    k_process = sorted({1, *k_list})
    aggs = ["top1"]
    if compare_aggregation in ("any", "both"):
        aggs.append("any")
    if compare_aggregation in ("weighted", "both"):
        aggs.append("weighted")

    results_map = process_trajectories_multi_k(
        base_dir=base_dir,
        k_values=k_process,
        aggregations=aggs,
        layer_key=layer_key,
        batch_size=batch_size,
    )

    top1_results = results_map[(1, "top1")]

    series: list[dict[str, Any]] = []
    any_colors = [MODEL_COLORS[2], MODEL_COLORS[4], MODEL_COLORS[6]]
    weighted_colors = [MODEL_COLORS[3], MODEL_COLORS[5], MODEL_COLORS[1]]

    for idx, k in enumerate(k_list):
        if compare_aggregation in ("any", "both"):
            res_any = results_map[(k, "any")]
            series.append(
                {
                    "label": "Top-{} decoded".format(k),
                    "df": res_any.trajectory_df,
                    "color": any_colors[idx % len(any_colors)],
                    "marker": "^",
                    "linestyle": "-",
                }
            )
        if compare_aggregation in ("weighted", "both"):
            res_weighted = results_map[(k, "weighted")]
            series.append(
                {
                    "label": "Top-{} weighted".format(k),
                    "df": res_weighted.trajectory_df,
                    "color": weighted_colors[idx % len(weighted_colors)],
                    "marker": "d",
                    "linestyle": "--",
                }
            )

    return plot_taken_optimal_k_compare(
        top1_df=top1_results.trajectory_df,
        series=series,
        output_dir=output_dir,
        k_values=k_list,
    )


def _aggregate_trajectory_from_steps(
    step_analyses: list[StepAnalysis],
    aggregation: str,
    trajectory_id: str,
    grid_size: int,
    complexity: float,
) -> TrajectoryAnalysis:
    """Aggregate trajectory-level metrics from per-step analyses."""
    n_steps = len(step_analyses)
    if n_steps == 0:
        return TrajectoryAnalysis(
            trajectory_id=trajectory_id,
            grid_size=grid_size,
            complexity=complexity,
            num_steps=0,
            n_valid_decoded=0,
            taken_optimal_gt_rate=0.0,
            taken_optimal_decoded_rate=0.0,
            gt_decoded_agreement_rate=0.0,
            error_steps=0,
            decoded_correct_on_errors=0,
            reverse_error_steps=0,
            reverse_recovered=0,
            step_analyses=[],
        )

    if aggregation == "any":
        valid_flags = [s.topk_any_valid for s in step_analyses]
    elif aggregation == "weighted":
        valid_flags = [s.topk_any_valid for s in step_analyses]
    else:
        valid_flags = [s.decoded_valid for s in step_analyses]

    n_valid = sum(1 for v in valid_flags if v)
    taken_optimal_gt_rate = (
        sum(1 for s in step_analyses if s.taken_is_optimal_gt) / n_steps
    )

    if aggregation == "any":
        taken_optimal_decoded_rate = (
            sum(
                1
                for s in step_analyses
                if s.topk_any_valid and s.topk_any_taken_optimal_decoded
            )
            / n_valid
            if n_valid > 0
            else 0.0
        )
        gt_decoded_agreement_rate = (
            sum(
                1
                for s in step_analyses
                if s.topk_any_valid and s.topk_any_optimal_gt_match
            )
            / n_valid
            if n_valid > 0
            else 0.0
        )
        error_steps = [
            s for s in step_analyses if not s.taken_is_optimal_gt and s.topk_any_valid
        ]
        decoded_correct_on_errors = sum(
            1 for s in error_steps if s.topk_any_optimal_gt_match
        )

    elif aggregation == "weighted":
        weighted_taken = [
            s.topk_weighted_taken_optimal_decoded
            for s in step_analyses
            if s.topk_any_valid and not pd.isna(s.topk_weighted_taken_optimal_decoded)
        ]
        weighted_match = [
            s.topk_weighted_optimal_gt_match
            for s in step_analyses
            if s.topk_any_valid and not pd.isna(s.topk_weighted_optimal_gt_match)
        ]
        taken_optimal_decoded_rate = (
            float(np.mean(weighted_taken)) if weighted_taken else 0.0
        )
        gt_decoded_agreement_rate = (
            float(np.mean(weighted_match)) if weighted_match else 0.0
        )

        error_steps = [
            s for s in step_analyses if not s.taken_is_optimal_gt and s.topk_any_valid
        ]
        error_weighted_match = [
            s.topk_weighted_optimal_gt_match
            for s in error_steps
            if not pd.isna(s.topk_weighted_optimal_gt_match)
        ]
        decoded_correct_on_errors = (
            int(round(float(np.sum(error_weighted_match))))
            if error_weighted_match
            else 0
        )

    else:
        taken_optimal_decoded_rate = (
            sum(
                1
                for s in step_analyses
                if s.decoded_valid and s.taken_is_optimal_decoded
            )
            / n_valid
            if n_valid > 0
            else 0.0
        )
        gt_decoded_agreement_rate = (
            sum(
                1
                for s in step_analyses
                if s.decoded_valid and s.optimal_gt_matches_decoded
            )
            / n_valid
            if n_valid > 0
            else 0.0
        )
        error_steps = [
            s for s in step_analyses if not s.taken_is_optimal_gt and s.decoded_valid
        ]
        decoded_correct_on_errors = sum(
            1 for s in error_steps if s.decoded_optimal_actions & s.optimal_actions_gt
        )

    if aggregation in {"any", "weighted"}:
        reverse_error_steps = [
            s
            for s in step_analyses
            if s.topk_any_valid and not s.topk_any_taken_optimal_decoded
        ]
    else:
        reverse_error_steps = [
            s
            for s in step_analyses
            if s.decoded_valid and not s.taken_is_optimal_decoded
        ]
    reverse_recovered = sum(1 for s in reverse_error_steps if s.taken_is_optimal_gt)

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
        reverse_error_steps=len(reverse_error_steps),
        reverse_recovered=reverse_recovered,
        step_analyses=step_analyses,
    )


def process_trajectories_multi_k(
    base_dir: Path,
    k_values: list[int],
    aggregations: list[str],
    layer_key: str = "model.layers.15.output",
    batch_size: int = 20,
) -> dict[tuple[int, str], AnalysisResults]:
    """Process trajectories once while computing multiple k/aggregation variants."""
    print(f"\nProcessing trajectories (multi-k) from: {base_dir}")
    print(f"k values: {sorted(set(k_values))}, aggregations: {aggregations}")

    k_list = sorted({int(k) for k in k_values if int(k) > 0})
    size_trajectories = discover_trajectory_files(base_dir)
    total_files = sum(len(files) for files in size_trajectories.values())
    print(
        f"Found {total_files} trajectory files across {len(size_trajectories)} size groups"
    )

    if not size_trajectories:
        raise ValueError(f"No trajectory files found in {base_dir}")

    all_trajectory_results: dict[tuple[int, str], list[TrajectoryAnalysis]] = {
        (k, agg): [] for k in k_list for agg in aggregations
    }
    all_step_results: dict[tuple[int, str], list[StepAnalysis]] = {
        (k, agg): [] for k in k_list for agg in aggregations
    }

    for size_key, files in sorted(size_trajectories.items()):
        print(f"\n  Processing {size_key}: {len(files)} files")
        total_batches = (len(files) + batch_size - 1) // batch_size

        for batch_idx, batch_files in enumerate(batch_file_list(files, batch_size)):
            for filepath in tqdm(
                batch_files,
                desc=f"{size_key} batch {batch_idx + 1}/{total_batches}",
                leave=False,
            ):
                parsed = parse_trajectory_filename(filepath.name)
                if not parsed:
                    continue

                metadata = load_trajectory_metadata(filepath)
                if not metadata:
                    continue

                grid_size = metadata["grid_size"]
                complexity = metadata["complexity"]
                trajectory_id = f"{parsed['model']}_size{grid_size}_comp{complexity}_{parsed['instance_id']}"

                max_k = max(k_list)
                step_analyses_by_k: dict[int, list[StepAnalysis]] = {
                    k: [] for k in k_list
                }

                for step_data in load_step_data(filepath):
                    grid_state = step_data.get("grid_state")
                    probes = step_data.get("probes")
                    taken_action = step_data.get("agent_action", "").upper()
                    step_id = step_data.get("step_id", -1)

                    if not grid_state or not probes or not taken_action:
                        continue

                    try:
                        gt_env = gridstate2env(grid_state)
                        optimal_gt = get_optimal_actions_gt(gt_env)
                    except Exception:
                        continue

                    manhattan_metrics = {
                        "avg_agent_distance": float("nan"),
                        "avg_goal_distance": float("nan"),
                        "min_agent_distance": float("nan"),
                        "min_goal_distance": float("nan"),
                        "max_agent_distance": float("nan"),
                        "max_goal_distance": float("nan"),
                    }
                    try:
                        true_agent_pos, true_goal_pos = get_true_positions(grid_state)
                        agent_cells, goal_cells = get_classified_cells(
                            probes, layer_key=layer_key
                        )
                        manhattan_metrics = calculate_manhattan_distance_metrics(
                            agent_cells, goal_cells, true_agent_pos, true_goal_pos
                        )
                    except Exception:
                        pass

                    try:
                        candidates_max = probe2env_topk_candidates(
                            probes, layer_key=layer_key, k=max_k
                        )
                    except Exception:
                        candidates_max = []

                    candidates_sorted = sorted(
                        candidates_max, key=lambda c: c.joint_prob, reverse=True
                    )
                    (
                        candidate_decoded_optimal,
                        candidate_valid,
                        candidate_taken_optimal,
                        candidate_match_gt,
                        joint_probs,
                    ) = _compute_candidate_metrics(
                        candidates_sorted, taken_action, optimal_gt
                    )

                    for k in k_list:
                        mask = [
                            c.agent_rank <= k and c.goal_rank <= k
                            for c in candidates_sorted
                        ]
                        candidates_k = [
                            c for c, keep in zip(candidates_sorted, mask) if keep
                        ]
                        decoded_opt_k = [
                            d
                            for d, keep in zip(candidate_decoded_optimal, mask)
                            if keep
                        ]
                        valid_k = [v for v, keep in zip(candidate_valid, mask) if keep]
                        taken_opt_k = [
                            t for t, keep in zip(candidate_taken_optimal, mask) if keep
                        ]
                        match_gt_k = [
                            m for m, keep in zip(candidate_match_gt, mask) if keep
                        ]
                        joint_probs_k = [
                            p for p, keep in zip(joint_probs, mask) if keep
                        ]

                        analysis = _build_step_analysis_from_metrics(
                            step_id=step_id,
                            trajectory_id=trajectory_id,
                            grid_size=grid_size,
                            complexity=complexity,
                            taken_action=taken_action,
                            optimal_gt=optimal_gt,
                            manhattan_metrics=manhattan_metrics,
                            candidates_sorted=candidates_k,
                            candidate_decoded_optimal=decoded_opt_k,
                            candidate_valid=valid_k,
                            candidate_taken_optimal=taken_opt_k,
                            candidate_match_gt=match_gt_k,
                            joint_probs=joint_probs_k,
                            k=k,
                        )
                        if analysis:
                            step_analyses_by_k[k].append(analysis)

                for k in k_list:
                    steps_for_k = step_analyses_by_k[k]
                    if not steps_for_k:
                        continue
                    for agg in aggregations:
                        traj = _aggregate_trajectory_from_steps(
                            steps_for_k, agg, trajectory_id, grid_size, complexity
                        )
                        all_trajectory_results[(k, agg)].append(traj)
                        all_step_results[(k, agg)].extend(steps_for_k)

            gc.collect()

    results_map: dict[tuple[int, str], AnalysisResults] = {}
    for k in k_list:
        for agg in aggregations:
            traj_df = pd.DataFrame(
                [t.to_dict() for t in all_trajectory_results[(k, agg)]]
            )
            step_df = pd.DataFrame([s.to_dict() for s in all_step_results[(k, agg)]])
            summary_df = compute_summary_by_size_complexity(traj_df, step_df)
            overall = compute_overall_summary(traj_df, step_df)
            overall["decode_mode"] = "topk" if agg != "top1" else "top1"
            overall["topk_k"] = int(k)
            overall["topk_aggregation"] = agg

            results_map[(k, agg)] = AnalysisResults(
                trajectory_df=traj_df,
                step_df=step_df,
                summary_by_size_complexity=summary_df,
                overall_summary=overall,
            )

    return results_map


def run_topk_sweep(
    base_dir: Path,
    output_dir: Path,
    layer_key: str = "model.layers.15.output",
    batch_size: int = 40,
    topk_max: int = 5,
    aggregation: str = "weighted",
) -> pd.DataFrame:
    """Run top-k analysis for k=1..N and generate a sweep comparison figure."""
    print(f"\nRunning top-k sweep: k=1..{topk_max} (aggregation={aggregation})")

    k_list = list(range(1, max(1, topk_max) + 1))
    aggs = ["any", "weighted"] if aggregation == "both" else [aggregation]

    results_map = process_trajectories_multi_k(
        base_dir=base_dir,
        k_values=k_list,
        aggregations=aggs,
        layer_key=layer_key,
        batch_size=batch_size,
    )

    sweep_rows: list[dict[str, Any]] = []
    for k in k_list:
        if aggregation == "both":
            results_any = results_map[(k, "any")]
            results_weighted = results_map[(k, "weighted")]

            save_results(
                results_any,
                output_dir=output_dir,
                prefix=f"topk_k{k}_any",
                include_topk_visualizations=True,
            )
            save_results(
                results_weighted,
                output_dir=output_dir,
                prefix=f"topk_k{k}_weighted",
                include_topk_visualizations=True,
            )

            overall_any = results_any.overall_summary
            overall_weighted = results_weighted.overall_summary

            mean_fraction_any = (
                float(results_any.step_df["topk_fraction_taken_optimal"].mean())
                if not results_any.step_df.empty
                and "topk_fraction_taken_optimal" in results_any.step_df.columns
                else float("nan")
            )
            mean_fraction_weighted = (
                float(results_weighted.step_df["topk_fraction_taken_optimal"].mean())
                if not results_weighted.step_df.empty
                and "topk_fraction_taken_optimal" in results_weighted.step_df.columns
                else float("nan")
            )

            sweep_rows.append(
                {
                    "k": k,
                    "overall_taken_optimal_gt": float(
                        overall_any.get("overall_taken_optimal_gt", float("nan"))
                    ),
                    "overall_taken_optimal_decoded_any": float(
                        overall_any.get("overall_taken_optimal_decoded", float("nan"))
                    ),
                    "overall_taken_optimal_decoded_weighted": float(
                        overall_weighted.get(
                            "overall_taken_optimal_decoded", float("nan")
                        )
                    ),
                    "overall_gt_decoded_agreement_any": float(
                        overall_any.get("overall_gt_decoded_agreement", float("nan"))
                    ),
                    "overall_gt_decoded_agreement_weighted": float(
                        overall_weighted.get(
                            "overall_gt_decoded_agreement", float("nan")
                        )
                    ),
                    "overall_error_recovery_rate_any": float(
                        overall_any.get("overall_error_recovery_rate", float("nan"))
                    ),
                    "overall_error_recovery_rate_weighted": float(
                        overall_weighted.get(
                            "overall_error_recovery_rate", float("nan")
                        )
                    ),
                    "mean_topk_fraction_taken_optimal_any": mean_fraction_any,
                    "mean_topk_fraction_taken_optimal_weighted": mean_fraction_weighted,
                }
            )
        else:
            results = results_map[(k, aggregation)]

            save_results(
                results,
                output_dir=output_dir,
                prefix=f"topk_k{k}_{aggregation}",
                include_topk_visualizations=True,
            )

            overall = results.overall_summary
            mean_fraction = (
                float(results.step_df["topk_fraction_taken_optimal"].mean())
                if not results.step_df.empty
                and "topk_fraction_taken_optimal" in results.step_df.columns
                else float("nan")
            )

            sweep_rows.append(
                {
                    "k": k,
                    "overall_taken_optimal_gt": float(
                        overall.get("overall_taken_optimal_gt", float("nan"))
                    ),
                    "overall_taken_optimal_decoded": float(
                        overall.get("overall_taken_optimal_decoded", float("nan"))
                    ),
                    "overall_gt_decoded_agreement": float(
                        overall.get("overall_gt_decoded_agreement", float("nan"))
                    ),
                    "overall_error_recovery_rate": float(
                        overall.get("overall_error_recovery_rate", float("nan"))
                    ),
                    "mean_topk_fraction_taken_optimal": mean_fraction,
                }
            )

    sweep_df = pd.DataFrame(sweep_rows)
    plot_topk_sweep(sweep_df, output_dir, aggregation=aggregation)

    sweep_path = output_dir / f"topk_sweep_k1-{max(1, topk_max)}_{aggregation}.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Saved: {sweep_path}")

    return sweep_df


def run_comprehensive_analysis(
    base_dir: Path,
    output_dir: Path,
    layer_key: str = "model.layers.15.output",
    batch_size: int = 40,
    topk: int = 3,
) -> dict[str, Any]:
    """Run top1 + topk(any) + topk(weighted), save all outputs, and produce combined figure."""
    print("\nRunning comprehensive analysis (top1 + topk-any + topk-weighted)")

    k_list = sorted({1, int(max(1, topk))})
    results_map = process_trajectories_multi_k(
        base_dir=base_dir,
        k_values=k_list,
        aggregations=["top1", "any", "weighted"],
        layer_key=layer_key,
        batch_size=batch_size,
    )

    top1_results = results_map[(1, "top1")]
    topk_any_results = results_map[(int(max(1, topk)), "any")]
    topk_weighted_results = results_map[(int(max(1, topk)), "weighted")]

    save_results(
        top1_results,
        output_dir=output_dir,
        prefix="top1",
        include_topk_visualizations=False,
    )
    save_results(
        topk_any_results,
        output_dir=output_dir,
        prefix=f"topk_any_k{max(1, topk)}",
        include_topk_visualizations=True,
    )
    save_results(
        topk_weighted_results,
        output_dir=output_dir,
        prefix=f"topk_weighted_k{max(1, topk)}",
        include_topk_visualizations=True,
    )

    plot_all_modes_comparison(
        top1_df=top1_results.trajectory_df,
        topk_any_df=topk_any_results.trajectory_df,
        topk_weighted_df=topk_weighted_results.trajectory_df,
        output_dir=output_dir,
        topk_k=max(1, topk),
    )

    comprehensive_summary = {
        "decode_mode": "comprehensive",
        "topk_k": int(max(1, topk)),
        "top1_overall": top1_results.overall_summary,
        "topk_any_overall": topk_any_results.overall_summary,
        "topk_weighted_overall": topk_weighted_results.overall_summary,
    }

    summary_path = output_dir / "comprehensive_overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump(comprehensive_summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    return comprehensive_summary


def save_results(
    results: AnalysisResults,
    output_dir: Path,
    prefix: str = "",
    include_topk_visualizations: bool = False,
) -> dict[str, Path]:
    """Save results to CSV files and generate visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    name_prefix = f"{prefix}_" if prefix else ""

    # Save trajectory-level metrics
    traj_path = output_dir / f"{name_prefix}trajectory_metrics.csv"
    results.trajectory_df.to_csv(traj_path, index=False)
    output_paths["trajectory_metrics"] = traj_path
    print(f"  Saved: {traj_path}")

    # Save step-level metrics
    step_path = output_dir / f"{name_prefix}step_metrics.csv"
    results.step_df.to_csv(step_path, index=False)
    output_paths["step_metrics"] = step_path
    print(f"  Saved: {step_path}")

    # Save summary
    summary_path = output_dir / f"{name_prefix}summary_by_size_complexity.csv"
    results.summary_by_size_complexity.to_csv(summary_path, index=False)
    output_paths["summary"] = summary_path
    print(f"  Saved: {summary_path}")

    # Save overall summary
    overall_path = output_dir / f"{name_prefix}overall_summary.json"
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
            if include_topk_visualizations:
                plot_topk_analysis(results.trajectory_df, results.step_df, output_dir)
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

    print("\n--- Reverse Recovery Analysis ---")
    print(
        f"Total reverse error steps (taken != optimal decoded): {overall.get('total_reverse_error_steps', 0)}"
    )
    print(
        f"Reverse recovered (taken is optimal GT): {overall.get('total_reverse_recovered', 0)}"
    )
    print(
        f"Reverse recovery rate: {overall.get('overall_reverse_recovery_rate', 0):.4f}"
    )

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

    parser.add_argument(
        "--decode-mode",
        type=str,
        default="top1",
        choices=[
            "top1",
            "topk",
            "topk-sweep",
            "comprehensive",
            "taken-optimal-k-compare",
        ],
        help="Decode strategy: top1, topk, topk-sweep (k=1..N), comprehensive, or taken-optimal-k-compare",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k agent and top-k goal cells used when --decode-mode=topk",
    )

    parser.add_argument(
        "--topk-max",
        type=int,
        default=5,
        help="Maximum k for --decode-mode=topk-sweep (runs k=1..topk-max)",
    )

    parser.add_argument(
        "--compare-k-values",
        type=str,
        default="5,10",
        help="Comma-separated k values for taken-optimal comparison plot (e.g., 5,10)",
    )

    parser.add_argument(
        "--compare-aggregation",
        type=str,
        default="any",
        choices=["any", "weighted", "both"],
        help="Aggregation curves to include in taken-optimal-k-compare",
    )

    parser.add_argument(
        "--topk-aggregation",
        type=str,
        default="weighted",
        choices=["top1", "any", "weighted", "both"],
        help="Aggregation policy for top-k trajectory metrics (use 'both' for sweeps)",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional filename prefix for saved artifacts",
    )

    args = parser.parse_args()

    base_path = Path(args.base_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.decode_mode == "comprehensive":
        run_comprehensive_analysis(
            base_dir=base_path,
            output_dir=output_path,
            layer_key=args.layer_key,
            batch_size=args.batch_size,
            topk=max(1, args.topk),
        )
    elif args.decode_mode == "topk-sweep":
        run_topk_sweep(
            base_dir=base_path,
            output_dir=output_path,
            layer_key=args.layer_key,
            batch_size=args.batch_size,
            topk_max=max(1, args.topk_max),
            aggregation=args.topk_aggregation,
        )
    elif args.decode_mode == "taken-optimal-k-compare":
        k_values = [
            int(v.strip())
            for v in args.compare_k_values.split(",")
            if v.strip().isdigit()
        ]
        run_taken_optimal_k_compare(
            base_dir=base_path,
            output_dir=output_path,
            layer_key=args.layer_key,
            batch_size=args.batch_size,
            k_values=k_values,
            compare_aggregation=args.compare_aggregation,
        )
    elif args.decode_mode == "topk":
        results = process_trajectories_topk(
            base_path,
            layer_key=args.layer_key,
            batch_size=args.batch_size,
            k=max(1, args.topk),
            aggregation=args.topk_aggregation,
        )
        save_results(
            results,
            output_path,
            prefix=args.output_prefix
            if args.output_prefix
            else f"topk_k{max(1, args.topk)}_{args.topk_aggregation}",
            include_topk_visualizations=True,
        )
        print_summary(results)
    else:
        results = process_trajectories(
            base_path,
            layer_key=args.layer_key,
            batch_size=args.batch_size,
        )
        save_results(
            results,
            output_path,
            prefix=args.output_prefix,
            include_topk_visualizations=False,
        )
        print_summary(results)


if __name__ == "__main__":
    main()
