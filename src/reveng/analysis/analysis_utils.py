"""Shared utilities for grid analysis and visualization.

This module contains reusable components for:
- Loading grids and metadata
- Computing optimal actions
- Parsing action distributions from logprobs
- Computing entropy and cross-entropy metrics
"""

import heapq
import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================

ACTION_NAME_TO_ID = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}
ACTION_NAMES_UPPER = set(ACTION_NAME_TO_ID.keys())
LOGPROB_EPS = 1e-12

# Type aliases for clarity
ActionID = int
ActionDist = dict[ActionID, float]
OptimalActionSet = set[ActionID]
GridCoord = tuple[int, int]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GridMetadata:
    """Metadata for a single grid instance."""

    grid_size: int
    complexity: float
    instance_id: int
    policy_metadata: list[list[dict[str, Any]]]


@dataclass
class CellMetrics:
    """Computed metrics for a single grid cell."""

    grid_id: str
    grid_size: int
    complexity: float
    instance_id: int
    x: int
    y: int
    llm_action: int
    num_optimal_actions: int
    entropy_bits: float
    optimal_entropy_bits: float
    cross_entropy_bits: Optional[float]
    jsd: Optional[float]
    optimal_mass: float
    is_action_optimal: int
    action_probs: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        base = {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "instance_id": self.instance_id,
            "x": self.x,
            "y": self.y,
            "llm_action": self.llm_action,
            "num_optimal_actions": self.num_optimal_actions,
            "entropy_bits": self.entropy_bits,
            "optimal_entropy_bits": self.optimal_entropy_bits,
            "cross_entropy_bits": self.cross_entropy_bits,
            "jsd": self.jsd,
            "optimal_mass": self.optimal_mass,
            "is_action_optimal": self.is_action_optimal,
        }
        base.update({f"p_{name}": prob for name, prob in self.action_probs.items()})
        return base


# =============================================================================
# File Parsing Utilities
# =============================================================================


def sanitize_label(value: str) -> str:
    """Sanitize a string for use as a file/directory label."""
    if not value:
        return "model"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def parse_filename(filepath: Path) -> tuple[int, float, int]:
    """Extract grid_size, complexity, and instance_id from filename.

    Expected format: grid_size{N}_complexity{X.XX}_{NNNN}_metadata.json
    """
    name = filepath.stem
    parts = name.split("_")
    try:
        grid_size = int(parts[1].replace("size", ""))
        complexity = float(parts[2].replace("complexity", ""))
        instance_id = int(parts[3])
        return grid_size, complexity, instance_id
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid filename format: {filepath.name}") from e


def grid_id_from_parts(grid_size: int, complexity: float, instance_id: int) -> str:
    """Construct grid ID from components."""
    return f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}"


# =============================================================================
# Optimal Actions Computation
# =============================================================================


def compute_optimal_actions(env: Any) -> list[list[OptimalActionSet]]:
    """Compute the set of optimal actions for each cell in the grid.

    Uses backward Dijkstra from the goal to compute shortest distances,
    then determines which actions reduce the distance to goal.

    Args:
        env: MiniGrid environment instance

    Returns:
        2D list where each cell contains the set of optimal action IDs
    """
    base_env = getattr(env, "unwrapped", env)
    grid = base_env.grid
    goal: GridCoord = tuple(base_env.goal_pos)
    width, height = grid.width, grid.height

    def is_passable(x: int, y: int) -> bool:
        """Check if a position is within bounds and not blocked."""
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        cell = grid.get(x, y)
        return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

    # (dx, dy, action_id): LEFT, RIGHT, UP, DOWN
    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    # Backward Dijkstra from goal
    distances: dict[GridCoord, float] = {goal: 0}
    heap: list[tuple[float, GridCoord]] = [(0, goal)]

    while heap:
        dist, (x, y) = heapq.heappop(heap)
        if dist > distances.get((x, y), float("inf")):
            continue

        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap, (new_dist, (nx, ny)))

    # Determine optimal actions for each cell
    optimal_actions: list[list[OptimalActionSet]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue

            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue

            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    neighbor_dist = distances.get((nx, ny), float("inf"))
                    if neighbor_dist == current_dist - 1:
                        optimal_actions[y][x].add(action)

    # Goal cell has no optimal actions (already at goal)
    gx, gy = goal
    optimal_actions[gy][gx] = set()

    return optimal_actions


def compute_optimal_distribution(optimal_actions: OptimalActionSet) -> ActionDist:
    """Compute uniform distribution over optimal actions.

    Args:
        optimal_actions: Set of optimal action IDs

    Returns:
        Dictionary mapping action IDs to probabilities (uniform over optimal actions)
    """
    if not optimal_actions:
        return {aid: 0.0 for aid in ACTION_ID_TO_NAME}

    prob = 1.0 / len(optimal_actions)
    return {aid: prob if aid in optimal_actions else 0.0 for aid in ACTION_ID_TO_NAME}


# =============================================================================
# Logprob Parsing
# =============================================================================


def normalize_action_token(token: Optional[str]) -> str:
    """Normalize an action token string to uppercase."""
    if token is None:
        return ""
    return token.strip().strip('"').strip("'").upper()


def is_action_token(token: str) -> bool:
    """Check if a token is a valid action name."""
    normalized = normalize_action_token(token)
    return normalized in ACTION_NAMES_UPPER


def find_action_token_entry(logprobs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Find the token entry corresponding to the action value in logprobs."""
    if not logprobs:
        return None

    n = len(logprobs)

    # Strategy 1: Structured search for "action": "<VALUE>" pattern
    for i, entry in enumerate(logprobs):
        token = entry.get("token", "")

        if "action" not in token.lower():
            continue

        # Look for colon after the "action" key
        colon_idx = None
        for j in range(i + 1, min(i + 5, n)):
            if ":" in logprobs[j].get("token", ""):
                colon_idx = j
                break

        if colon_idx is None:
            continue

        # Search for action value after colon
        for k in range(colon_idx + 1, min(colon_idx + 5, n)):
            candidate_token = logprobs[k].get("token", "")

            if candidate_token.strip() in ("", '"', "'"):
                continue

            if is_action_token(candidate_token):
                return logprobs[k]

            if "}" in candidate_token:
                break

    # Strategy 2: Fallback - find any high-confidence standalone action token
    for entry in logprobs:
        token = entry.get("token", "")
        if is_action_token(token):
            logprob = entry.get("logprob")
            if logprob is not None and logprob > -1.0:
                return entry

    return None


def distribution_from_logprobs(
    logprobs: Optional[list[dict[str, Any]]],
) -> Optional[ActionDist]:
    """Extract action probability distribution from logprobs.

    Args:
        logprobs: List of token logprob entries from the LLM

    Returns:
        Dictionary mapping action IDs to probabilities, or None if parsing fails
    """
    if not logprobs:
        return None

    token_entry = find_action_token_entry(logprobs)
    if not token_entry:
        return None

    entries: dict[ActionID, float] = {}

    def register_action(token_value: Optional[str], logprob: Optional[float]) -> None:
        action = normalize_action_token(token_value)
        if action in ACTION_NAME_TO_ID and logprob is not None:
            action_id = ACTION_NAME_TO_ID[action]
            entries[action_id] = max(entries.get(action_id, -math.inf), logprob)

    register_action(token_entry.get("token"), token_entry.get("logprob"))

    for candidate in token_entry.get("top_logprobs") or []:
        register_action(candidate.get("token"), candidate.get("logprob"))

    if not entries:
        return None

    # Convert logprobs to probabilities using numerically stable softmax
    max_logprob = max(entries.values())
    probs = {aid: math.exp(lp - max_logprob) for aid, lp in entries.items()}

    total = sum(probs.values())
    if total <= 0:
        return None

    return {aid: probs.get(aid, 0.0) / total for aid in ACTION_ID_TO_NAME}


# =============================================================================
# Entropy and Information Theory Metrics
# =============================================================================


def shannon_entropy(dist: ActionDist) -> float:
    """Compute Shannon entropy of a probability distribution.

    H(X) = -sum_i p_i * log2(p_i)

    Args:
        dist: Dictionary mapping action IDs to probabilities

    Returns:
        Entropy in bits
    """
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def cross_entropy(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Compute cross-entropy between uniform optimal and model distributions.

    H(p_opt, q_model) = -sum_a p_opt(a) * log2(q_model(a))

    Args:
        optimal_actions: Set of optimal action IDs
        dist: Model's action probability distribution
        eps: Small epsilon to avoid log(0)

    Returns:
        Cross-entropy in bits, or None if no optimal actions
    """
    if not optimal_actions:
        return None

    weight = 1.0 / len(optimal_actions)
    ce = 0.0
    for action in optimal_actions:
        model_prob = max(dist.get(action, 0.0), eps)
        ce -= weight * math.log2(model_prob)
    return ce


def compute_optimal_mass(optimal_actions: OptimalActionSet, dist: ActionDist) -> float:
    """Compute total probability mass assigned to optimal actions."""
    return sum(dist.get(action, 0.0) for action in optimal_actions)


def optimal_entropy(num_optimal_actions: int) -> float:
    """Entropy of uniform distribution over optimal actions: H = log2(k)."""
    if num_optimal_actions <= 0:
        return 0.0
    return math.log2(num_optimal_actions)


def kl_divergence(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Compute KL divergence from optimal to model distribution.

    D_KL(optimal || model) = H(optimal, model) - H(optimal)

    This measures the "excess" bits needed beyond the optimal entropy.
    If the model perfectly matches optimal, KL = 0.

    Args:
        optimal_actions: Set of optimal action IDs
        dist: Model's action probability distribution
        eps: Small epsilon to avoid log(0)

    Returns:
        KL divergence in bits, or None if no optimal actions
    """
    ce = cross_entropy(optimal_actions, dist, eps)
    if ce is None:
        return None
    h_opt = optimal_entropy(len(optimal_actions))
    return ce - h_opt


def jensen_shannon_divergence(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Compute Jensen-Shannon divergence between optimal and model distributions.

    JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)

    JSD is symmetric and bounded in [0, 1] when using log base 2.
    JSD = 0 means identical distributions, JSD = 1 means completely different.

    Args:
        optimal_actions: Set of optimal action IDs
        dist: Model's action probability distribution
        eps: Small epsilon for numerical stability

    Returns:
        JSD in [0, 1], or None if no optimal actions
    """
    if not optimal_actions:
        return None

    # Compute uniform optimal distribution
    num_optimal = len(optimal_actions)
    p_opt = 1.0 / num_optimal

    # Compute mixture distribution M = 0.5 * (P + Q)
    # and KL divergences
    kl_p_m = 0.0  # D_KL(optimal || M)
    kl_q_m = 0.0  # D_KL(model || M)

    for action_id in ACTION_ID_TO_NAME:
        # Optimal distribution: uniform over optimal actions
        p = p_opt if action_id in optimal_actions else 0.0
        # Model distribution
        q = dist.get(action_id, 0.0)
        # Mixture
        m = 0.5 * (p + q)

        # Add to KL(P || M)
        if p > 0 and m > eps:
            kl_p_m += p * math.log2(p / m)

        # Add to KL(Q || M)
        if q > 0 and m > eps:
            kl_q_m += q * math.log2(q / m)

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
    return jsd


# =============================================================================
# Data Loading
# =============================================================================


def load_environments(dataset_path: str) -> dict[str, Any]:
    """Load grid environments from pickle file."""
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def discover_metadata_files(metadata_dir: Path) -> list[Path]:
    """Find all metadata JSON files in a directory."""
    return sorted(metadata_dir.glob("*_metadata.json"))


def _load_single_metadata(fpath: Path) -> Optional[tuple[str, GridMetadata]]:
    """Load a single metadata file."""
    try:
        grid_size, complexity, instance_id = parse_filename(fpath)
        key = grid_id_from_parts(grid_size, complexity, instance_id)

        with open(fpath, "r") as f:
            policy_metadata = json.load(f)

        return key, GridMetadata(
            grid_size=grid_size,
            complexity=complexity,
            instance_id=instance_id,
            policy_metadata=policy_metadata,
        )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Skipping invalid file {fpath.name}: {e}")
        return None


def load_metadata_batch(
    metadata_files: list[Path], max_workers: int = 8, show_progress: bool = False
) -> dict[str, GridMetadata]:
    """Load multiple metadata files in parallel."""
    metadata_dict: dict[str, GridMetadata] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_single_metadata, fpath): fpath
            for fpath in metadata_files
        }
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(
                completed, total=len(futures), desc="Loading metadata", leave=False
            )
        for future in completed:
            result = future.result()
            if result is not None:
                key, metadata = result
                metadata_dict[key] = metadata

    return metadata_dict


def load_single_grid_metadata(
    grid_id: str, metadata_dir: Path
) -> Optional[GridMetadata]:
    """Load metadata for a specific grid by ID.

    Args:
        grid_id: Grid identifier (e.g., "grid_size5_complexity0.30_0001")
        metadata_dir: Directory containing metadata files

    Returns:
        GridMetadata instance or None if not found
    """
    metadata_file = metadata_dir / f"{grid_id}_metadata.json"
    if not metadata_file.exists():
        return None

    result = _load_single_metadata(metadata_file)
    return result[1] if result else None


# =============================================================================
# Cell and Grid Processing
# =============================================================================


def process_cell(
    cell: Any,
    optimal_set: OptimalActionSet,
    grid_id: str,
    metadata: GridMetadata,
    x: int,
    y: int,
) -> Optional[CellMetrics]:
    """Process a single cell and compute its metrics."""
    if not isinstance(cell, dict):
        return None

    if not optimal_set:
        return None

    dist = distribution_from_logprobs(cell.get("logprobs"))
    if dist is None:
        return None

    num_optimal = len(optimal_set)
    entropy_bits = shannon_entropy(dist)
    optimal_entropy_bits = optimal_entropy(num_optimal)
    cross_entropy_bits = cross_entropy(optimal_set, dist)
    jsd_value = jensen_shannon_divergence(optimal_set, dist)
    optimal_mass_val = compute_optimal_mass(optimal_set, dist)
    llm_action = cell.get("llm_response", -1)

    action_probs = {
        ACTION_ID_TO_NAME[aid].lower(): dist.get(aid, 0.0) for aid in ACTION_ID_TO_NAME
    }

    return CellMetrics(
        grid_id=grid_id,
        grid_size=metadata.grid_size,
        complexity=metadata.complexity,
        instance_id=metadata.instance_id,
        x=x,
        y=y,
        llm_action=llm_action,
        num_optimal_actions=num_optimal,
        entropy_bits=entropy_bits,
        optimal_entropy_bits=optimal_entropy_bits,
        cross_entropy_bits=cross_entropy_bits,
        jsd=jsd_value,
        optimal_mass=optimal_mass_val,
        is_action_optimal=int(llm_action in optimal_set),
        action_probs=action_probs,
    )


def process_grid(grid_id: str, env: Any, metadata: GridMetadata) -> list[CellMetrics]:
    """Process all cells in a grid and return their metrics."""
    results: list[CellMetrics] = []
    optimal_actions = compute_optimal_actions(env)

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            cell_result = process_cell(
                cell=cell,
                optimal_set=optimal_actions[y][x],
                grid_id=grid_id,
                metadata=metadata,
                x=x,
                y=y,
            )
            if cell_result:
                results.append(cell_result)

    return results


def compute_grid_mean_cross_entropy(
    grid_id: str, env: Any, metadata: GridMetadata
) -> Optional[float]:
    """Compute mean cross-entropy across all cells in a grid.

    Args:
        grid_id: Grid identifier
        env: MiniGrid environment instance
        metadata: Grid metadata with policy information

    Returns:
        Mean cross-entropy in bits, or None if no valid cells
    """
    cell_metrics = process_grid(grid_id, env, metadata)
    valid_ce = [
        m.cross_entropy_bits for m in cell_metrics if m.cross_entropy_bits is not None
    ]

    if not valid_ce:
        return None

    return sum(valid_ce) / len(valid_ce)


def compute_grid_mean_jsd(
    grid_id: str, env: Any, metadata: GridMetadata
) -> Optional[float]:
    """Compute mean Jensen-Shannon divergence across all cells in a grid.

    JSD is bounded [0, 1] and symmetric, making it easier to compare across
    grids with different numbers of optimal actions per cell.

    Args:
        grid_id: Grid identifier
        env: MiniGrid environment instance
        metadata: Grid metadata with policy information

    Returns:
        Mean JSD in [0, 1], or None if no valid cells
    """
    optimal_actions_grid = compute_optimal_actions(env)
    jsd_values: list[float] = []

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue

            optimal_set = optimal_actions_grid[y][x]
            if not optimal_set:
                continue

            dist = distribution_from_logprobs(cell.get("logprobs"))
            if dist is None:
                continue

            jsd = jensen_shannon_divergence(optimal_set, dist)
            if jsd is not None:
                jsd_values.append(jsd)

    if not jsd_values:
        return None

    return sum(jsd_values) / len(jsd_values)
