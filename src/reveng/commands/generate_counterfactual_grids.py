"""Generate counterfactual grids from stored trajectory step-0 observations."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from reveng.analysis.analysis_utils import (
    ACTION_ID_TO_NAME,
    OptimalActionSet,
    compute_optimal_actions_from_text_grid,
)
from reveng.commands.get_trajectory.get_trajectory_fn import get_trajectory
from reveng.commands.get_trajectory.get_trajectory_utils import (
    upload_files_to_huggingface,
)
from reveng.commands.get_trajectory.rate_limiter import RateLimiter
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)

logger = logging.getLogger(__file__)

AGENT_MOVED = "agent_moved"
GOAL_MOVED = "goal_moved"
RELATION_DISJOINT = "disjoint"
RELATION_OVERLAP = "overlap"
RELATION_SAME = "same"
RELATION_ORDER = (RELATION_DISJOINT, RELATION_SAME, RELATION_OVERLAP)
COUNTERFACTUAL_TYPES = (AGENT_MOVED, GOAL_MOVED)
POSITION_RE = re.compile(
    r"_size(?P<size>\d+)_comp(?P<complexity>\d+\.\d+)_(?P<id>\d+)\.json$"
)


@dataclass(frozen=True)
class SourceGrid:
    """Parsed step-0 grid and metadata from a trajectory file."""

    source_path: Path
    actual_instance_id: int
    source_step_id: int
    original_grid: list[list[str]]
    agent_position: tuple[int, int]
    goal_position: tuple[int, int]
    original_optimal_action_set: OptimalActionSet
    original_distance_to_goal: int


@dataclass(frozen=True)
class CounterfactualTrajectoryWorkItem:
    """One counterfactual grid to run through `get_trajectory` for a single step."""

    source_counterfactual_path: Path
    grid_size: int
    grid_complexity: float
    requested_instance_id: int
    actual_instance_id: int
    source_filename: str
    counterfactual_type: Literal["agent_moved", "goal_moved"]
    counterfactual_id: str
    relation_to_original: str
    grid: list[list[str]]


def parse_grid_state_to_layout(grid_state: list[str]) -> list[list[str]]:
    """Parse trajectory `grid_state` lines into a plain grid layout."""
    if len(grid_state) < 2:
        raise ValueError(
            "grid_state must include a header row and at least one data row"
        )

    grid_layout: list[list[str]] = []
    expected_width: int | None = None

    for row in grid_state[1:]:
        parts = row.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed grid_state row: {row!r}")

        cells = parts[1:]
        if expected_width is None:
            expected_width = len(cells)
        elif len(cells) != expected_width:
            raise ValueError("Inconsistent row width while parsing grid_state")

        grid_layout.append(cells)

    if not grid_layout:
        raise ValueError("No grid rows found in grid_state")

    return grid_layout


def extract_agent_and_goal_positions(
    grid_layout: list[list[str]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Extract the unique agent and goal coordinates from a parsed grid."""
    agent_positions: list[tuple[int, int]] = []
    goal_positions: list[tuple[int, int]] = []

    for y, row in enumerate(grid_layout):
        for x, cell in enumerate(row):
            if cell == "A":
                agent_positions.append((x, y))
            elif cell == "G":
                goal_positions.append((x, y))

    if len(agent_positions) != 1:
        raise ValueError(
            f"Expected exactly one agent in grid, found {len(agent_positions)}"
        )
    if len(goal_positions) != 1:
        raise ValueError(
            f"Expected exactly one goal in grid, found {len(goal_positions)}"
        )

    return agent_positions[0], goal_positions[0]


def classify_relation(
    original_optimal_action_set: OptimalActionSet,
    counterfactual_optimal_action_set: OptimalActionSet,
) -> str:
    """Classify a counterfactual optimal-action set against the original one."""
    if counterfactual_optimal_action_set == original_optimal_action_set:
        return RELATION_SAME
    if counterfactual_optimal_action_set.isdisjoint(original_optimal_action_set):
        return RELATION_DISJOINT
    return RELATION_OVERLAP


def parse_instance_id_from_filename(path: Path) -> int | None:
    """Extract the integer instance id from a trajectory filename."""
    match = POSITION_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("id"))


def parse_size_and_complexity_from_filename(path: Path) -> tuple[int, float] | None:
    """Extract `(size, complexity)` from a trajectory filename."""
    match = POSITION_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("size")), float(match.group("complexity"))


def sanitize_model_name(model_name: str) -> str:
    """Sanitize a model name for filenames."""
    return model_name.replace("/", "_").replace(".", "_")


def build_counterfactual_trajectory_output_path(
    output_root: Path,
    model_name: str,
    work_item: CounterfactualTrajectoryWorkItem,
) -> Path:
    """Build the output path for one counterfactual trajectory file."""
    model_sanitized = sanitize_model_name(model_name)
    return (
        output_root
        / f"size{work_item.grid_size}"
        / (
            f"{model_sanitized}_size{work_item.grid_size}_"
            f"comp{work_item.grid_complexity:.1f}_"
            f"requested{work_item.requested_instance_id}_"
            f"source{work_item.actual_instance_id}_"
            f"{work_item.counterfactual_type}_{work_item.counterfactual_id}.json"
        )
    )


def counterfactual_grid_to_env(
    grid_layout: list[list[str]],
) -> FullObservabilityTextWrapper:
    """Reconstruct a wrapped env from a stored counterfactual grid layout."""
    if not grid_layout or not grid_layout[0]:
        raise ValueError("grid layout must be non-empty")

    size = max(len(grid_layout), len(grid_layout[0]))
    env = Simple2DNavigationEnv(size=size, render_mode=None)
    env.set_env_from_list(grid_layout)
    return FullObservabilityTextWrapper(env)


def load_counterfactual_summary(
    summary_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Load the batch summary and return successful counterfactual JSON paths."""
    if not summary_path.exists():
        raise ValueError(f"Counterfactual summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text())
    successful_paths = [
        Path(entry["output_path"])
        for entry in summary.get("successes", [])
        if "output_path" in entry
    ]
    return summary, successful_paths


def expand_counterfactual_trajectory_work_items(
    source_counterfactual_path: Path,
    counterfactual_payload: dict[str, Any],
) -> list[CounterfactualTrajectoryWorkItem]:
    """Expand one counterfactual JSON file into per-grid work items."""
    required_top_level_fields = [
        "grid_size",
        "grid_complexity",
        "requested_instance_id",
        "actual_instance_id",
        "source_filename",
        "agent_moved_counterfactuals",
        "goal_moved_counterfactuals",
    ]
    for field in required_top_level_fields:
        if field not in counterfactual_payload:
            raise ValueError(
                f"Counterfactual file {source_counterfactual_path} is missing field: {field}"
            )

    work_items: list[CounterfactualTrajectoryWorkItem] = []
    family_to_key = {
        AGENT_MOVED: "agent_moved_counterfactuals",
        GOAL_MOVED: "goal_moved_counterfactuals",
    }

    for counterfactual_type, payload_key in family_to_key.items():
        entries = counterfactual_payload[payload_key]
        if not isinstance(entries, list):
            raise ValueError(
                f"{payload_key} must be a list in {source_counterfactual_path}"
            )

        for entry in entries:
            required_entry_fields = [
                "counterfactual_id",
                "counterfactual_type",
                "relation_to_original",
                "grid",
            ]
            for field in required_entry_fields:
                if field not in entry:
                    raise ValueError(
                        f"{payload_key} entry in {source_counterfactual_path} is missing field: {field}"
                    )

            work_items.append(
                CounterfactualTrajectoryWorkItem(
                    source_counterfactual_path=source_counterfactual_path,
                    grid_size=int(counterfactual_payload["grid_size"]),
                    grid_complexity=float(counterfactual_payload["grid_complexity"]),
                    requested_instance_id=int(
                        counterfactual_payload["requested_instance_id"]
                    ),
                    actual_instance_id=int(
                        counterfactual_payload["actual_instance_id"]
                    ),
                    source_filename=str(counterfactual_payload["source_filename"]),
                    counterfactual_type=entry["counterfactual_type"],
                    counterfactual_id=str(entry["counterfactual_id"]),
                    relation_to_original=str(entry["relation_to_original"]),
                    grid=entry["grid"],
                )
            )

    return work_items


def discover_size_complexity_pairs(
    trajectory_root: Path,
) -> list[tuple[int, float]]:
    """Discover all available `(grid_size, grid_complexity)` pairs under a root."""
    if not trajectory_root.exists():
        raise ValueError(f"Trajectory directory does not exist: {trajectory_root}")

    pairs = {
        parsed
        for path in trajectory_root.glob("size*/*.json")
        if (parsed := parse_size_and_complexity_from_filename(path)) is not None
    }
    if not pairs:
        raise ValueError(f"No trajectory JSONs found under: {trajectory_root}")

    return sorted(pairs, key=lambda item: (item[0], item[1]))


def discover_candidate_files(
    trajectory_root: Path,
    grid_size: int,
    grid_complexity: float,
    instance_id: int,
) -> tuple[list[Path], bool]:
    """Find candidate trajectory files and order them for evaluation."""
    size_dir = trajectory_root / f"size{grid_size}"
    if not size_dir.exists():
        raise ValueError(f"Trajectory directory does not exist: {size_dir}")

    fragment = f"_size{grid_size}_comp{grid_complexity:.1f}_"
    all_candidates = sorted(
        path
        for path in size_dir.glob("*.json")
        if fragment in path.name and parse_instance_id_from_filename(path) is not None
    )

    if not all_candidates:
        raise ValueError(
            f"No trajectory JSONs found for size={grid_size}, complexity={grid_complexity:.1f}"
        )

    requested = [
        path
        for path in all_candidates
        if parse_instance_id_from_filename(path) == instance_id
    ]
    others = [path for path in all_candidates if path not in requested]

    return requested + others, bool(requested)


def load_source_grid(path: Path) -> SourceGrid:
    """Load the original step-0 grid and optimal actions from a trajectory file."""
    data = json.loads(path.read_text())
    steps = data.get("steps", [])
    if not steps:
        raise ValueError("trajectory JSON has no steps")

    first_step = steps[0]
    source_step_id = first_step.get("step_id")
    if source_step_id != 0:
        raise ValueError(f"expected first step_id to be 0, found {source_step_id}")

    grid_state = first_step.get("grid_state")
    if not isinstance(grid_state, list):
        raise ValueError("steps[0].grid_state must be a list of strings")

    original_grid = parse_grid_state_to_layout(grid_state)
    agent_position, goal_position = extract_agent_and_goal_positions(original_grid)
    optimal_actions, distances = compute_optimal_actions_from_text_grid(
        original_grid, goal_position
    )
    original_optimal_action_set = optimal_actions.get(agent_position)
    original_distance = distances.get(agent_position)

    if original_optimal_action_set is None or original_distance is None:
        raise ValueError("original grid is not solvable from the agent position")

    actual_instance_id = parse_instance_id_from_filename(path)
    if actual_instance_id is None:
        raise ValueError(f"could not parse instance_id from filename: {path.name}")

    return SourceGrid(
        source_path=path,
        actual_instance_id=actual_instance_id,
        source_step_id=source_step_id,
        original_grid=original_grid,
        agent_position=agent_position,
        goal_position=goal_position,
        original_optimal_action_set=set(original_optimal_action_set),
        original_distance_to_goal=original_distance,
    )


def action_set_to_names(action_set: OptimalActionSet) -> list[str]:
    """Serialize action ids into stable ordered action names."""
    return [ACTION_ID_TO_NAME[action_id] for action_id in sorted(action_set)]


def position_to_list(position: tuple[int, int]) -> list[int]:
    """Convert a coordinate tuple to a JSON-friendly list."""
    return [position[0], position[1]]


def clone_grid(grid_layout: list[list[str]]) -> list[list[str]]:
    """Return a shallow clone of a 2D grid layout."""
    return [row[:] for row in grid_layout]


def move_entity(
    grid_layout: list[list[str]],
    from_position: tuple[int, int],
    to_position: tuple[int, int],
    entity_symbol: str,
) -> list[list[str]]:
    """Move the agent or goal to a new position in a copied grid."""
    moved_grid = clone_grid(grid_layout)
    moved_grid[from_position[1]][from_position[0]] = "_"
    moved_grid[to_position[1]][to_position[0]] = entity_symbol
    return moved_grid


def enumerate_counterfactual_candidates(
    source_grid: SourceGrid,
    counterfactual_type: str,
) -> list[dict[str, Any]]:
    """Enumerate valid counterfactuals in deterministic row-major order."""
    if counterfactual_type not in COUNTERFACTUAL_TYPES:
        raise ValueError(f"Unsupported counterfactual_type: {counterfactual_type}")

    candidates: list[dict[str, Any]] = []

    for y, row in enumerate(source_grid.original_grid):
        for x, cell in enumerate(row):
            target_position = (x, y)
            if cell == "#":
                continue

            if counterfactual_type == AGENT_MOVED:
                if target_position in (
                    source_grid.agent_position,
                    source_grid.goal_position,
                ):
                    continue
                moved_grid = move_entity(
                    source_grid.original_grid,
                    source_grid.agent_position,
                    target_position,
                    "A",
                )
                agent_position = target_position
                goal_position = source_grid.goal_position
                moved_entity = "agent"
                from_position = source_grid.agent_position
            else:
                if target_position in (
                    source_grid.goal_position,
                    source_grid.agent_position,
                ):
                    continue
                moved_grid = move_entity(
                    source_grid.original_grid,
                    source_grid.goal_position,
                    target_position,
                    "G",
                )
                agent_position = source_grid.agent_position
                goal_position = target_position
                moved_entity = "goal"
                from_position = source_grid.goal_position

            optimal_actions, distances = compute_optimal_actions_from_text_grid(
                moved_grid, goal_position
            )
            counterfactual_optimal_action_set = optimal_actions.get(agent_position)
            distance_to_goal = distances.get(agent_position)

            if counterfactual_optimal_action_set is None or distance_to_goal is None:
                continue

            relation = classify_relation(
                source_grid.original_optimal_action_set,
                counterfactual_optimal_action_set,
            )

            candidates.append(
                {
                    "counterfactual_type": counterfactual_type,
                    "grid": moved_grid,
                    "agent_position": position_to_list(agent_position),
                    "goal_position": position_to_list(goal_position),
                    "moved_entity": moved_entity,
                    "from_position": position_to_list(from_position),
                    "to_position": position_to_list(target_position),
                    "optimal_action_set_ids": sorted(counterfactual_optimal_action_set),
                    "optimal_action_set_names": action_set_to_names(
                        counterfactual_optimal_action_set
                    ),
                    "relation_to_original": relation,
                    "distance_to_goal": distance_to_goal,
                }
            )

    return candidates


def summarize_candidate_relations(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    """Count relation labels for a counterfactual family."""
    summary = {relation: 0 for relation in RELATION_ORDER}
    for candidate in candidates:
        summary[candidate["relation_to_original"]] += 1
    return summary


def select_counterfactuals(
    candidates: list[dict[str, Any]],
    counterfactual_type: str,
    num_disjoint: int,
    num_same: int,
) -> list[dict[str, Any]] | None:
    """Pick the required disjoint and same candidates for one family."""
    disjoint = [
        candidate
        for candidate in candidates
        if candidate["relation_to_original"] == RELATION_DISJOINT
    ]
    same = [
        candidate
        for candidate in candidates
        if candidate["relation_to_original"] == RELATION_SAME
    ]

    if len(disjoint) < num_disjoint or len(same) < num_same:
        return None

    selected = disjoint[:num_disjoint] + same[:num_same]
    finalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        entry = dict(candidate)
        entry["counterfactual_id"] = f"{counterfactual_type}_{index:02d}"
        finalized.append(entry)
    return finalized


def evaluate_source_file(
    source_grid: SourceGrid,
    num_disjoint: int,
    num_same: int,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, int]]]:
    """Evaluate whether a source file can satisfy both counterfactual families."""
    agent_candidates = enumerate_counterfactual_candidates(source_grid, AGENT_MOVED)
    goal_candidates = enumerate_counterfactual_candidates(source_grid, GOAL_MOVED)

    counts = {
        AGENT_MOVED: summarize_candidate_relations(agent_candidates),
        GOAL_MOVED: summarize_candidate_relations(goal_candidates),
    }

    selected_agent = select_counterfactuals(
        agent_candidates,
        AGENT_MOVED,
        num_disjoint=num_disjoint,
        num_same=num_same,
    )
    selected_goal = select_counterfactuals(
        goal_candidates,
        GOAL_MOVED,
        num_disjoint=num_disjoint,
        num_same=num_same,
    )

    if selected_agent is None or selected_goal is None:
        return None, counts

    return {
        AGENT_MOVED: selected_agent,
        GOAL_MOVED: selected_goal,
    }, counts


def format_failure_message(
    failure_details: list[dict[str, Any]],
    grid_size: int,
    grid_complexity: float,
    instance_id: int,
) -> str:
    """Format a readable error message when no source file satisfies the constraints."""
    lines = [
        (
            "Unable to satisfy counterfactual selection requirements for "
            f"size={grid_size}, complexity={grid_complexity:.1f}, requested_instance_id={instance_id}."
        ),
        "Evaluated sources:",
    ]

    for detail in failure_details:
        if "error" in detail:
            lines.append(f"- {detail['source']}: error={detail['error']}")
            continue

        agent_counts = detail[AGENT_MOVED]
        goal_counts = detail[GOAL_MOVED]
        lines.append(
            "- "
            f"{detail['source']}: "
            f"agent(disjoint={agent_counts[RELATION_DISJOINT]}, same={agent_counts[RELATION_SAME]}, overlap={agent_counts[RELATION_OVERLAP]}), "
            f"goal(disjoint={goal_counts[RELATION_DISJOINT]}, same={goal_counts[RELATION_SAME]}, overlap={goal_counts[RELATION_OVERLAP]})"
        )

    return "\n".join(lines)


def build_output_payload(
    source_grid: SourceGrid,
    selected_counterfactuals: dict[str, list[dict[str, Any]]],
    relation_counts: dict[str, dict[str, int]],
    requested_instance_id: int,
    grid_size: int,
    grid_complexity: float,
    trajectory_root: Path,
    fallback_used: bool,
    num_counterfactuals: int,
    num_disjoint: int,
    num_same: int,
) -> dict[str, Any]:
    """Build the final JSON payload for the selected source file."""
    try:
        relative_source_path = str(source_grid.source_path.relative_to(Path.cwd()))
    except ValueError:
        relative_source_path = str(source_grid.source_path)

    return {
        "grid_size": grid_size,
        "grid_complexity": grid_complexity,
        "requested_instance_id": requested_instance_id,
        "actual_instance_id": source_grid.actual_instance_id,
        "source_filename": source_grid.source_path.name,
        "source_relative_path": relative_source_path,
        "source_step_id": source_grid.source_step_id,
        "fallback_used": fallback_used,
        "trajectory_root": str(trajectory_root),
        "selection_requirements": {
            "num_counterfactuals": num_counterfactuals,
            "num_disjoint": num_disjoint,
            "num_same": num_same,
        },
        "original_agent_position": position_to_list(source_grid.agent_position),
        "original_goal_position": position_to_list(source_grid.goal_position),
        "original_distance_to_goal": source_grid.original_distance_to_goal,
        "original_optimal_action_set_ids": sorted(
            source_grid.original_optimal_action_set
        ),
        "original_optimal_action_set_names": action_set_to_names(
            source_grid.original_optimal_action_set
        ),
        "candidate_relation_counts": relation_counts,
        "original_grid": source_grid.original_grid,
        "agent_moved_counterfactuals": selected_counterfactuals[AGENT_MOVED],
        "goal_moved_counterfactuals": selected_counterfactuals[GOAL_MOVED],
    }


def generate_counterfactual_grids(
    grid_size: int,
    grid_complexity: float,
    instance_id: int,
    trajectory_root: str = "data/trajectories_test_full",
    output_dir: str = "data/counterfactual_grids",
    num_counterfactuals: int = 10,
    num_disjoint: int = 8,
    num_same: int = 2,
    fallback_search: bool = True,
) -> str:
    """Generate and save counterfactual grids for one size/complexity bucket."""
    if num_counterfactuals <= 0:
        raise ValueError("num_counterfactuals must be positive")
    if num_disjoint < 0 or num_same < 0:
        raise ValueError("num_disjoint and num_same must be non-negative")
    if num_disjoint + num_same != num_counterfactuals:
        raise ValueError(
            "num_disjoint + num_same must equal num_counterfactuals for exact selection"
        )

    trajectory_root_path = Path(trajectory_root)
    output_root = Path(output_dir)

    candidate_files, requested_present = discover_candidate_files(
        trajectory_root=trajectory_root_path,
        grid_size=grid_size,
        grid_complexity=grid_complexity,
        instance_id=instance_id,
    )

    if not requested_present and not fallback_search:
        raise ValueError(
            f"Requested instance_id={instance_id} was not found for size={grid_size}, "
            f"complexity={grid_complexity:.1f}"
        )

    if not fallback_search:
        candidate_files = candidate_files[:1]

    failure_details: list[dict[str, Any]] = []

    for candidate_path in candidate_files:
        try:
            source_grid = load_source_grid(candidate_path)
            selected_counterfactuals, relation_counts = evaluate_source_file(
                source_grid=source_grid,
                num_disjoint=num_disjoint,
                num_same=num_same,
            )
        except ValueError as exc:
            logger.warning("Skipping candidate %s: %s", candidate_path.name, exc)
            failure_details.append({"source": candidate_path.name, "error": str(exc)})
            continue

        if selected_counterfactuals is None:
            logger.warning(
                "Candidate %s did not satisfy selection requirements: "
                "agent(disjoint=%s, same=%s, overlap=%s), "
                "goal(disjoint=%s, same=%s, overlap=%s)",
                candidate_path.name,
                relation_counts[AGENT_MOVED][RELATION_DISJOINT],
                relation_counts[AGENT_MOVED][RELATION_SAME],
                relation_counts[AGENT_MOVED][RELATION_OVERLAP],
                relation_counts[GOAL_MOVED][RELATION_DISJOINT],
                relation_counts[GOAL_MOVED][RELATION_SAME],
                relation_counts[GOAL_MOVED][RELATION_OVERLAP],
            )
            failure_details.append(
                {
                    "source": candidate_path.name,
                    AGENT_MOVED: relation_counts[AGENT_MOVED],
                    GOAL_MOVED: relation_counts[GOAL_MOVED],
                }
            )
            continue

        fallback_used = source_grid.actual_instance_id != instance_id
        if fallback_used:
            logger.warning(
                "Falling back from requested instance_id=%s to source instance_id=%s for "
                "size=%s complexity=%.1f (%s)",
                instance_id,
                source_grid.actual_instance_id,
                grid_size,
                grid_complexity,
                source_grid.source_path.name,
            )
        payload = build_output_payload(
            source_grid=source_grid,
            selected_counterfactuals=selected_counterfactuals,
            relation_counts=relation_counts,
            requested_instance_id=instance_id,
            grid_size=grid_size,
            grid_complexity=grid_complexity,
            trajectory_root=trajectory_root_path,
            fallback_used=fallback_used,
            num_counterfactuals=num_counterfactuals,
            num_disjoint=num_disjoint,
            num_same=num_same,
        )

        output_path = (
            output_root
            / f"size{grid_size}"
            / (
                f"counterfactuals_size{grid_size}_comp{grid_complexity:.1f}_"
                f"requested{instance_id}_source{source_grid.actual_instance_id}.json"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")

        logger.info("Saved counterfactual grids to %s", output_path)
        return str(output_path)

    error_message = format_failure_message(
        failure_details=failure_details,
        grid_size=grid_size,
        grid_complexity=grid_complexity,
        instance_id=instance_id,
    )
    logger.error("%s", error_message)
    raise ValueError(error_message)


def generate_counterfactual_grids_all_pairs(
    instance_id: int = 0,
    trajectory_root: str = "data/trajectories_test_full",
    output_dir: str = "data/counterfactual_grids",
    num_counterfactuals: int = 10,
    num_disjoint: int = 8,
    num_same: int = 2,
    fallback_search: bool = True,
    continue_on_error: bool = True,
    summary_file_name: str = "counterfactual_batch_summary.json",
) -> str:
    """Generate counterfactual grids for every discovered size/complexity pair."""
    trajectory_root_path = Path(trajectory_root)
    output_root = Path(output_dir)
    pairs = discover_size_complexity_pairs(trajectory_root_path)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for grid_size, grid_complexity in pairs:
        try:
            output_path = generate_counterfactual_grids(
                grid_size=grid_size,
                grid_complexity=grid_complexity,
                instance_id=instance_id,
                trajectory_root=str(trajectory_root_path),
                output_dir=str(output_root),
                num_counterfactuals=num_counterfactuals,
                num_disjoint=num_disjoint,
                num_same=num_same,
                fallback_search=fallback_search,
            )
        except ValueError as exc:
            failure = {
                "grid_size": grid_size,
                "grid_complexity": grid_complexity,
                "error": str(exc),
            }
            logger.warning(
                "Batch generation failed for size=%s complexity=%.1f: %s",
                grid_size,
                grid_complexity,
                exc,
            )
            failures.append(failure)
            if not continue_on_error:
                raise
            continue

        successes.append(
            {
                "grid_size": grid_size,
                "grid_complexity": grid_complexity,
                "output_path": output_path,
            }
        )

    summary = {
        "trajectory_root": str(trajectory_root_path),
        "output_dir": str(output_root),
        "requested_instance_id": instance_id,
        "pairs_discovered": [
            {"grid_size": grid_size, "grid_complexity": grid_complexity}
            for grid_size, grid_complexity in pairs
        ],
        "success_count": len(successes),
        "failure_count": len(failures),
        "successes": successes,
        "failures": failures,
    }

    summary_path = output_root / summary_file_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Saved batch counterfactual summary to %s", summary_path)

    if failures and not continue_on_error:
        raise ValueError(f"Batch generation failed; see summary at {summary_path}")

    return str(summary_path)


def get_counterfactual_trajectories_all_pairs(
    counterfactual_summary_path: str = "data/counterfactual_grids/counterfactual_batch_summary.json",
    output_dir: str = "data/counterfactual_trajectories",
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    observation_placeholders: list[str] = ["grid_state"],
    verbose: bool = False,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
    max_workers: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    continue_on_error: bool = True,
    summary_file_name: str = "counterfactual_trajectory_batch_summary.json",
) -> str:
    """Generate one-step trajectories for all selected counterfactual grids."""
    summary_path = Path(counterfactual_summary_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    source_summary, source_counterfactual_paths = load_counterfactual_summary(
        summary_path
    )
    source_failures = source_summary.get("failures", [])
    if source_failures:
        logger.warning(
            "Source counterfactual summary contains %s failed size/complexity pairs; "
            "only successful counterfactual files will be processed.",
            len(source_failures),
        )

    work_items: list[CounterfactualTrajectoryWorkItem] = []
    manifest_failures: list[dict[str, Any]] = []

    for source_path in source_counterfactual_paths:
        try:
            payload = json.loads(source_path.read_text())
            work_items.extend(
                expand_counterfactual_trajectory_work_items(source_path, payload)
            )
        except Exception as exc:
            failure = {
                "source_counterfactual_path": str(source_path),
                "output_path": None,
                "model_name": None,
                "grid_size": None,
                "grid_complexity": None,
                "requested_instance_id": None,
                "actual_instance_id": None,
                "counterfactual_type": None,
                "counterfactual_id": None,
                "relation_to_original": None,
                "source_filename": None,
                "error": str(exc),
            }
            logger.warning(
                "Failed to expand counterfactual source %s: %s",
                source_path,
                exc,
            )
            manifest_failures.append(failure)
            if not continue_on_error:
                raise ValueError(
                    f"Failed to expand counterfactual source {source_path}: {exc}"
                )

    tasks = list(product(model_names, work_items))
    logger.info(
        "Generating %s one-step counterfactual trajectories across %s models and %s counterfactual grids.",
        len(tasks),
        len(model_names),
        len(work_items),
    )

    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)
        logger.info(
            "Rate limiting enabled: %s requests per %s seconds (%.2f requests/second)",
            rate_limit,
            rate_limit_period,
            rate_limit / rate_limit_period,
        )

    successful_records: list[dict[str, Any]] = []
    successful_paths: list[str] = []

    def _run_single_task(
        task: tuple[str, CounterfactualTrajectoryWorkItem],
    ) -> dict[str, Any]:
        model_name, work_item = task
        if rate_limiter is not None:
            rate_limiter.acquire()

        output_path = build_counterfactual_trajectory_output_path(
            output_root=output_root,
            model_name=model_name,
            work_item=work_item,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            env = counterfactual_grid_to_env(work_item.grid)
            get_trajectory(
                grid_size=work_item.grid_size,
                grid_complexity=work_item.grid_complexity,
                max_steps_per_trajectory=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_logprobs=top_logprobs,
                seed=seed,
                reasoning_effort=reasoning_effort,
                model_name=model_name,
                observation_placeholders=observation_placeholders,
                output_path=str(output_path),
                verbose=verbose,
                env=env,
                use_safe_reset=True,
                transform_type="base",
            )
            return {
                "status": "success",
                "source_counterfactual_path": str(work_item.source_counterfactual_path),
                "output_path": str(output_path),
                "model_name": model_name,
                "grid_size": work_item.grid_size,
                "grid_complexity": work_item.grid_complexity,
                "requested_instance_id": work_item.requested_instance_id,
                "actual_instance_id": work_item.actual_instance_id,
                "counterfactual_type": work_item.counterfactual_type,
                "counterfactual_id": work_item.counterfactual_id,
                "relation_to_original": work_item.relation_to_original,
                "source_filename": work_item.source_filename,
            }
        except Exception as exc:
            logger.warning(
                "Failed to generate counterfactual trajectory for %s %s %s %s: %s",
                work_item.source_filename,
                work_item.counterfactual_type,
                work_item.counterfactual_id,
                model_name,
                exc,
            )
            return {
                "status": "error",
                "source_counterfactual_path": str(work_item.source_counterfactual_path),
                "output_path": str(output_path),
                "model_name": model_name,
                "grid_size": work_item.grid_size,
                "grid_complexity": work_item.grid_complexity,
                "requested_instance_id": work_item.requested_instance_id,
                "actual_instance_id": work_item.actual_instance_id,
                "counterfactual_type": work_item.counterfactual_type,
                "counterfactual_id": work_item.counterfactual_id,
                "relation_to_original": work_item.relation_to_original,
                "source_filename": work_item.source_filename,
                "error": str(exc),
            }

    if max_workers is None:
        max_workers = min(32, len(tasks)) if tasks else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_task, task): task for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(tasks),
            desc="Generating counterfactual trajectories",
            unit="trajectory",
        ):
            result = future.result()
            if result["status"] == "success":
                successful_records.append(result)
                successful_paths.append(result["output_path"])
            else:
                manifest_failures.append(
                    {k: v for k, v in result.items() if k != "status"}
                )
                if not continue_on_error:
                    raise ValueError(
                        "Failed to generate counterfactual trajectory for "
                        f"{result['source_filename']} {result['counterfactual_id']} "
                        f"({result['model_name']}): {result['error']}"
                    )

    upload_urls_by_path: dict[str, str] = {}
    if hf_repo_id is not None and successful_paths:
        upload_urls = upload_files_to_huggingface(
            file_paths=successful_paths,
            repo_id=hf_repo_id,
            path_prefix=hf_path_prefix,
            hf_token=hf_token,
        )
        upload_urls_by_path = dict(zip(successful_paths, upload_urls, strict=False))

    for record in successful_records:
        if record["output_path"] in upload_urls_by_path:
            record["upload_url"] = upload_urls_by_path[record["output_path"]]

    manifest = {
        "counterfactual_summary_path": str(summary_path),
        "output_dir": str(output_root),
        "generation_settings": {
            "model_names": model_names,
            "max_steps_per_trajectory": 1,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_logprobs": top_logprobs,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "observation_placeholders": observation_placeholders,
            "verbose": verbose,
            "hf_repo_id": hf_repo_id,
            "hf_path_prefix": hf_path_prefix,
            "max_workers": max_workers,
            "enable_rate_limit": enable_rate_limit,
            "rate_limit": rate_limit,
            "rate_limit_period": rate_limit_period,
            "continue_on_error": continue_on_error,
        },
        "pairs_discovered": source_summary.get("pairs_discovered", []),
        "source_counterfactual_files": [
            str(path) for path in source_counterfactual_paths
        ],
        "success_count": len(successful_records),
        "failure_count": len(manifest_failures),
        "successes": successful_records,
        "failures": manifest_failures,
    }

    manifest_path = output_root / summary_file_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Saved counterfactual trajectory batch summary to %s", manifest_path)
    return str(manifest_path)


__all__ = [
    "AGENT_MOVED",
    "CounterfactualTrajectoryWorkItem",
    "GOAL_MOVED",
    "RELATION_DISJOINT",
    "RELATION_OVERLAP",
    "RELATION_SAME",
    "build_counterfactual_trajectory_output_path",
    "classify_relation",
    "counterfactual_grid_to_env",
    "discover_size_complexity_pairs",
    "discover_candidate_files",
    "evaluate_source_file",
    "expand_counterfactual_trajectory_work_items",
    "extract_agent_and_goal_positions",
    "generate_counterfactual_grids",
    "generate_counterfactual_grids_all_pairs",
    "get_counterfactual_trajectories_all_pairs",
    "load_counterfactual_summary",
    "load_source_grid",
    "parse_grid_state_to_layout",
    "parse_size_and_complexity_from_filename",
    "parse_instance_id_from_filename",
    "sanitize_model_name",
]
