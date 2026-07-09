"""Tests for the counterfactual grid generation command."""

import json
from importlib import import_module
from pathlib import Path

import pytest

import reveng.commands.generate_counterfactual_grids as counterfactual_cmds

from reveng.commands.generate_counterfactual_grids import (
    AGENT_MOVED,
    CounterfactualTrajectoryWorkItem,
    GOAL_MOVED,
    RELATION_DISJOINT,
    RELATION_SAME,
    counterfactual_grid_to_env,
    classify_relation,
    discover_size_complexity_pairs,
    expand_counterfactual_trajectory_work_items,
    extract_agent_and_goal_positions,
    generate_counterfactual_grids,
    generate_counterfactual_grids_all_pairs,
    get_counterfactual_trajectories_all_pairs,
    parse_grid_state_to_layout,
)


SUCCESS_GRID = [
    ["#", "#", "#", "#", "#", "#", "#"],
    ["#", "_", "_", "_", "_", "_", "#"],
    ["#", "_", "_", "_", "_", "_", "#"],
    ["#", "_", "_", "_", "_", "_", "#"],
    ["#", "_", "A", "_", "G", "_", "#"],
    ["#", "_", "_", "_", "_", "_", "#"],
    ["#", "#", "#", "#", "#", "#", "#"],
]

FAILING_GRID = [
    ["#", "#", "#", "#", "#", "#", "#"],
    ["#", "_", "_", "_", "_", "_", "#"],
    ["#", "A", "_", "_", "_", "_", "#"],
    ["#", "_", "_", "_", "_", "_", "#"],
    ["#", "_", "_", "_", "G", "_", "#"],
    ["#", "_", "_", "_", "_", "_", "#"],
    ["#", "#", "#", "#", "#", "#", "#"],
]


def make_grid_state(grid_layout: list[list[str]]) -> list[str]:
    """Render a compact grid_state payload for test fixtures."""
    width = len(grid_layout[0])
    lines = ["  " + " ".join(str(i) for i in range(width))]
    for y, row in enumerate(grid_layout):
        lines.append(f"{y} " + " ".join(row))
    return lines


def write_trajectory_file(
    path: Path, grid_layout: list[list[str]], step_id: int = 0
) -> None:
    """Write a minimal trajectory JSON file containing a single step-0 grid."""
    payload = {
        "steps": [
            {
                "step_id": step_id,
                "grid_state": make_grid_state(grid_layout),
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_counterfactual_file(
    path: Path,
    grid_layout: list[list[str]],
    *,
    grid_size: int = 7,
    grid_complexity: float = 0.0,
    requested_instance_id: int = 8,
    actual_instance_id: int = 8,
    source_filename: str = "mock_source.json",
    malformed: bool = False,
) -> None:
    """Write a minimal counterfactual JSON file for trajectory batch tests."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if malformed:
        path.write_text(json.dumps({"grid_size": grid_size}))
        return

    payload = {
        "grid_size": grid_size,
        "grid_complexity": grid_complexity,
        "requested_instance_id": requested_instance_id,
        "actual_instance_id": actual_instance_id,
        "source_filename": source_filename,
        "agent_moved_counterfactuals": [
            {
                "counterfactual_id": "agent_moved_00",
                "counterfactual_type": "agent_moved",
                "relation_to_original": "disjoint",
                "grid": grid_layout,
            }
        ],
        "goal_moved_counterfactuals": [
            {
                "counterfactual_id": "goal_moved_00",
                "counterfactual_type": "goal_moved",
                "relation_to_original": "same",
                "grid": grid_layout,
            }
        ],
    }
    path.write_text(json.dumps(payload))


def write_counterfactual_batch_summary(
    path: Path,
    *,
    successes: list[dict],
    failures: list[dict] | None = None,
) -> None:
    """Write a batch summary JSON for counterfactual trajectory tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pairs_discovered": [
            {
                "grid_size": success["grid_size"],
                "grid_complexity": success["grid_complexity"],
            }
            for success in successes
        ],
        "success_count": len(successes),
        "failure_count": len(failures or []),
        "successes": successes,
        "failures": failures or [],
    }
    path.write_text(json.dumps(payload))


class TestGenerateCounterfactualGrids:
    """Test suite for counterfactual source selection and output generation."""

    def test_parse_grid_state_and_extract_positions(self):
        grid_state = make_grid_state(SUCCESS_GRID)

        parsed = parse_grid_state_to_layout(grid_state)
        agent_position, goal_position = extract_agent_and_goal_positions(parsed)

        assert parsed == SUCCESS_GRID
        assert agent_position == (2, 4)
        assert goal_position == (4, 4)

    def test_classify_relation_labels(self):
        assert classify_relation({1, 3}, {1, 3}) == RELATION_SAME
        assert classify_relation({1, 3}, {0, 2}) == RELATION_DISJOINT
        assert classify_relation({1, 3}, {1, 2}) == "overlap"

    def test_generate_counterfactual_grids_success(self, tmp_path: Path):
        trajectory_root = tmp_path / "trajectories"
        output_dir = tmp_path / "counterfactuals"
        source_file = trajectory_root / "size7" / "mock_model_size7_comp0.0_8.json"
        write_trajectory_file(source_file, SUCCESS_GRID)

        output_path = Path(
            generate_counterfactual_grids(
                grid_size=7,
                grid_complexity=0.0,
                instance_id=8,
                trajectory_root=str(trajectory_root),
                output_dir=str(output_dir),
                fallback_search=False,
            )
        )

        payload = json.loads(output_path.read_text())

        assert output_path.exists()
        assert payload["actual_instance_id"] == 8
        assert payload["source_step_id"] == 0
        assert payload["fallback_used"] is False
        assert payload["original_grid"] == SUCCESS_GRID
        assert payload["original_optimal_action_set_ids"] == [1]
        assert payload["original_optimal_action_set_names"] == ["RIGHT"]
        assert len(payload["agent_moved_counterfactuals"]) == 10
        assert len(payload["goal_moved_counterfactuals"]) == 10

        assert (
            sum(
                1
                for entry in payload["agent_moved_counterfactuals"]
                if entry["relation_to_original"] == RELATION_DISJOINT
            )
            == 8
        )
        assert (
            sum(
                1
                for entry in payload["agent_moved_counterfactuals"]
                if entry["relation_to_original"] == RELATION_SAME
            )
            == 2
        )
        assert (
            sum(
                1
                for entry in payload["goal_moved_counterfactuals"]
                if entry["relation_to_original"] == RELATION_DISJOINT
            )
            == 8
        )
        assert (
            sum(
                1
                for entry in payload["goal_moved_counterfactuals"]
                if entry["relation_to_original"] == RELATION_SAME
            )
            == 2
        )
        assert payload["candidate_relation_counts"][AGENT_MOVED] == {
            "disjoint": 9,
            "same": 2,
            "overlap": 12,
        }
        assert payload["candidate_relation_counts"][GOAL_MOVED] == {
            "disjoint": 9,
            "same": 2,
            "overlap": 12,
        }

    def test_generate_counterfactual_grids_fallback_search(self, tmp_path: Path):
        trajectory_root = tmp_path / "trajectories"
        output_dir = tmp_path / "counterfactuals"
        failing_file = trajectory_root / "size7" / "mock_model_size7_comp0.0_0.json"
        success_file = trajectory_root / "size7" / "mock_model_size7_comp0.0_8.json"
        write_trajectory_file(failing_file, FAILING_GRID)
        write_trajectory_file(success_file, SUCCESS_GRID)

        output_path = Path(
            generate_counterfactual_grids(
                grid_size=7,
                grid_complexity=0.0,
                instance_id=0,
                trajectory_root=str(trajectory_root),
                output_dir=str(output_dir),
                fallback_search=True,
            )
        )

        payload = json.loads(output_path.read_text())

        assert payload["actual_instance_id"] == 8
        assert payload["source_filename"] == success_file.name
        assert payload["fallback_used"] is True

    def test_generate_counterfactual_grids_error_includes_counts(self, tmp_path: Path):
        trajectory_root = tmp_path / "trajectories"
        output_dir = tmp_path / "counterfactuals"
        failing_file = trajectory_root / "size7" / "mock_model_size7_comp0.0_0.json"
        write_trajectory_file(failing_file, FAILING_GRID)

        with pytest.raises(ValueError) as exc_info:
            generate_counterfactual_grids(
                grid_size=7,
                grid_complexity=0.0,
                instance_id=0,
                trajectory_root=str(trajectory_root),
                output_dir=str(output_dir),
                fallback_search=True,
            )

        message = str(exc_info.value)
        assert failing_file.name in message
        assert "agent(disjoint=3, same=8, overlap=12)" in message
        assert "goal(disjoint=1, same=11, overlap=11)" in message

    def test_generate_counterfactual_grids_all_pairs(self, tmp_path: Path):
        trajectory_root = tmp_path / "trajectories"
        output_dir = tmp_path / "counterfactuals"
        write_trajectory_file(
            trajectory_root / "size7" / "mock_model_size7_comp0.0_8.json",
            SUCCESS_GRID,
        )
        write_trajectory_file(
            trajectory_root / "size9" / "mock_model_size9_comp0.2_8.json",
            SUCCESS_GRID,
        )

        pairs = discover_size_complexity_pairs(trajectory_root)
        summary_path = Path(
            generate_counterfactual_grids_all_pairs(
                instance_id=8,
                trajectory_root=str(trajectory_root),
                output_dir=str(output_dir),
            )
        )

        payload = json.loads(summary_path.read_text())

        assert pairs == [(7, 0.0), (9, 0.2)]
        assert summary_path.exists()
        assert payload["success_count"] == 2
        assert payload["failure_count"] == 0
        assert payload["pairs_discovered"] == [
            {"grid_size": 7, "grid_complexity": 0.0},
            {"grid_size": 9, "grid_complexity": 0.2},
        ]
        assert {
            (entry["grid_size"], entry["grid_complexity"])
            for entry in payload["successes"]
        } == {(7, 0.0), (9, 0.2)}

    def test_expand_counterfactual_trajectory_work_items(self, tmp_path: Path):
        counterfactual_path = tmp_path / "counterfactuals.json"
        write_counterfactual_file(counterfactual_path, SUCCESS_GRID)

        payload = json.loads(counterfactual_path.read_text())
        work_items = expand_counterfactual_trajectory_work_items(
            counterfactual_path, payload
        )

        assert len(work_items) == 2
        assert all(
            isinstance(item, CounterfactualTrajectoryWorkItem) for item in work_items
        )
        assert {item.counterfactual_type for item in work_items} == {
            "agent_moved",
            "goal_moved",
        }
        assert {item.counterfactual_id for item in work_items} == {
            "agent_moved_00",
            "goal_moved_00",
        }

    def test_counterfactual_grid_to_env(self):
        env = counterfactual_grid_to_env(SUCCESS_GRID)

        assert isinstance(env, counterfactual_cmds.FullObservabilityTextWrapper)
        assert tuple(env.unwrapped.agent_pos) == (2, 4)
        assert tuple(env.unwrapped.goal_pos) == (4, 4)

    def test_get_counterfactual_trajectories_all_pairs_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        summary_path = tmp_path / "counterfactual_batch_summary.json"
        counterfactual_path = tmp_path / "size7" / "counterfactuals_size7_comp0.0.json"
        output_dir = tmp_path / "counterfactual_trajectories"
        write_counterfactual_file(
            counterfactual_path,
            SUCCESS_GRID,
            grid_size=7,
            grid_complexity=0.0,
            requested_instance_id=8,
            actual_instance_id=8,
            source_filename="mock_source.json",
        )
        write_counterfactual_batch_summary(
            summary_path,
            successes=[
                {
                    "grid_size": 7,
                    "grid_complexity": 0.0,
                    "output_path": str(counterfactual_path),
                }
            ],
        )

        calls = []

        def fake_get_trajectory(**kwargs):
            calls.append(kwargs)
            Path(kwargs["output_path"]).write_text(
                json.dumps(
                    {
                        "steps": [{"step_id": 0}],
                        "model_params": {
                            "max_steps_per_trajectory": kwargs[
                                "max_steps_per_trajectory"
                            ]
                        },
                    }
                )
            )

        monkeypatch.setattr(counterfactual_cmds, "get_trajectory", fake_get_trajectory)

        manifest_path = Path(
            get_counterfactual_trajectories_all_pairs(
                counterfactual_summary_path=str(summary_path),
                output_dir=str(output_dir),
                model_names=["provider/model.v1"],
                max_workers=1,
            )
        )

        manifest = json.loads(manifest_path.read_text())

        assert manifest_path.exists()
        assert manifest["success_count"] == 2
        assert manifest["failure_count"] == 0
        assert len(calls) == 2
        assert {call["max_steps_per_trajectory"] for call in calls} == {1}
        assert {call["use_safe_reset"] for call in calls} == {True}
        assert all(
            isinstance(call["env"], counterfactual_cmds.FullObservabilityTextWrapper)
            for call in calls
        )
        assert any(
            "agent_moved_agent_moved_00" in call["output_path"] for call in calls
        )
        assert any("goal_moved_goal_moved_00" in call["output_path"] for call in calls)
        assert {
            (entry["counterfactual_type"], entry["counterfactual_id"])
            for entry in manifest["successes"]
        } == {
            ("agent_moved", "agent_moved_00"),
            ("goal_moved", "goal_moved_00"),
        }

    def test_get_counterfactual_trajectories_all_pairs_records_malformed_source_failure(
        self, tmp_path: Path
    ):
        summary_path = tmp_path / "counterfactual_batch_summary.json"
        malformed_path = tmp_path / "size7" / "counterfactuals_bad.json"
        output_dir = tmp_path / "counterfactual_trajectories"
        write_counterfactual_file(malformed_path, SUCCESS_GRID, malformed=True)
        write_counterfactual_batch_summary(
            summary_path,
            successes=[
                {
                    "grid_size": 7,
                    "grid_complexity": 0.0,
                    "output_path": str(malformed_path),
                }
            ],
        )

        manifest_path = Path(
            get_counterfactual_trajectories_all_pairs(
                counterfactual_summary_path=str(summary_path),
                output_dir=str(output_dir),
                max_workers=1,
                continue_on_error=True,
            )
        )

        manifest = json.loads(manifest_path.read_text())

        assert manifest["success_count"] == 0
        assert manifest["failure_count"] == 1
        assert manifest["failures"][0]["source_counterfactual_path"] == str(
            malformed_path
        )

    def test_get_counterfactual_trajectories_all_pairs_strict_stop_on_malformed_source(
        self, tmp_path: Path
    ):
        summary_path = tmp_path / "counterfactual_batch_summary.json"
        malformed_path = tmp_path / "size7" / "counterfactuals_bad.json"
        output_dir = tmp_path / "counterfactual_trajectories"
        write_counterfactual_file(malformed_path, SUCCESS_GRID, malformed=True)
        write_counterfactual_batch_summary(
            summary_path,
            successes=[
                {
                    "grid_size": 7,
                    "grid_complexity": 0.0,
                    "output_path": str(malformed_path),
                }
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            get_counterfactual_trajectories_all_pairs(
                counterfactual_summary_path=str(summary_path),
                output_dir=str(output_dir),
                max_workers=1,
                continue_on_error=False,
            )

        assert "Failed to expand counterfactual source" in str(exc_info.value)

    def test_cli_registers_counterfactual_trajectory_command(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cli_module = import_module("reveng.commands.cli")
        captured = {}

        def fake_subcommand_cli_from_dict(commands):
            captured.update(commands)

        monkeypatch.setattr(
            cli_module.tyro.extras,
            "subcommand_cli_from_dict",
            fake_subcommand_cli_from_dict,
        )

        cli_module.main()

        assert "get_counterfactual_trajectories_all_pairs" in captured


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
