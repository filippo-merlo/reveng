"""Baseline experiment runners.

This module provides two main experiment runners:
- `run_baseline_trajectories`: generate and save trajectories for agents across
    a dataset of environments (with optional perturbations).
- `run_baseline_policies`: elicit and save full action policies for agents on
    each environment, plus summary statistics.

The CLI at the bottom of the file exposes a `--mode` flag to run
either trajectories, policies, or both. A `metadata.json` is written to
the base output directory and per-mode summaries are saved in subfolders.
"""

import argparse
import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from reveng.agents.alpha_start_agent import Agent, AlphaStarAgent, RandomAgent
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.env_transformations import (
    IsoDifficultyTransformationFactory,
)
from reveng.experiments.policy_elicitation import elicit_policy
from reveng.trajectory_generator.trajectory_generator import generate_trajectories


def run_baseline_policies(
    dataset: Dict[str, Simple2DNavigationEnv],
    agents: List[Agent],
    perturbations: Optional[
        List[Callable[[Simple2DNavigationEnv], Simple2DNavigationEnv]]
    ] = None,
    output_dir: str = "src/reveng/experiments/policy_results/",
) -> None:
    """
    Elicit full policies for multiple agents across a dataset of environments.

    For every combination of environment, agent, and optional perturbation, this function
    queries :func:`elicit_policy` to obtain the agent's preferred action at every reachable
    position. Results are persisted to disk alongside aggregate summary statistics to
    support downstream analysis.

    Args:
        dataset: Mapping of grid identifiers to configured environments.
        agents: Agents to evaluate.
        perturbations: Optional list of environment transformation callables. Each callable
            receives an environment instance and returns a transformed copy. When omitted,
            only the base environment is evaluated.
        output_dir: Directory where policy artifacts and metadata are written. The directory
            structure mirrors ``run_baseline_trajectories`` but stores policy maps instead of
            trajectory files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    policies_path = output_path / "policies"
    policies_path.mkdir(exist_ok=True)

    if perturbations is None:
        perturbations = []

    all_results: List[Dict] = []

    total_combinations = (
        len(dataset) * len(agents) * (len(perturbations) if perturbations else 1)
    )
    progress = tqdm(
        total=total_combinations,
        desc="Policy elicitations",
        leave=False,
        ncols=80,
        unit="it",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    print("Starting baseline policy elicitation:")
    print(f"  Environments: {len(dataset)}")
    print(f"  Agents: {len(agents)}")
    print(
        f"  Perturbations: {len(perturbations) if perturbations else 1} (including base)"
    )
    print(f"  Total combinations: {total_combinations}")
    print(f"  Output directory: {output_path}")
    print()

    for grid_id, base_env in dataset.items():
        for agent in agents:
            agent.reset()

            perturbation_list = [("base", None)] + [
                (getattr(p, "__name__", f"pert_{idx}"), p)
                for idx, p in enumerate(perturbations)
            ]

            for perturbation_name, perturbation_fn in perturbation_list:
                try:
                    env = base_env
                    if perturbation_fn is not None:
                        env = perturbation_fn(env)

                    policy_map, policy_metadata = elicit_policy(env, agent)

                    policy_filename = (
                        f"{grid_id}_{agent.name}_{perturbation_name}_policy.json"
                    )
                    policy_filepath = policies_path / policy_filename

                    policy_record = {
                        "grid_id": grid_id,
                        "agent_name": agent.name,
                        "perturbation": perturbation_name,
                        "policy": policy_map,
                        "metadata": policy_metadata,
                    }

                    with open(policy_filepath, "w") as policy_file:
                        json.dump(policy_record, policy_file, indent=2, default=str)

                    total_cells = (
                        len(policy_map) * len(policy_map[0]) if policy_map else 0
                    )
                    evaluated_cells = sum(
                        1 for row in policy_map for action in row if action != -1
                    )

                    result_entry = {
                        "grid_id": grid_id,
                        "agent_name": agent.name,
                        "perturbation": perturbation_name,
                        "policy_filename": policy_filename,
                        "evaluated_cells": evaluated_cells,
                        "total_cells": total_cells,
                        "coverage": (evaluated_cells / total_cells)
                        if total_cells
                        else 0.0,
                    }
                    all_results.append(result_entry)

                    progress.write(
                        f"✓ Policy saved: env={grid_id}, agent={agent.name}, perturb={perturbation_name}"
                    )

                except Exception as exc:
                    progress.write(
                        f"✗ Error for env={grid_id}, agent={agent.name}, perturb={perturbation_name}: {exc}"
                    )
                    error_entry = {
                        "grid_id": grid_id,
                        "agent_name": agent.name,
                        "perturbation": perturbation_name,
                        "error": str(exc),
                    }
                    all_results.append(error_entry)

                finally:
                    progress.update(1)

    progress.close()

    summary = generate_policy_summary(all_results, agents, policies_path)
    with open(output_path / "policy_summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    total_policies = len([r for r in all_results if "policy_filename" in r])

    print(f"\n{'=' * 60}")
    print("Policy elicitation complete!")
    print(f"  Total policies saved: {total_policies}")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 60}")


def run_baseline_trajectories(
    dataset: Dict[str, Simple2DNavigationEnv],
    agents: List[Agent],
    perturbations: Optional[
        List[Callable[[Simple2DNavigationEnv], Simple2DNavigationEnv]]
    ] = None,
    num_trajectories_per_config: int = 1,
    output_dir: str = "src/reveng/experiments/results/",
    max_steps_per_trajectory: Optional[int] = None,
) -> None:
    """
    Run multiple agents on a dataset of environments with optional perturbations.

    This function generates trajectories for all combinations of:
    - Environments (from dataset)
    - Agents (from agents list)
    - Perturbations (optional transformations applied to environments)

    Results are saved in a structured format for efficient analysis.

    Args:
        dataset: Dictionary mapping grid_id -> Simple2DNavigationEnv
        agents: List of Agent instances to evaluate
        perturbations: Optional list of environment transformation functions.
                      Each function takes an env and returns a transformed env.
                      If None, uses only the base environment.
        num_trajectories_per_config: Number of trajectories to generate per
                                     (environment, agent, perturbation) combination
        output_dir: Directory to save results
        max_steps_per_trajectory: Optional max steps per trajectory (uses env.max_steps if None)

    Output Structure:
        output_dir/
            metadata.json              # Experiment metadata
            trajectories/
                {grid_id}_{agent_name}_{perturbation_name}_{traj_idx}.json
            summary.json              # Aggregate statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_path / "trajectories"
    trajectories_path.mkdir(exist_ok=True)

    # Initialize perturbations list
    if perturbations is None:
        perturbations = []

    # Track all results for summary
    all_results = []

    total_combinations = (
        len(dataset) * len(agents) * (len(perturbations) if perturbations else 1)
    )
    progress = tqdm(
        total=total_combinations,
        desc="Trajectory runs",
        leave=False,
        ncols=80,
        unit="it",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    print("Starting baseline experiment:")
    print(f"  Environments: {len(dataset)}")
    print(f"  Agents: {len(agents)}")
    print(
        f"  Perturbations: {len(perturbations) if perturbations else 1} (including base)"
    )
    print(f"  Trajectories per config: {num_trajectories_per_config}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Output directory: {output_path}")
    print()

    # Iterate over all combinations
    for grid_id, base_env in dataset.items():
        for agent in agents:
            # Reset agent state for each environment
            agent.reset()

            # Test base environment and all perturbations
            perturbation_list = [("base", None)] + [
                (getattr(p, "__name__", f"pert_{i}"), p)
                for i, p in enumerate(perturbations)
            ]

            for perturbation_name, perturbation_fn in perturbation_list:
                try:
                    # Use the environment directly from dataset
                    env = base_env

                    traj_metadata = {
                        "grid_id": grid_id,
                        "perturbation_name": perturbation_name,
                        "agent_name": agent.name,
                    }

                    # Apply perturbation if provided
                    if perturbation_fn is not None:
                        env = perturbation_fn(env)

                    # Generate trajectories using existing infrastructure
                    trajectories = generate_trajectories(
                        env=env,
                        agent=agent,
                        num_trajectories=num_trajectories_per_config,
                        max_steps_per_trajectory=max_steps_per_trajectory
                        or env.max_steps,
                        reset_between_trajectories=True,
                        traj_metadata=traj_metadata,
                    )

                    # Save each trajectory
                    for traj_idx, trajectory in enumerate(trajectories):
                        # Create unique filename
                        filename = f"{grid_id}_{agent.name}_{perturbation_name}_{traj_idx:03d}.json"
                        filepath = trajectories_path / filename

                        # Save trajectory
                        with open(filepath, "w") as f:
                            json.dump(asdict(trajectory), f, indent=2)

                        # Track result
                        result_entry = {
                            "grid_id": grid_id,
                            "agent_name": agent.name,
                            "perturbation": perturbation_name,
                            "trajectory_idx": traj_idx,
                            "filename": filename,
                            "final_reward": trajectory.final_reward,
                            "num_steps": len(trajectory.steps),
                            "success": trajectory.final_reward > 0
                            if trajectory.final_reward is not None
                            else False,
                        }
                        all_results.append(result_entry)

                    progress.write(
                        f"✓ Trajectories: env={grid_id}, agent={agent.name}, perturb={perturbation_name}, count={len(trajectories)}"
                    )

                except Exception as e:
                    progress.write(
                        f"✗ Error for env={grid_id}, agent={agent.name}, perturb={perturbation_name}: {e}"
                    )
                    # Log error but continue
                    error_entry = {
                        "grid_id": grid_id,
                        "agent_name": agent.name,
                        "perturbation": perturbation_name,
                        "error": str(e),
                    }
                    all_results.append(error_entry)

                finally:
                    progress.update(1)

    progress.close()

    # Generate and save summary statistics
    summary = generate_trajectory_summary(all_results, agents)
    with open(output_path / "trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Experiment complete!")
    print(
        f"  Total trajectories generated: {len([r for r in all_results if 'filename' in r])}"
    )
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 60}")


def generate_policy_summary(
    results: List[Dict], agents: List[Agent], policies_dir: Optional[Path] = None
) -> Dict:
    """Aggregate policy elicitation results by agent and perturbation.

    If `policies_dir` is provided, this will also compute, for each agent, the
    average percentage of grid squares where the agent's action differs from the
    AStar agent's action (averaged across grids/perturbations where both policies
    are available). If AStar was not run or comparisons cannot be performed, the
    value will be 0.0.
    """

    valid_results = [r for r in results if "policy_filename" in r]
    errors = [r for r in results if "error" in r]

    summary: Dict[str, Dict] = {
        "total_policies": len(valid_results),
        "total_errors": len(errors),
        "average_evaluated_cells": 0.0,
        "average_coverage": 0.0,
        "by_agent": {},
        "by_perturbation": {},
    }

    if valid_results:
        summary["average_evaluated_cells"] = sum(
            r["evaluated_cells"] for r in valid_results
        ) / len(valid_results)
        summary["average_coverage"] = sum(r["coverage"] for r in valid_results) / len(
            valid_results
        )

    # Build a lookup of AStar policies by (grid_id, perturbation)
    astar_policies = {}
    if policies_dir is not None:
        for r in results:
            if r.get("agent_name") == "AStar" and "policy_filename" in r:
                key = (r.get("grid_id"), r.get("perturbation"))
                try:
                    with open(Path(policies_dir) / r["policy_filename"], "r") as pf:
                        astar_policies[key] = json.load(pf).get("policy")
                except Exception:
                    # If we can't load, leave out this entry
                    pass

    for agent in agents:
        agent_results = [r for r in results if r.get("agent_name") == agent.name]
        agent_valid = [r for r in agent_results if "policy_filename" in r]
        agent_errors = [r for r in agent_results if "error" in r]

        if agent_results:
            summary["by_agent"][agent.name] = {
                "num_policies": len(agent_valid),
                "num_errors": len(agent_errors),
                "average_evaluated_cells": sum(
                    r["evaluated_cells"] for r in agent_valid
                )
                / len(agent_valid)
                if agent_valid
                else 0.0,
                "average_coverage": sum(r["coverage"] for r in agent_valid)
                / len(agent_valid)
                if agent_valid
                else 0.0,
            }

        # Compute average difference from AStar if possible
        avg_diff = 0.0
        if policies_dir is not None and agent.name != "AStar":
            diffs = []
            for r in agent_valid:
                key = (r.get("grid_id"), r.get("perturbation"))
                astar_policy = astar_policies.get(key)
                if not astar_policy:
                    continue
                try:
                    with open(Path(policies_dir) / r["policy_filename"], "r") as pf:
                        agent_policy = json.load(pf).get("policy")
                except Exception:
                    continue

                # Count AStar-defined cells
                astar_defined = [
                    (i, j)
                    for i, row in enumerate(astar_policy)
                    for j, a in enumerate(row)
                    if a != -1
                ]
                if not astar_defined:
                    continue

                # Count differing cells where both have defined actions
                diff_count = 0
                defined_count = 0
                for i, j in astar_defined:
                    try:
                        a_act = astar_policy[i][j]
                        b_act = agent_policy[i][j]
                    except Exception:
                        continue
                    if b_act == -1:
                        # agent undefined at this cell; treat as difference
                        defined_count += 1
                        diff_count += 1
                    else:
                        defined_count += 1
                        if b_act != a_act:
                            diff_count += 1

                if defined_count > 0:
                    diffs.append(diff_count / defined_count)

            if diffs:
                avg_diff = sum(diffs) / len(diffs)

        # store the computed metric
        summary["by_agent"][agent.name]["avg_diff_from_astar"] = avg_diff

    perturbation_names = set(
        r.get("perturbation") for r in results if r.get("perturbation") is not None
    )

    for perturbation in perturbation_names:
        pert_results = [r for r in results if r.get("perturbation") == perturbation]
        pert_valid = [r for r in pert_results if "policy_filename" in r]
        pert_errors = [r for r in pert_results if "error" in r]

        summary["by_perturbation"][perturbation] = {
            "num_policies": len(pert_valid),
            "num_errors": len(pert_errors),
            "average_evaluated_cells": sum(r["evaluated_cells"] for r in pert_valid)
            / len(pert_valid)
            if pert_valid
            else 0.0,
            "average_coverage": sum(r["coverage"] for r in pert_valid) / len(pert_valid)
            if pert_valid
            else 0.0,
        }

    return summary


def generate_trajectory_summary(results: List[Dict], agents: List[Agent]) -> Dict:
    """Generate summary statistics from experiment results."""
    summary = {
        "total_trajectories": len([r for r in results if "filename" in r]),
        "total_errors": len([r for r in results if "error" in r]),
        "by_agent": {},
        "by_perturbation": {},
    }

    # Compute per-agent statistics
    for agent in agents:
        agent_results = [
            r for r in results if r.get("agent_name") == agent.name and "filename" in r
        ]
        if agent_results:
            summary["by_agent"][agent.name] = {
                "num_trajectories": len(agent_results),
                "avg_steps": sum(r["num_steps"] for r in agent_results)
                / len(agent_results),
                "avg_reward": sum(r["final_reward"] or 0 for r in agent_results)
                / len(agent_results),
                "success_rate": sum(r["success"] for r in agent_results)
                / len(agent_results),
            }

    # Compute per-perturbation statistics
    perturbations = set(r.get("perturbation") for r in results if "perturbation" in r)
    for pert in perturbations:
        pert_results = [
            r for r in results if r.get("perturbation") == pert and "filename" in r
        ]
        if pert_results:
            summary["by_perturbation"][pert] = {
                "num_trajectories": len(pert_results),
                "avg_steps": sum(r["num_steps"] for r in pert_results)
                / len(pert_results),
                "avg_reward": sum(r["final_reward"] or 0 for r in pert_results)
                / len(pert_results),
                "success_rate": sum(r["success"] for r in pert_results)
                / len(pert_results),
            }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset pickle file"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1,
        help="Number of trajectories per configuration (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/reveng/experiments/baseline_results",
        help="Base output directory (default: src/reveng/experiments/baseline_results). Policy results will be written under <output-dir>/policy_results when applicable.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["trajectories", "policies", "both"],
        default="trajectories",
        help="Which experiment to run: 'trajectories', 'policies', or 'both' (default: trajectories)",
    )
    parser.add_argument(
        "--use-perturbations",
        action="store_true",
        help="Apply iso-difficulty transformations",
    )
    parser.add_argument(
        "--run-policies",
        action="store_true",
        help="Also run policy elicitation baseline",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    # Create agents
    agents = [
        AlphaStarAgent(name="AStar"),
        RandomAgent(name="Random"),
    ]

    # Setup perturbations if requested
    perturbations = None
    if args.use_perturbations:
        factory = IsoDifficultyTransformationFactory()
        perturbations = factory.get_transformations()

    # Decide which experiments to run based on mode
    run_trajectories = args.mode in ("trajectories", "both")
    run_policies = args.mode in ("policies", "both")

    # Write a single metadata.json to the base output directory
    base_output_path = Path(args.output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_name": "baseline",
        "timestamp": datetime.now().isoformat(),
        "num_environments": len(dataset),
        "num_agents": len(agents),
        "num_perturbations": len(perturbations) if perturbations else 1,
        "num_trajectories_per_config": args.num_trajectories,
        "agents": [agent.name for agent in agents],
    }
    with open(base_output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if run_trajectories:
        run_baseline_trajectories(
            dataset=dataset,
            agents=agents,
            perturbations=perturbations,
            num_trajectories_per_config=args.num_trajectories,
            output_dir=args.output_dir,
        )

    if run_policies:
        # place policy outputs under a subdirectory of the base output
        print("\nRunning policy elicitation baseline...")
        run_baseline_policies(
            dataset=dataset,
            agents=agents,
            perturbations=perturbations,
            output_dir=args.output_dir,
        )
