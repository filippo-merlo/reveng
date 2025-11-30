import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from reveng.agents.llm_agent import (
    PartiallyObservableLLMAgent,
    PartiallyObservableWithChatHistoryLLMAgent,
    PartiallyObservableWithNoteLLMAgent,
)
from reveng.datatypes import CustomJSONEncoder
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.policy_inspector.policy_elicitation import collect_trajectories


def remove_already_processed_environments(
    environments: List[Tuple[str, Simple2DNavigationEnv]], output_base: Path
) -> List[Tuple[str, Simple2DNavigationEnv]]:
    """Remove environments that have already been processed."""
    return [
        env
        for env in environments
        if not (output_base / f"{env[0]}_trajectories.json").exists()
    ]


def get_env_subset(
    environments: List[Tuple[str, Simple2DNavigationEnv]],
) -> List[Tuple[str, Simple2DNavigationEnv]]:
    for i, (grid_id, env) in enumerate(environments):
        if env.width == 11 and env.height == 11:
            cutoff = i
            break
    return environments[:cutoff]


def _process_single_environment(
    model_name: str,
    output_base: Path,
    grid_id: str,
    env,
    top_logprobs: int = 20,
    use_note: bool = False,
    use_logprobs: bool = True,
    num_trajectories: int = 2,
    max_steps_per_trajectory: int = 10,
    dynamic_steps_per_trajectory: bool = False,
):
    """Run policy elicitation and visualizations for a single environment.

    Returns a tuple of (grid_id, cost_summary_dict).
    """
    # Create a dedicated agent per environment to avoid shared mutable state across threads
    if use_note:
        llm_agent = PartiallyObservableWithNoteLLMAgent(
            model_name=model_name, name="LLM agent with note"
        )
    else:
        llm_agent = PartiallyObservableLLMAgent(
            model_name=model_name, name="LLM agent without note"
        )

    llm_agent = PartiallyObservableWithChatHistoryLLMAgent(
        model_name=model_name, name="LLM agent with chat history"
    )

    trajectories = collect_trajectories(
        env,
        grid_id,
        llm_agent,
        num_trajectories=num_trajectories,
        max_steps_per_trajectory=max_steps_per_trajectory,
        top_logprobs=top_logprobs,
        use_logprobs=use_logprobs,
        dynamic_steps_per_trajectory=dynamic_steps_per_trajectory,
    )

    # Save metadata
    trajectories_path = output_base / f"{grid_id}_trajectories.json"

    with open(trajectories_path, "w") as f:
        json.dump(trajectories, f, indent=2, cls=CustomJSONEncoder)

    return grid_id, llm_agent.get_cost_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full policy experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/baseline_grids.pkl",
        help="Path to dataset pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="partial_obs_results_threads",
        help="Base output directory (default: partial_obs_results_threads)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="together_ai/openai/gpt-oss-20b",
        help="Model name to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers to use (default: 2)",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Number of top logprobs to return (default: 20)",
    )
    parser.add_argument(
        "--use-logprobs",
        action="store_true",
        help="Use logprobs to select actions",
    )
    parser.add_argument(
        "--use-note",
        action="store_true",
        help="Use note-based policy elicitation",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=2,
        help="Number of trajectories to collect for each environment (default: 2)",
    )
    parser.add_argument(
        "--max-steps-per-trajectory",
        type=int,
        default=10,
        help="Maximum number of steps per trajectory (default: 10). Ignored if dynamic-steps-per-trajectory is True.",
    )
    parser.add_argument(
        "--dynamic-steps-per-trajectory",
        action="store_true",
        help="Use dynamic steps per trajectory",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    print(f"Dataset loaded: {len(dataset)} environments")

    # Model name for worker agents
    model = args.model_name

    # Create output directory structure: results/{model_name}/
    # Sanitize model name to avoid creating nested directories
    safe_model_name = args.model_name.replace("/", "_")
    output_base = (
        Path(args.output_dir) / f"{safe_model_name}{'_note' if args.use_note else ''}"
    )
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_base}")

    # Iterate through environments
    environments = list(dataset.items())  # All environments
    environments = get_env_subset(environments)

    environments = remove_already_processed_environments(environments, output_base)
    print(f"Remaining environments to process: {len(environments)}")
    # Parallel processing of environments
    cost_summaries = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                _process_single_environment,
                model,
                output_base,
                grid_id,
                env,
                args.top_logprobs,
                args.use_note,
                args.use_logprobs,
                args.num_trajectories,
                args.max_steps_per_trajectory,
                args.dynamic_steps_per_trajectory,
            ): grid_id
            for grid_id, env in environments
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing environments"
        ):
            grid_id = futures[future]
            try:
                _, cost_summary = future.result()
                cost_summaries.append(cost_summary)
            except Exception as exc:
                print(f"Environment {grid_id} failed with error: {exc}")
                raise exc

    # Aggregate and print cost summary across all workers
    total_cost = sum(cs.get("total_cost", 0.0) for cs in cost_summaries)
    total_calls = sum(cs.get("call_count", 0) for cs in cost_summaries)
    avg_cost_per_call = (total_cost / total_calls) if total_calls > 0 else 0.0

    print("\nPolicy elicitation complete!")
    print(
        f"Cost summary: {{'total_cost': {total_cost:.6f}, 'call_count': {total_calls}, 'avg_cost_per_call': {avg_cost_per_call:.6f}}}"
    )
