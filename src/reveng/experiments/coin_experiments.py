import json
from pathlib import Path

from reveng.agents.llm_agent import LLMAgent
from reveng.datatypes import CustomJSONEncoder
from reveng.environment_generator.coin_minigrid import CoinMinigridEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)
from reveng.policy_inspector.policy_elicitation import generate_one_trajectory
from reveng.environment_generator.utils import remove_coin, clone_env


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("coin_experiments_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_no_coin = Path("coin_experiments_results_no_coin")
    output_dir_no_coin.mkdir(parents=True, exist_ok=True)

    # Model name
    model_name = "together_ai/openai/gpt-oss-20b"

    # Create CoinMinigridEnv
    print("Creating CoinMinigridEnv with two equal-length paths...")
    env = CoinMinigridEnv(size=9, max_steps=100)
    env.reset()

    env_no_coin = remove_coin(clone_env(env))

    # Create LLM agent with instrumental goals template
    print(f"Creating LLM agent with model: {model_name}")
    template_path = (
        Path(__file__).parent.parent / "templates" / "grid_full_observability_coin.j2"
    )
    agent = LLMAgent(
        model_name=model_name, name="LLM agent", template_path=template_path
    )

    # Collect one trajectory
    print("Collecting trajectory...")
    trajectory = generate_one_trajectory(
        env=env,
        grid_id="coin_env_two_paths",
        agent=agent,
        max_steps_per_trajectory=50,
        top_logprobs=1,
        use_logprobs=False,
        text_wrapper_cls=FullObservabilityTextWrapper,
        save_images=True,
        image_save_dir=output_dir,
    )

    # Save trajectory
    trajectory_path = output_dir / "coin_trajectory.json"
    print(f"Saving trajectory to {trajectory_path}")
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2, cls=CustomJSONEncoder)

    # Print cost summary
    cost_summary = agent.get_cost_summary()
    print("\nTrajectory collection complete!")
    print(f"Cost summary: {cost_summary}")
    print(f"Final reward: {trajectory.final_reward}")
    print(f"Number of steps: {len(trajectory.steps)}")

    # Print coin collection info if available
    if hasattr(trajectory, "info") and trajectory.info:
        last_info = (
            trajectory.info[-1]
            if isinstance(trajectory.info, list)
            else trajectory.info
        )
        if "coin_collected" in last_info:
            print(f"Coin collected: {last_info['coin_collected']}")

    # Now collect a trajectory without the coin
    print("\n" + "=" * 60)
    print("Collecting trajectory WITHOUT coin...")
    print("=" * 60)

    # Collect trajectory without coin
    trajectory_no_coin = generate_one_trajectory(
        env=env_no_coin,
        grid_id="coin_env_two_paths_no_coin",
        agent=agent,
        max_steps_per_trajectory=50,
        top_logprobs=1,
        use_logprobs=False,
        text_wrapper_cls=FullObservabilityTextWrapper,
        save_images=True,
        image_save_dir=output_dir_no_coin,
    )

    # Save trajectory without coin
    trajectory_no_coin_path = output_dir_no_coin / "coin_trajectory_no_coin.json"
    print(f"Saving no-coin trajectory to {trajectory_no_coin_path}")
    with open(trajectory_no_coin_path, "w") as f:
        json.dump(trajectory_no_coin, f, indent=2, cls=CustomJSONEncoder)

    # Print cost summary for no-coin trajectory
    cost_summary_no_coin = agent.get_cost_summary()
    print("\nNo-coin trajectory collection complete!")
    print(f"Cost summary (total): {cost_summary_no_coin}")
    print(f"Final reward: {trajectory_no_coin.final_reward}")
    print(f"Number of steps: {len(trajectory_no_coin.steps)}")

    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(
        f"With coin - Steps: {len(trajectory.steps)}, Reward: {trajectory.final_reward}"
    )
    print(
        f"Without coin - Steps: {len(trajectory_no_coin.steps)}, Reward: {trajectory_no_coin.final_reward}"
    )
