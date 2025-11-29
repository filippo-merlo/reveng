import json
from pathlib import Path

from reveng.agents.llm_agent import LLMAgent
from reveng.datatypes import CustomJSONEncoder
from reveng.environment_generator.rooms_minigrid import RoomsMinigridEnv
from reveng.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)
from reveng.policy_inspector.policy_elicitation import generate_one_trajectory


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("instrumental_goals_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model name
    model_name = "together_ai/openai/gpt-oss-120b"

    # Create RoomsMinigridEnv with door and key
    print("Creating RoomsMinigridEnv with door and key...")
    env = RoomsMinigridEnv(add_door_key=True, max_steps=100)
    # env.reset()

    # Create LLM agent with instrumental goals template
    print(f"Creating LLM agent with model: {model_name}")
    template_path = (
        Path(__file__).parent.parent
        / "templates"
        / "grid_full_observability_instrumental_goals.j2"
    )
    agent = LLMAgent(
        model_name=model_name, name="LLM agent", template_path=template_path
    )

    # Collect one trajectory
    print("Collecting trajectory...")
    trajectory = generate_one_trajectory(
        env=env,
        grid_id="rooms_env_with_door_key",
        agent=agent,
        max_steps_per_trajectory=50,
        top_logprobs=1,
        use_logprobs=False,
        text_wrapper_cls=FullObservabilityTextWrapper,
        save_images=True,
        image_save_dir=output_dir,
    )

    # Save trajectory
    trajectory_path = output_dir / "rooms_trajectory.json"
    print(f"Saving trajectory to {trajectory_path}")
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2, cls=CustomJSONEncoder)

    # Print cost summary
    cost_summary = agent.get_cost_summary()
    print("\nTrajectory collection complete!")
    print(f"Cost summary: {cost_summary}")
    print(f"Final reward: {trajectory.final_reward}")
    print(f"Number of steps: {len(trajectory.steps)}")
