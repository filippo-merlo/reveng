import pandas as pd
import numpy as np
import argparse
from custom_minigrid import Simple2DNavigationEnv
import reveng.environment_generator.wrappers.text_obs_wrapper as text_wrappers
import reveng.trajectory_generator.trajectory_generator as traj_gen
import reveng.agents as agents
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--results-dir", type=str, default="grids_for_probing")
    parser.add_argument(
        "--file-name",
        type=str,
        default="grids_for_probing.csv",
        help="Name of the output CSV file",
    )
    args = parser.parse_args()

    results = []
    complexities = np.linspace(0.0, 1.0, args.num_envs)

    print(
        f"Generating {args.num_envs} environments of size {args.size} with complexities: {complexities}"
    )

    for env_idx in range(args.num_envs):
        env = Simple2DNavigationEnv(size=args.size, complexity=complexities[env_idx])

        wrapped_env = text_wrappers.FullObservabilityTextWrapper(env)

        observation_str, info = wrapped_env.reset()
        print(observation_str)

        width = wrapped_env.unwrapped.width
        height = wrapped_env.unwrapped.height
        print(f"Width: {width}, Height: {height}")

        env_cell_types = []  # list of tuples (x, y, cell_type)
        for j in range(height):
            for i in range(width):
                cell_type = wrapped_env._get_cell_type_at_position(
                    wrapped_env.unwrapped, None, i, j
                )
                env_cell_types.append((i, j, cell_type))

        print("Generating a trajectory from AlphaStar")
        agent = agents.AlphaStarAgent()

        trajectory = traj_gen.generate_one_trajectory(
            env=wrapped_env,
            observation=observation_str,
            info=info,
            agent=agent,
            max_steps_per_trajectory=args.size**2,
        )
        print(f"Optimal trajectory length: {len(trajectory.steps)}")
        optimal_trajectory_length = len(trajectory.steps)

        results.append(
            {
                "env_idx": env_idx,
                "observation": observation_str,
                "cell_types": repr(env_cell_types),
                "classes_map": repr(wrapped_env.symbols),
                "optimal_trajectory_length": optimal_trajectory_length,
            }
        )

    df = pd.DataFrame(results)

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, args.file_name)
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
