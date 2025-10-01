import pandas as pd
import numpy as np
import argparse
from custom_minigrid import Simple2DNavigationEnv
import reveng.environment_generator.wrappers.text_obs_wrapper as text_wrappers
import reveng.trajectory_generator.trajectory_generator as traj_gen
import reveng.agents as agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--results-path", type=str, default="grids_for_probing.csv")
    parser.add_argument("--distance-prediction", action="store_true", help="Only save each environment once, do not loop over (x,y) coordinates.")
    args = parser.parse_args()

    results = []
    complexities = np.linspace(0.0, 1.0, args.num_envs)

    print(
        f"Generating {args.num_envs} environments of size {args.size} with complexities: {complexities}"
    )

    for env_idx in range(args.num_envs):

        one_env_results = []
        env = Simple2DNavigationEnv(size=args.size, complexity=complexities[env_idx])

        wrapped_env = text_wrappers.FullObservabilityTextWrapper(env)

        observation_str, info = wrapped_env.reset()
        print(observation_str)

        width = wrapped_env.unwrapped.width
        height = wrapped_env.unwrapped.height
        print(f"Width: {width}, Height: {height}")

        if args.distance_prediction:
            width = 1
            height = 1

        for j in range(height):
            for i in range(width):
                cell_type = wrapped_env._get_cell_type_at_position(
                    wrapped_env.unwrapped, None, i, j
                )
                symbol = wrapped_env.symbols[cell_type]

                one_env_results.append(
                    {
                        "env_idx": env_idx,
                        "observation": observation_str,
                        "x": i,
                        "y": j,
                        "cell_type": cell_type,
                        "symbol": symbol,
                        "classes_map": repr(wrapped_env.symbols),
                    }
                )
        
        print(f"Generating a trajectory from alphastar")
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
        for one_env_result in one_env_results:
            one_env_result["optimal_trajectory_length"] = optimal_trajectory_length
            results.append(one_env_result)

    df = pd.DataFrame(results)
    df.to_csv("grids_for_probing.csv", index=False)
