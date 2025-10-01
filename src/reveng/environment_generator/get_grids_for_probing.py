import pandas as pd
import numpy as np
import argparse
from custom_minigrid import Simple2DNavigationEnv
import reveng.environment_generator.wrappers.text_obs_wrapper as text_wrappers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--results-path", type=str, default="grids_for_probing.csv")
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

        for j in range(height):
            for i in range(width):
                cell_type = wrapped_env._get_cell_type_at_position(
                    wrapped_env.unwrapped, None, i, j
                )
                symbol = wrapped_env.symbols[cell_type]

                results.append(
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

    df = pd.DataFrame(results)
    df.to_csv("grids_for_probing.csv", index=False)
