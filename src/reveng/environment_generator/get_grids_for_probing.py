import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import reveng.agents as agents
import reveng.environment_generator.wrappers.text_obs_wrapper as text_wrappers
import reveng.trajectory_generator.trajectory_generator as traj_gen
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.llm_interface import BaseLLMInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=0,
        help="0: only save the initial observation, >0: save K trajectory steps for each environment",
    )
    parser.add_argument("--results-dir", type=str, default="outputs/grids_for_probing")
    parser.add_argument(
        "--file-name",
        type=str,
        default="grids_for_probing.csv",
        help="Name of the output CSV file",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--complexity",
        type=int,
        choices=range(100),
        help="Complexity percentage (0-100) for all environments. If not set, will be evenly spaced 0 to 100.",
    )
    args = parser.parse_args()

    results = []
    if args.complexity is not None:
        print(f"Using fixed complexity: {args.complexity}")
        complexities = np.array([args.complexity / 100] * args.num_envs)
    else:
        print(f"Interpolating complexities from 0 to 1 over {args.num_envs} steps.")
        complexities = np.linspace(0.0, 1.0, args.num_envs)

    print(f"Generating {args.num_envs} environments of size {args.size}")

    for env_idx in range(args.num_envs):
        env = Simple2DNavigationEnv(size=args.size, complexity=complexities[env_idx])

        partially_observable_env = text_wrappers.LoggingFogOfWarTextWrapper(env)
        width = partially_observable_env.unwrapped.width
        height = partially_observable_env.unwrapped.height
        print(f"Width: {width}, Height: {height}")
        po_observation_str, info = partially_observable_env.reset(seed=args.seed)
        print(f"Partially observable observation: {po_observation_str}")
        po_template_path = (
            Path(__file__).parent.parent / "templates" / "grid_partial_observability.j2"
        )
        po_interface = BaseLLMInterface(model_name=None, template_path=po_template_path)
        po_prompt = po_interface.render_template(grid_state=po_observation_str)
        print(f"Partially observable prompt: {po_prompt}")

        # To get the underlying grid we render the observation with a fully observable seen mask.
        ones_seen_mask = np.ones((height, width), dtype=bool)
        fo_observation_str = partially_observable_env._render(seen_mask=ones_seen_mask)
        print(f"Fully observable observation: {fo_observation_str}")
        fo_template_path = (
            Path(__file__).parent.parent / "templates" / "grid_full_observability.j2"
        )
        fo_interface = BaseLLMInterface(model_name=None, template_path=fo_template_path)
        fo_prompt = fo_interface.render_template(grid_state=fo_observation_str)
        print(f"Fully observable prompt: {fo_prompt}")

        po_cell_types = partially_observable_env.partially_observable_cell_type_log[0]
        fo_cell_types = partially_observable_env.fully_observable_cell_type_log[0]

        print("Generating a trajectory from AlphaStar")
        agent = agents.AlphaStarAgent()

        trajectory = traj_gen.generate_one_trajectory(
            env=partially_observable_env,
            observation=po_observation_str,
            info=info,
            agent=agent,
            max_steps_per_trajectory=args.size**2,
        )
        optimal_trajectory_length = len(trajectory.steps)
        print(f"Optimal trajectory length: {optimal_trajectory_length}")

        if args.trajectory_steps == 0:
            print("Only saving the initial observation")
            results.append(
                {
                    "env_idx": env_idx,
                    "fo_observation": fo_observation_str,
                    "fo_prompt": fo_prompt,
                    "po_observation": po_observation_str,
                    "po_prompt": po_prompt,
                    "fo_cell_types": fo_cell_types,
                    "po_cell_types": po_cell_types,
                    "classes_map": repr(partially_observable_env.grid_cells),
                    "optimal_trajectory_length": optimal_trajectory_length,
                    "trajectory_step": 0,
                }
            )
        elif args.trajectory_steps > 0:
            stepsize = len(trajectory.steps) // args.trajectory_steps
            steps_to_save = np.linspace(
                0, len(trajectory.steps) - 1, args.trajectory_steps, dtype=int
            )
            print(
                f"Saving {args.trajectory_steps} trajectory steps at indices: {steps_to_save}"
            )
            assert (
                trajectory.steps[0].observation
                == partially_observable_env.partially_observable_observation_log[0]
            )
            for step_idx, step in enumerate(trajectory.steps):
                assert (
                    step.observation
                    == partially_observable_env.partially_observable_observation_log[
                        step_idx
                    ]
                )
                if step_idx not in steps_to_save:
                    continue
                fo_observation = (
                    partially_observable_env.fully_observable_observation_log[step_idx]
                )
                fo_prompt = fo_interface.render_template(grid_state=fo_observation)
                po_observation = (
                    partially_observable_env.partially_observable_observation_log[
                        step_idx
                    ]
                )
                po_prompt = po_interface.render_template(grid_state=po_observation)
                fo_cell_types = partially_observable_env.fully_observable_cell_type_log[
                    step_idx
                ]
                po_cell_types = (
                    partially_observable_env.partially_observable_cell_type_log[
                        step_idx
                    ]
                )
                results.append(
                    {
                        "env_idx": env_idx,
                        "fo_observation": fo_observation,
                        "fo_prompt": fo_prompt,
                        "po_observation": po_observation,
                        "po_prompt": po_prompt,
                        "fo_cell_types": fo_cell_types,
                        "po_cell_types": po_cell_types,
                        "classes_map": repr(partially_observable_env.grid_cells),
                        "optimal_trajectory_length": optimal_trajectory_length
                        - step_idx,
                        "trajectory_step": step_idx,
                    }
                )
        else:
            raise ValueError(f"Invalid trajectory steps: {args.trajectory_steps}")

    df = pd.DataFrame(results)

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, args.file_name)
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
