"""
Example runner: generate trajectories and visualize replays.

What this script demonstrates
- Building a MiniGrid environment with the same wrapper-selection pattern used in
    `environment_generator/utils.py` (image or text observations, full/partial observability)
- Running a simple policy (random or A* on fully observable envs) through
    the shared `generate_trajectories` API
- Replaying trajectories with `visualize_trajectory` on the SAME environment instance
    that generated them so the visualization matches the rollout (same start/goal)

Quick start (from repo root)
- Random policy, image observations, small grid
        PYTHONPATH=src python -m reveng.trajectory_generator.example \
            --trajectories 1 --obs-modality image --observability full --size 6

- A* policy, larger grid
        PYTHONPATH=src python -m reveng.trajectory_generator.example \
            --policy astar --trajectories 1 --max-steps 200 \
            --obs-modality image --observability full --size 12

- With interior obstacles (vertical wall with a gap)
        PYTHONPATH=src python -m reveng.trajectory_generator.example \
            --policy astar --with-obstacles --trajectories 1 --max-steps 200 \
            --obs-modality image --observability full --size 12

Notes
- Visualization backend: the environment is created with `render_mode="human"`, which
    is ideal for interactive windows but does not return RGB frames. If you want to save
    a GIF during replay, switch the `render_mode` to `"rgb_array"` in `build_wrapped_env`
    (see the inline comment below), then rerun. The call to `visualize_trajectory(..., save_gif_path="run.gif")`
    will only write a file when frames are actually available from the renderer.
- Module execution: using the `-m` form from repo root ensures imports resolve with the
    `src/` layout. Alternatively, install the package (editable) and run without PYTHONPATH.
"""

import argparse
import time

# Third-party imports
import minigrid.wrappers as mg_wrappers

# Project imports
import reveng.environment_generator.custom_minigrid as custom_minigrid
import reveng.environment_generator.wrappers.rgb_obs_wrappers as rgb_wrappers
import reveng.environment_generator.wrappers.text_obs_wrapper as text_wrappers
import reveng.trajectory_generator.policies as policies
import reveng.trajectory_generator.trajectory_generator as traj_gen
from reveng.trajectory_generator.trajectory_generator import visualize_trajectory


def build_wrapped_env(
    size: int, obs_modality: str, observability: str, with_obstacles: bool = False
):
    """
    Construct a Simple2DNavigationEnv and wrap it according to modality/observability.

    Args:
      size: Grid width/height.
      obs_modality: "image" or "text" observations.
      observability: "full" or "partial" (partial text not implemented).
      with_obstacles: When True, inject a simple interior wall with a gap.

    Returns:
      A wrapped gymnasium-compatible environment ready for rollout.
    """
    base_env = custom_minigrid.Simple2DNavigationEnv(render_mode="rgb_array", size=size)

    # Optionally inject simple interior obstacles that persist across resets
    if with_obstacles:
        from types import MethodType

        from minigrid.core.world_object import Wall

        orig_gen_grid = base_env._gen_grid

        def _gen_grid_with_obstacles(self, width, height):
            # Call original grid generator (sets borders, goal, agent pos/dir)
            orig_gen_grid(width, height)

            # Add a vertical wall column with a single gap
            wall_x = max(2, width // 2)
            gap_y = height // 2

            # Avoid blocking start/goal and the gap cell
            forbidden = {
                tuple(self.agent_start_pos),
                tuple(self.goal_pos),
                (wall_x, gap_y),
            }

            for y in range(1, height - 1):  # keep outer border intact
                pos = (wall_x, y)
                if pos in forbidden:
                    continue
                cell = self.grid.get(*pos)
                if cell is None:
                    self.put_obj(Wall(), *pos)

        base_env._gen_grid = MethodType(_gen_grid_with_obstacles, base_env)

    if obs_modality == "image":
        if observability == "full":
            wrapper_cls = mg_wrappers.RGBImgObsWrapper
        elif observability == "partial":
            wrapper_cls = rgb_wrappers.OmnidirectionalFogOfWarRGBImgObsWrapper
        else:
            raise ValueError(f"Unknown observability: {observability}")
    elif obs_modality == "text":
        if observability == "full":
            wrapper_cls = text_wrappers.FullObservabilityTextWrapper
        else:
            raise NotImplementedError(
                "Partial text observability is not implemented yet"
            )
    else:
        raise ValueError(f"Unknown observation modality: {obs_modality}")

    return wrapper_cls(base_env)


def main():
    parser = argparse.ArgumentParser(description="Random trajectory generation demo")
    parser = argparse.ArgumentParser(
        description="Trajectory generation + visualization demo"
    )
    parser.add_argument(
        "--obs-modality",
        choices=["image", "text"],
        default="image",
        help="Observation modality",
    )
    parser.add_argument(
        "--observability",
        choices=["full", "partial"],
        default="full",
        help="Observability level",
    )
    parser.add_argument(
        "--trajectories", type=int, default=1, help="Number of trajectories"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Max steps per trajectory"
    )
    parser.add_argument(
        "--policy",
        choices=["random", "astar"],
        default="random",
        help="Policy to use for action selection",
    )
    parser.add_argument(
        "--with-obstacles",
        action="store_true",
        help="Add simple interior obstacles to the grid (vertical wall with a gap)",
    )
    args = parser.parse_args()

    env = build_wrapped_env(
        args.size,
        args.obs_modality,
        args.observability,
        with_obstacles=args.with_obstacles,
    )

    try:
        # Debug: show initial start/goal for this env instance
        try:
            base_env = env.unwrapped
            start_pos = tuple(base_env.agent_start_pos)
            goal_pos = tuple(base_env.goal_pos)
            md = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
            print(f"Start={start_pos} Goal={goal_pos} ManhattanDistance={md}")
        except Exception:
            pass
        # Choose the agent based on CLI flag
        if args.policy == "astar":
            if args.observability != "full":
                print(
                    "[note] A* policy assumes full observability; proceeding anyway by using base env state."
                )
            agent = policies.astar_policy
        else:
            agent = policies.random_policy

        trajectories = traj_gen.generate_trajectories(
            env=env,
            agent=agent,  # selected policy from policies.py
            num_trajectories=args.trajectories,
            max_steps_per_trajectory=args.max_steps,
            include_thoughts=False,
            reset_between_trajectories=True,
        )

        # Print a short summary
        print(f"Generated {len(trajectories)} trajectories")
        for i, traj in enumerate(trajectories, start=1):
            reached_goal = traj.final_reward and traj.final_reward > 0
            reason = (
                "reached goal"
                if reached_goal
                else (
                    "hit max steps"
                    if len(traj.steps) >= args.max_steps
                    else "truncated/ended early"
                )
            )
            print(
                f"Trajectory {i}: {len(traj.steps)} steps, total reward={traj.final_reward} ({reason})"
            )

        # Visualize by replaying actions from trajectories on the SAME env
        # so start/goal positions and dynamics match the generation run
        for i, traj in enumerate(trajectories, start=1):
            print(f"Replaying Trajectory {i}...")
            visualize_trajectory(
                traj,
                env,
                sleep=0.05,
                save_gif_path=f"reveng/trajectory_generator/trajectory_gifs/{args.policy}_run.gif",
            )
            # Print final position vs goal for clarity
            try:
                agent_pos = tuple(env.unwrapped.agent_pos)
                goal_pos = tuple(env.unwrapped.goal_pos)
                print(
                    f"EndPos={agent_pos} Goal={goal_pos} at_goal={agent_pos == goal_pos}"
                )
            except Exception:
                pass
            time.sleep(0.3)

        # Keep the window visible briefly after completion
        time.sleep(1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
