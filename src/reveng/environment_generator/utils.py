import os
import time

import matplotlib.pyplot as plt
import pygame
from minigrid.wrappers import RGBImgObsWrapper

from custom_minigrid import Simple2DNavigationEnv
from wrappers.rgb_obs_wrappers import OmnidirectionalFogOfWarRGBImgObsWrapper
from wrappers.text_obs_wrapper import FullObservabilityTextWrapper, FogOfWarTextWrapper


class ObsWrapperRegistry:
    wrappers = {
        "image": {
            "full": RGBImgObsWrapper,
            "partial": OmnidirectionalFogOfWarRGBImgObsWrapper,
        },
        "text": {
            "full": FullObservabilityTextWrapper,
            "partial": FogOfWarTextWrapper,
        },
    }

    @staticmethod
    def get_wrapper(modality: str, observability: str):
        return ObsWrapperRegistry.wrappers.get(modality, {}).get(observability)


def run_random_episodes(
    episodes=5,
    size=10,
    obs_modality: str = "image",
    observability: str = "full",
    save_images=False,
    config_path=None,
):
    """
    Runs episodes with a random agent
    """
    base_env = Simple2DNavigationEnv(render_mode="human", size=size)
    obs_wrapper = ObsWrapperRegistry.get_wrapper(obs_modality, observability)
    if obs_modality == "text" and config_path:
        env = obs_wrapper(base_env, config_path=config_path)
    else:
        env = obs_wrapper(base_env)

    for i in range(episodes):
        # Reset the environment
        env.reset()
        total_reward = 0
        print(f"--- Starting Episode {i + 1}/{episodes} ---")

        terminated, truncated = False, False

        # Run the episode until done
        while not (terminated or truncated):
            env.render()

            # Choose a random action
            action = env.action_space.sample()
            print(f"Action sampled: {action} ({base_env.actions(action).name})")

            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Save observation images if requested
            if save_images and obs_modality == "image":
                # Create images directory if it doesn't exist
                if not os.path.exists("images"):
                    os.makedirs("images")

                plt.figure(figsize=(8, 8))
                plt.imshow(obs["image"])
                plt.title(f"Episode {i + 1}, Step {base_env.step_count}")
                plt.savefig(f"images/episode_{i + 1}_step_{base_env.step_count}.png")
                plt.close()

            # A small delay to make the simulation watchable
            time.sleep(0.1)

        # --- Episode End ---
        env.render()

        # Print Episode Summary
        print(f"--- Episode {i + 1} Finished ---")
        if terminated:
            print("Goal was reached!")
        elif truncated:
            print("Time limit (max_steps) was reached.")
        print(f"Total reward for the episode: {total_reward}\n")

        time.sleep(1.5)  # Pause before the next episode

    env.close()


def manual_control(
    size=10,
    obs_modality: str = "image",
    observability: str = "full",
    save_images=True,
    config_path=None,
):
    base_env = Simple2DNavigationEnv(render_mode="human", size=size)
    obs_wrapper = ObsWrapperRegistry.get_wrapper(obs_modality, observability)
    if obs_modality == "text" and config_path:
        env = obs_wrapper(base_env, config_path=config_path)
    else:
        env = obs_wrapper(base_env)
    env.reset()

    # Map pygame keys to environment actions for cleaner handling
    key_to_action = {
        pygame.K_LEFT: base_env.actions.LEFT,
        pygame.K_RIGHT: base_env.actions.RIGHT,
        pygame.K_UP: base_env.actions.UP,
        pygame.K_DOWN: base_env.actions.DOWN,
    }

    running = True
    while running:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(obs)

                    # Save observation images if requested
                    if save_images and obs_modality == "image":
                        # Create images directory if it doesn't exist
                        if not os.path.exists("images"):
                            os.makedirs("images")

                        plt.figure(figsize=(8, 8))
                        plt.imshow(obs["image"])
                        plt.title(f"Step {base_env.step_count}")
                        plt.savefig(f"images/{base_env.step_count}.png")
                        plt.close()

                    print(
                        f"Step: {base_env.step_count}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
                    )

                    if terminated or truncated:
                        print("Episode finished. Resetting.")
                        env.reset()

    env.close()
