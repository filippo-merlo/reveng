import argparse

from utils import run_random_episodes, manual_control

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simulation in the Simple2DNavigationEnv."
    )
    parser.add_argument(
        "--mode",
        choices=["random", "manual"],
        default="manual",
        help="The mode to run the simulation in: 'random' for random actions, 'manual' for user control.",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run in 'random' mode. (Default: 5)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=20,
        help="The height and width of the grid for the environment. (Default: 10)",
    )
    parser.add_argument(
        "--obs-modality",
        choices=["image", "text"],
        default="image",
        help="Observation modality: 'image' for RGB images, 'text' for text descriptions. (Default: image)",
    )
    parser.add_argument(
        "--observability",
        choices=["full", "partial"],
        default="full",
        help="Observability level: 'full' for complete environment visibility, 'partial' for limited visibility. (Default: full)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save observation images during execution (only works with image modality)",
    )
    args = parser.parse_args()

    if args.mode == "random":
        print(
            f"Running in random mode for {args.episodes} episodes with a grid size of {args.size}x{args.size}."
            f" Observation modality: {args.obs_modality}, Observability: {args.observability}"
        )
        run_random_episodes(
            episodes=args.episodes, 
            size=args.size, 
            obs_modality=args.obs_modality, 
            observability=args.observability,
            save_images=args.save_images
        )
    elif args.mode == "manual":
        print(
            f"Running in manual mode with a grid size of {args.size}x{args.size}."
            f" Observation modality: {args.obs_modality}, Observability: {args.observability}"
        )
        manual_control(
            size=args.size, 
            obs_modality=args.obs_modality, 
            observability=args.observability,
            save_images=args.save_images
        )
