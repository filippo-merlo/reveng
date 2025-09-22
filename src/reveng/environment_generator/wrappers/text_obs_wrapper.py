import gymnasium
from gymnasium.spaces import Text


class FullObservabilityTextWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Define the new observation space.
        self.observation_space = Text(max_length=4096, charset="utf-8")

        # Map agent's direction to a character
        self.dir_map = {0: ">", 1: "v", 2: "<", 3: "^"}

    def observation(self, obs):
        """
        Generates the text observation from the environment's grid state.
        """
        env = self.unwrapped

        # 1. Start with the mission description
        mission = f"Mission: {env.mission}\n"

        # 2. Create a character grid
        grid_repr = []
        for j in range(env.height):
            row_str = ""
            for i in range(env.width):
                # Check for agent position first
                if i == env.agent_pos[0] and j == env.agent_pos[1]:
                    # row_str += self.dir_map[env.agent_dir] # Use this if the direction is not discarded
                    row_str += "A"
                    continue

                cell = env.grid.get(i, j)
                if cell is None:
                    row_str += "_"
                elif cell.type == "wall":
                    row_str += "#"
                elif cell.type == "goal":
                    row_str += "G"
                else:
                    row_str += "?"
            grid_repr.append(row_str)

        grid_str = "\n".join(grid_repr)

        # 3. Add the legend
        # legend = (
        #     "\n--- Legend ---\n> v < ^ : Agent\n# : Wall\nG : Goal\n---------------\n"
        # )
        legend = (
            "\n--- Legend ---\n A : Agent\n# : Wall\nG : Goal\n---------------\n"
        )

        # Combine the strings to create the full observation
        full_observation = mission + grid_str + legend
        return full_observation
