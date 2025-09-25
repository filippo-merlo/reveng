import gymnasium
from gymnasium.spaces import Text
from reveng.environment_generator.wrappers.fog_of_war import FogOfWarWrapper


class TextObsMixin:
    """
    A mixin class that provides a method to render a MiniGrid environment
    as a text observation. It can optionally apply a fog-of-war mask.
    """

    def _render_text_observation(self, seen_mask=None):
        """
        Generates the text observation from the environment's grid state.
        If a seen_mask is provided, unseen cells are replaced with fog.
        """
        env = self.unwrapped

        # 1. Start with the mission description
        mission = f"Mission: {env.mission}\n"

        # 2. Create a character grid
        grid_repr = []
        for j in range(env.height):
            row_str = ""
            for i in range(env.width):
                # If a mask exists and the cell is not visible, apply fog
                if seen_mask is not None and not seen_mask[j, i]:
                    row_str += self.fog_char
                    continue

                # Check for agent position first
                if i == env.agent_pos[0] and j == env.agent_pos[1]:
                    row_str += "A"
                    continue

                # Render the cell content
                cell = env.grid.get(i, j)
                if cell is None:
                    row_str += "_"
                elif cell.type == "wall":
                    row_str += "#"
                elif cell.type == "goal":
                    row_str += "G"
                else:
                    row_str += "?"  # For other objects like keys, doors, etc.
            grid_repr.append(row_str)

        grid_str = "\n".join(grid_repr)

        # 3. Combine all parts into the final observation string
        # The 'self.legend' attribute is expected to be defined by the class using this mixin.
        return mission + grid_str + self.legend


class FullObservabilityTextWrapper(gymnasium.ObservationWrapper, TextObsMixin):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        self.legend = "\n--- Legend ---\nA : Agent\n# : Wall\nG : Goal\n_ : Empty Floor\n---------------\n"

    def observation(self, obs):
        """Generates a fully observable text grid by calling the mixin method."""
        # Call the shared renderer without a mask to show everything.
        return self._render_text_observation()


class FogOfWarTextWrapper(FogOfWarWrapper, TextObsMixin):
    """
    Provides a text observation with a line-of-sight based "fog of war"
    effect. Inherits mask-calculation from FogOfWarWrapper and rendering
    from TextObsMixin.
    """

    def __init__(self, env, view_radius=None, fog_char="?"):
        super().__init__(env, view_radius=view_radius)
        self.fog_char = fog_char
        self.observation_space = Text(max_length=4096, charset="utf-8")
        self.legend = "\n--- Legend ---\nA : Agent\n# : Wall\nG : Goal\n_ : Empty Floor\n---------------\n"

    def reset(self, **kwargs):
        """Resets the environment and ensures the first observation has fog."""
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        """Updates the mask and then renders the text grid with fog."""
        # 1. Update self.seen_mask from the FogOfWarWrapper parent
        super().observation(obs)

        # 2. Call the shared renderer, passing the updated mask
        return self._render_text_observation(seen_mask=self.seen_mask)
