import gymnasium
from gymnasium.spaces import Text
from reveng.environment_generator.wrappers.fog_of_war import FogOfWarWrapper

# Default configuration for symbols. This can be overridden at initialization.
DEFAULT_SYMBOLS_CONFIG = {
    "agent": "A",
    "wall": "#",
    "goal": "G",
    "empty": " ",
    "unseen_obj": "?",  # For objects like keys, doors, etc.
    "fog": "*",  # Character for unseen areas in fog of war
}


class TextObsMixin:
    """
    A mixin class that provides methods to configure and render a MiniGrid
    environment as a text observation. It can optionally apply a fog-of-war mask.
    """

    def _setup_from_config(self, config=None):
        """
        Initializes symbols and legend from a configuration dictionary.
        """
        # Use default config if none is provided
        final_config = DEFAULT_SYMBOLS_CONFIG.copy()
        if config:
            final_config.update(config)

        self.symbols = final_config
        self._generate_legend()

    def _generate_legend(self):
        """Dynamically creates the legend string from the current symbols."""
        legend_items = [
            f"{symbol} : {name.replace('_', ' ').title()}"
            for name, symbol in self.symbols.items()
        ]

        legend_str = "\n".join(legend_items)
        self.legend = f"\n--- Legend ---\n{legend_str}\n---------------\n"

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
                    row_str += self.symbols["fog"]
                    continue

                # Check for agent position first
                if i == env.agent_pos[0] and j == env.agent_pos[1]:
                    row_str += self.symbols["agent"]
                    continue

                # Render the cell content
                cell = env.grid.get(i, j)
                if cell is None:
                    row_str += self.symbols["empty"]
                elif cell.type == "wall":
                    row_str += self.symbols["wall"]
                elif cell.type == "goal":
                    row_str += self.symbols["goal"]
                else:
                    # For other objects like keys, doors, etc.
                    row_str += self.symbols["unseen_obj"]
            grid_repr.append(row_str)

        grid_str = "\n".join(grid_repr)

        # 3. Combine all parts into the final observation string
        return mission + grid_str + self.legend


class FullObservabilityTextWrapper(gymnasium.ObservationWrapper, TextObsMixin):
    def __init__(self, env, config=None):
        super().__init__(env)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        # Setup symbols and legend from the configuration
        self._setup_from_config(config)

    def observation(self, obs):
        """Generates a fully observable text grid by calling the mixin method."""
        return self._render_text_observation()


class FogOfWarTextWrapper(FogOfWarWrapper, TextObsMixin):
    """
    Provides a text observation with a line-of-sight based "fog of war"
    effect. Inherits mask-calculation from FogOfWarWrapper and rendering
    from TextObsMixin.
    """

    def __init__(self, env, view_radius=None, config=None):
        super().__init__(env, view_radius=view_radius)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        # Setup symbols and legend from the configuration
        self._setup_from_config(config)

    def reset(self, **kwargs):
        """Resets the environment and ensures the first observation has fog."""
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        """Updates the mask and then renders the text grid with fog."""
        super().observation(obs)  # Updates self.seen_mask
        return self._render_text_observation(seen_mask=self.seen_mask)
