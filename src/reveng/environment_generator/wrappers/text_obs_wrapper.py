import json
import numpy as np
from pathlib import Path

import gymnasium
from gymnasium.spaces import Text
from reveng.environment_generator.wrappers.fog_of_war import FogOfWarWrapper

# Default configuration for symbols. This can be overridden at initialization.
DEFAULT_SYMBOLS_CONFIG = {
    "agent": "A",
    "wall": "#",
    "goal": "G",
    "empty": " ",
    "unknown_obj": "?",  # For objects like keys, doors, etc.
    "fog": "*",  # Character for unseen areas in fog of war
}


class TextObsMixin:
    """
    A mixin class that provides methods to configure and render a MiniGrid
    environment as a text observation. It can optionally apply a fog-of-war mask.
    """

    def _setup_from_config(self, config_path=None):
        """
        Initializes symbols and legend from a configuration dictionary.
        """
        # Use default config if none is provided
        final_config = DEFAULT_SYMBOLS_CONFIG.copy()

        # Load from JSON file if provided
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as f:
                file_config = json.load(f)
                final_config.update(file_config)

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

    def _get_cell_type_at_position(self, env, seen_mask, i, j):
        # If a mask exists and the cell is not visible, apply fog
        if seen_mask is not None and not seen_mask[j, i]:
            return "fog"

        # Check for agent position first
        if i == env.agent_pos[0] and j == env.agent_pos[1]:
            return "agent"

        # Render the cell content
        cell = env.grid.get(i, j)
        if cell is None:
            return "empty"
        elif cell.type in self.symbols:
            return cell.type
        else:
            # For other objects like keys, doors, etc.
            return "unknown_obj"

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
                cell_type = self._get_cell_type_at_position(env, seen_mask, i, j)
                symbol = self.symbols[cell_type]
                row_str += symbol

            grid_repr.append(row_str)

        grid_str = "\n".join(grid_repr)

        # 3. Combine all parts into the final observation string
        return mission + grid_str + self.legend


class FullObservabilityTextWrapper(gymnasium.ObservationWrapper, TextObsMixin):
    def __init__(self, env, config_path=None):
        super().__init__(env)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        # Setup symbols and legend from the configuration
        self._setup_from_config(config_path)

    def observation(self, obs):
        """Generates a fully observable text grid by calling the mixin method."""
        return self._render_text_observation()


class FogOfWarTextWrapper(FogOfWarWrapper, TextObsMixin):
    """
    Provides a text observation with a line-of-sight based "fog of war"
    effect. Inherits mask-calculation from FogOfWarWrapper and rendering
    from TextObsMixin.
    """

    def __init__(self, env, view_radius=None, config_path=None):
        # FogOfWarWrapper.__init__(self, env, view_radius=view_radius)
        super().__init__(env, view_radius=view_radius)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        # Setup symbols and legend from the configuration
        self._setup_from_config(config_path)

    def reset(self, **kwargs):
        """Resets the environment and ensures the first observation has fog."""
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        """Updates the mask and then renders the text grid with fog."""
        super().observation(obs)  # Updates self.seen_mask
        return self._render_text_observation(seen_mask=self.seen_mask)


class LoggingFogOfWarTextWrapper(FogOfWarTextWrapper):
    def __init__(self, env, view_radius=None, config_path=None):
        super().__init__(env, view_radius=view_radius, config_path=config_path)
        self.partially_observable_observation_log = []
        self.fully_observable_observation_log = []
        self.partially_observable_cell_type_log = []
        self.fully_observable_cell_type_log = []

    def save_cell_types(self):
        po_cell_types = []  # list of tuples (x, y, cell_type)
        fo_cell_types = []  # list of tuples (x, y, cell_type)
        height = self.unwrapped.height
        width = self.unwrapped.width
        for j in range(height):
            for i in range(width):
                po_cell_type = self._get_cell_type_at_position(
                    self.unwrapped, self.seen_mask, i, j
                )
                po_cell_types.append((i, j, po_cell_type))
                fo_cell_type = self._get_cell_type_at_position(
                    self.unwrapped, None, i, j
                )
                fo_cell_types.append((i, j, fo_cell_type))
        self.partially_observable_cell_type_log.append(repr(po_cell_types))
        self.fully_observable_cell_type_log.append(repr(fo_cell_types))

    def observation(self, obs):
        obs = super().observation(obs)
        self.partially_observable_observation_log.append(obs)
        fully_observable_mask = np.ones(
            (self.unwrapped.height, self.unwrapped.width), dtype=bool
        )
        fo_obs = self._render_text_observation(seen_mask=fully_observable_mask)
        self.fully_observable_observation_log.append(fo_obs)
        self.save_cell_types()
        return obs
