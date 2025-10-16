import json
from pathlib import Path

import gymnasium
from gymnasium.spaces import Text

from reveng.environment_generator.wrappers.fog_of_war import FogOfWarWrapper

# Default configuration for symbols. This can be overridden at initialization.
DEFAULT_SYMBOLS_CONFIG = {
    "agent": "A",
    "wall": "#",
    "goal": "G",
    "empty": "_",
    "unknown_obj": "?",  # For objects like keys, doors, etc.
    "fog": "*",  # Character for unseen areas in fog of war
    # "path": "P",  # Optional: previously visited path (not used by renderer)
}


class TextObsMixin:
    """
    A mixin that provides methods to configure and render a MiniGrid environment
    as a text observation that aligns with the template formats:
      - Full observability: grid with coordinates and space-separated symbols
      - Partial observability: same format, but hidden cells rendered as '*'
    """

    def _setup_from_config(self, config_path=None):
        """
        Initializes symbols from a configuration dictionary or JSON file.
        """
        final_config = DEFAULT_SYMBOLS_CONFIG.copy()

        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as f:
                file_config = json.load(f)
                final_config.update(file_config)

        self.symbols = final_config

    def _get_cell_type_at_position(self, env, seen_mask, i, j):
        """
        Returns the logical cell type at (i, j) accounting for optional fog mask.
        Order of precedence:
          1) Agent position is always visible and rendered as 'agent'
          2) If seen_mask is provided and cell is not seen -> 'fog'
          3) Otherwise render the underlying cell (wall/goal/empty/unknown)
        """
        # Agent always visible
        if i == env.agent_pos[0] and j == env.agent_pos[1]:
            return "agent"

        # Apply fog if not visible
        if seen_mask is not None and not seen_mask[j, i]:
            return "fog"

        # Render underlying cell
        cell = env.grid.get(i, j)
        if cell is None:
            return "empty"
        elif cell.type in self.symbols:
            return cell.type
        else:
            return "unknown_obj"

    def _render_grid_with_coordinates(self, seen_mask=None):
        """
        Renders a whitespace-separated grid with row/column coordinates,
        matching the formats shown in:
          - grid_full_observability.j2 (when seen_mask is None)
          - grid_partial_observability.j2 (when seen_mask is provided)

        Example layout:
          <row/col header>
        0 <row 0>
        1 <row 1>
        ...
        """
        env = self.unwrapped

        # Determine widths for index alignment based on max index value
        row_idx_width = max(1, len(str(env.height - 1)))
        col_idx_width = max(1, len(str(env.width - 1)))

        # Header with column indices
        col_header = " ".join(str(i).rjust(col_idx_width) for i in range(env.width))
        header = " " * (row_idx_width + 1) + col_header

        lines = [header]

        # Rows with row index and spaced symbols
        for j in range(env.height):
            cells = []
            for i in range(env.width):
                cell_type = self._get_cell_type_at_position(env, seen_mask, i, j)
                symbol = self.symbols[cell_type]
                cells.append(symbol.rjust(col_idx_width))
            row_str = f"{str(j).rjust(row_idx_width)} " + " ".join(cells)
            lines.append(row_str)

        return "\n".join(lines)


class FullObservabilityTextWrapper(gymnasium.ObservationWrapper, TextObsMixin):
    """
    Observation wrapper that returns a fully observable grid formatted to align
    with grid_full_observability.j2:
      - Includes a header row with column indices
      - Each row begins with the row index
      - Cells are space-separated symbols
      - No mission or legend text is included
    """

    def __init__(self, env, config_path=None):
        super().__init__(env)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        self._setup_from_config(config_path)

    def observation(self, obs):
        return self._render_grid_with_coordinates(seen_mask=None)


class FogOfWarTextWrapper(FogOfWarWrapper, TextObsMixin):
    """
    Observation wrapper that returns a partially observable grid formatted to align
    with grid_partial_observability.j2:
      - Same coordinate and spacing format as the full observability case
      - Hidden/unseen cells are rendered as '*' (fog)
      - No mission or legend text is included
    """

    def __init__(self, env, view_radius=None, config_path=None):
        super().__init__(env, view_radius=view_radius)
        self.observation_space = Text(max_length=4096, charset="utf-8")
        self._setup_from_config(config_path)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        # Updates self.seen_mask in the parent wrapper
        super().observation(obs)
        return self._render_grid_with_coordinates(seen_mask=self.seen_mask)


class LoggingFogOfWarTextWrapper(FogOfWarTextWrapper):
    """
    Extends FogOfWarTextWrapper by logging both partial and full observability
    grid strings as well as the underlying cell types at each step.
    """

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

        # Fully observable snapshot for logging
        fo_obs = self._render_grid_with_coordinates(seen_mask=None)
        self.fully_observable_observation_log.append(fo_obs)

        self.save_cell_types()
        return obs
