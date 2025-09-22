from gymnasium import spaces
from gymnasium.core import ObservationWrapper
import numpy as np


class OmnidirectionalFogOfWarRGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper that provides a fully-observable RGB image of the grid with a
    "fog of war" effect. The agent has an omnidirectional view, revealing a
    square area around it, regardless of its facing direction.

    #TODO: account for walls blocking sight
    """

    def __init__(self, env, tile_size=8, fog_color=100, view_radius=None):
        """
        Args:
            env: The MiniGrid environment to wrap.
            tile_size (int): The number of pixels for each grid cell.
            fog_color (int or tuple): The color of the fog.
            view_radius (int): The radius of the agent's square view.
                               A radius of 1 means a 3x3 view. If None,
                               it defaults to half the agent's original
                               triangular view size.
        """
        super().__init__(env)

        self.tile_size = tile_size
        if isinstance(fog_color, int):
            self.fog_color = (fog_color, fog_color, fog_color)
        else:
            self.fog_color = fog_color

        if view_radius is None:
            # Default radius is based on the env's default triangular view size.
            # For agent_view_size=7, this results in a radius of 3 (a 7x7 square).
            self.view_radius = (self.unwrapped.agent_view_size - 1) // 2
        else:
            self.view_radius = view_radius

        # The observation space is a full-sized RGB image
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.unwrapped.height * tile_size,
                self.unwrapped.width * tile_size,
                3,
            ),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

        self.seen_mask = None

    def reset(self, **kwargs):
        """
        Resets the environment and the seen mask.
        """
        obs, info = self.env.reset(**kwargs)
        self.seen_mask = np.zeros(
            (self.unwrapped.height, self.unwrapped.width), dtype=bool
        )
        return self.observation(obs), info

    def observation(self, obs):
        """
        Processes the observation to update the seen mask and apply the fog of war.
        """
        # 1. Update the cumulative seen mask with an omnidirectional view
        self._update_seen_mask()

        # 2. Get the full, unmasked RGB frame
        full_rgb_img = self.unwrapped.get_frame(
            highlight=False, tile_size=self.tile_size
        )

        # 3. Apply the fog of war
        masked_img = self._apply_fog(full_rgb_img)

        # 4. Return the modified observation
        return {**obs, "image": masked_img}

    def _update_seen_mask(self):
        """
        Updates the seen_mask based on a square area around the agent.
        This method ignores walls and provides a simple omnidirectional view.
        """
        agent_x, agent_y = self.unwrapped.agent_pos

        # Define the bounds of the square view, clamping to the grid edges
        min_x = max(0, agent_x - self.view_radius)
        max_x = min(self.unwrapped.width, agent_x + self.view_radius + 1)
        min_y = max(0, agent_y - self.view_radius)
        max_y = min(self.unwrapped.height, agent_y + self.view_radius + 1)

        # Use efficient numpy slicing to mark the entire area as seen
        self.seen_mask[min_y:max_y, min_x:max_x] = True

    def _apply_fog(self, rgb_img):
        """
        Applies the fog of war to a full RGB image.
        """
        pixel_seen_mask = np.kron(
            self.seen_mask, np.ones((self.tile_size, self.tile_size), dtype=bool)
        )
        pixel_seen_mask_3d = np.stack([pixel_seen_mask] * 3, axis=-1)
        fog_img = np.full_like(rgb_img, self.fog_color)
        masked_img = np.where(pixel_seen_mask_3d, rgb_img, fog_img)
        return masked_img.astype(np.uint8)
