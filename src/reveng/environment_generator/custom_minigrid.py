import random
from enum import IntEnum

from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class Simple2DNavigationEnv(MiniGridEnv):
    # https://minigrid.farama.org/content/create_env_tutorial/
    class Actions(IntEnum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    class ExtendedActions(IntEnum):
        # https://docs.python.org/3/howto/enum.html#restricted-enum-subclassing
        # Creating a subclass of an Enum with numbers is not possible
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3
        QUIT = 4

    def __init__(
        self,
        size=11,
        complexity=0.0,  # 0.0: empty room, 1.0: perfect maze
        agent_start_dir: int | None = None,
        agent_start_pos: tuple[int, int] | None = None,
        goal_pos: tuple[int, int] | None = None,
        max_steps: int | None = None,
        allow_quit_action: bool = False,
        **kwargs,
    ):
        self.complexity = max(0.0, min(1.0, complexity))  # Clamp between 0 and 1
        self.agent_start_pos_user = agent_start_pos
        self.agent_start_dir_user = agent_start_dir
        self.goal_pos_user = goal_pos
        self.allow_quit_action = allow_quit_action

        if size % 2 == 0:
            size += 1
            print(f"Grid size must be odd for maze generation. Adjusting to {size}.")

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        if self.allow_quit_action:
            self.actions = Simple2DNavigationEnv.ExtendedActions
        else:
            self.actions = Simple2DNavigationEnv.Actions

        self.action_space = spaces.Discrete(len(self.actions))

        self._action_to_direction = {
            self.actions.LEFT: 2,
            self.actions.RIGHT: 0,
            self.actions.UP: 3,
            self.actions.DOWN: 1,
        }

    @staticmethod
    def _gen_mission():
        return "Reach the Goal"

    def _generate_perfect_maze(self, width, height):
        """Generates a perfect maze using randomized Depth-First Search."""
        # Start with a grid full of walls
        for i in range(width):
            for j in range(height):
                self.grid.set(i, j, Wall())

        start_x, start_y = (
            random.randint(0, (width - 3) // 2) * 2 + 1,
            random.randint(0, (height - 3) // 2) * 2 + 1,
        )

        stack = [(start_x, start_y)]
        self.grid.set(start_x, start_y, None)

        while stack:
            cx, cy = stack[-1]
            neighbors = []

            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 < nx < width - 1
                    and 0 < ny < height - 1
                    and self.grid.get(nx, ny)
                    and self.grid.get(nx, ny).type == "wall"
                ):
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = random.choice(neighbors)
                self.grid.set(nx, ny, None)
                self.grid.set((cx + nx) // 2, (cy + ny) // 2, None)
                stack.append((nx, ny))
            else:
                stack.pop()

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # 1. Generate a perfect, complex maze
        self._generate_perfect_maze(width, height)

        # 2. Identify all internal walls that can be removed
        internal_walls = []
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if self.grid.get(i, j) is not None:
                    internal_walls.append((i, j))

        random.shuffle(internal_walls)

        # 3. Calculate how many walls to remove
        num_walls_to_remove = int((1.0 - self.complexity) * len(internal_walls))

        # 4. Remove the walls
        for i in range(num_walls_to_remove):
            self.grid.set(internal_walls[i][0], internal_walls[i][1], None)

        # 5. Place Agent and Goal in valid empty spaces
        empty_cells = []
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if self.grid.get(i, j) is None:
                    empty_cells.append((i, j))

        if len(empty_cells) < 2:
            raise Exception("Maze generation failed, not enough empty cells.")

        # Place Goal
        self.goal_pos = (
            self.goal_pos_user
            if self.goal_pos_user in empty_cells
            else random.choice(empty_cells)
        )
        self.put_obj(Goal(), *self.goal_pos)
        empty_cells.remove(self.goal_pos)

        # Place Agent
        self.agent_pos = (
            self.agent_start_pos_user
            if self.agent_start_pos_user in empty_cells
            else random.choice(empty_cells)
        )
        self.agent_dir = (
            self.agent_start_dir_user
            if self.agent_start_dir_user is not None
            else random.randint(0, 3)
        )

        self.mission = "Reach the Goal"

    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # Handle QUIT action
        if (
            hasattr(self, "allow_quit_action")
            and self.allow_quit_action
            and action == self.ExtendedActions.QUIT
        ):
            terminated = True
            obs = self.gen_obs()
            return obs, reward, terminated, truncated, {}

        # Set agent's direction based on the action
        if action in self._action_to_direction:
            self.agent_dir = self._action_to_direction[action]
        else:
            raise ValueError(f"Unknown action: {action}")

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Move if the cell is empty or can be overlapped
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)

        if self.agent_pos == self.goal_pos:
            terminated = True
            reward = self._reward()

        # Check if the episode is truncated
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    print("--- Low complexity maze (complexity=0.2) ---")
    low_complexity_env = Simple2DNavigationEnv(
        size=21, complexity=0.2, render_mode="human"
    )
    low_complexity_env.reset()
    low_complexity_env.render()
    import time

    time.sleep(3)
    low_complexity_env.close()

    print("--- Highest complexity maze (complexity=1.0) ---")
    low_complexity_env = Simple2DNavigationEnv(
        size=21, complexity=1.0, render_mode="human"
    )
    low_complexity_env.reset()
    low_complexity_env.render()
    import time

    time.sleep(3)
    low_complexity_env.close()
