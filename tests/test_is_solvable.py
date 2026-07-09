"""Tests for the is_solvable function in environment_generator/utils.py"""

import pytest
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.utils import is_solvable


class TestIsSolvable:
    """Test suite for the is_solvable function"""

    def test_simple_solvable_path(self):
        """Test a simple straight-line path from agent to goal"""
        env = Simple2DNavigationEnv(size=5, render_mode=None)
        env.reset()

        # Create a simple solvable grid with a straight path
        grid = [
            ["#", "#", "#", "#", "#"],
            ["#", "A", "_", "_", "#"],
            ["#", "_", "_", "_", "#"],
            ["#", "_", "_", "G", "#"],
            ["#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is True

    def test_empty_room(self):
        """Test an empty room where agent can reach goal"""
        env = Simple2DNavigationEnv(size=7, render_mode=None)
        env.reset()

        # Empty room - definitely solvable
        grid = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", "A", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "G", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is True

    def test_unsolvable_blocked_goal(self):
        """Test when the goal is completely blocked by walls"""
        env = Simple2DNavigationEnv(size=7, render_mode=None)
        env.reset()

        # Goal completely surrounded by walls
        grid = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", "A", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "#", "#", "#", "#"],
            ["#", "_", "_", "#", "G", "#", "#"],
            ["#", "_", "_", "#", "#", "#", "#"],
            ["#", "_", "_", "_", "_", "_", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is False

    def test_unsolvable_blocked_agent(self):
        """Test when the agent is completely blocked by walls"""
        env = Simple2DNavigationEnv(size=7, render_mode=None)
        env.reset()

        # Agent completely surrounded by walls
        grid = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", "#", "#", "#", "_", "_", "#"],
            ["#", "#", "A", "#", "_", "_", "#"],
            ["#", "#", "#", "#", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "G", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is False

    def test_maze_with_path(self):
        """Test a maze with a complex but valid path"""
        env = Simple2DNavigationEnv(size=9, render_mode=None)
        env.reset()

        # Complex maze with a winding path
        grid = [
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", "A", "_", "#", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "#", "_", "#", "#", "_", "#"],
            ["#", "#", "_", "#", "_", "#", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "#", "_", "#", "#"],
            ["#", "_", "#", "#", "#", "#", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "_", "_", "_", "#"],
            ["#", "#", "#", "_", "#", "#", "#", "G", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is True

    def test_maze_without_path(self):
        """Test a maze with no valid path"""
        env = Simple2DNavigationEnv(size=9, render_mode=None)
        env.reset()

        # Maze with no path - wall separates agent from goal
        grid = [
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", "A", "_", "#", "_", "_", "_", "_", "#"],
            ["#", "_", "_", "#", "_", "#", "#", "_", "#"],
            ["#", "#", "_", "#", "_", "#", "_", "_", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", "_", "#", "#", "#", "#", "_", "_", "#"],
            ["#", "_", "_", "_", "_", "_", "_", "_", "#"],
            ["#", "#", "#", "_", "#", "#", "#", "G", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is False

    def test_agent_already_at_goal(self):
        """Test when agent starts at the goal position"""
        # This is an edge case - when agent and goal are at same position
        # The is_solvable function should return True immediately
        env = Simple2DNavigationEnv(
            size=5, render_mode=None, agent_start_pos=(2, 2), goal_pos=(2, 2)
        )
        env.reset()

        assert is_solvable(env) is True

    def test_adjacent_agent_and_goal(self):
        """Test when agent and goal are adjacent"""
        env = Simple2DNavigationEnv(size=5, render_mode=None)
        env.reset()

        # Agent and goal are neighbors
        grid = [
            ["#", "#", "#", "#", "#"],
            ["#", "A", "G", "_", "#"],
            ["#", "_", "_", "_", "#"],
            ["#", "_", "_", "_", "#"],
            ["#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is True

    def test_long_corridor(self):
        """Test a long corridor path"""
        env = Simple2DNavigationEnv(size=11, render_mode=None)
        env.reset()

        # Long corridor
        grid = [
            ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
            ["#", "A", "_", "_", "_", "_", "_", "_", "_", "G", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is True

    def test_u_shaped_path(self):
        """Test a U-shaped path"""
        env = Simple2DNavigationEnv(size=7, render_mode=None)
        env.reset()

        # U-shaped path
        grid = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", "A", "_", "_", "_", "G", "#"],
            ["#", "_", "#", "#", "#", "_", "#"],
            ["#", "_", "#", "#", "#", "_", "#"],
            ["#", "_", "_", "_", "_", "_", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert is_solvable(env) is True

    def test_randomly_generated_env_low_complexity(self):
        """Test with randomly generated environment with low complexity"""
        # Randomly generated environments should always be solvable
        env = Simple2DNavigationEnv(size=11, complexity=0.2, render_mode=None)
        env.reset()

        assert is_solvable(env) is True

    def test_randomly_generated_env_high_complexity(self):
        """Test with randomly generated environment with high complexity"""
        # Even high complexity randomly generated environments should be solvable
        env = Simple2DNavigationEnv(size=11, complexity=0.9, render_mode=None)
        env.reset()

        assert is_solvable(env) is True

    def test_multiple_random_environments(self):
        """Test that randomly generated environments are always solvable"""
        # Test multiple random environments with varying complexity
        for complexity in [0.0, 0.3, 0.5, 0.7, 1.0]:
            env = Simple2DNavigationEnv(
                size=11, complexity=complexity, render_mode=None
            )
            env.reset()
            result = is_solvable(env)
            assert result is True, (
                f"Randomly generated env with complexity={complexity} should be solvable"
            )

    def test_multiple_random_seeds(self):
        """Test that is_solvable returns True across multiple random seeds"""
        # Test 10 different random environments
        for _ in range(10):
            env = Simple2DNavigationEnv(size=9, complexity=0.5, render_mode=None)
            env.reset()
            assert is_solvable(env) is True, (
                "All randomly generated environments should be solvable"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
