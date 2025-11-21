"""Tests for the goal_exists function in environment_generator/utils.py"""

import pytest
from reveng.environment_generator.custom_minigrid import Simple2DNavigationEnv
from reveng.environment_generator.utils import goal_exists


class TestGoalExists:
    """Test suite for the goal_exists function"""

    def test_goal_exists_in_random_env(self):
        """Test that a randomly generated environment has a goal"""
        env = Simple2DNavigationEnv(size=7, render_mode=None)
        env.reset()

        assert goal_exists(env) is True

    def test_goal_exists_in_custom_env(self):
        """Test that a custom environment with goal has goal_exists return True"""
        env = Simple2DNavigationEnv(size=5, render_mode=None)
        env.reset()

        grid = [
            ["#", "#", "#", "#", "#"],
            ["#", "A", "_", "_", "#"],
            ["#", "_", "_", "_", "#"],
            ["#", "_", "_", "G", "#"],
            ["#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert goal_exists(env) is True

    def test_goal_exists_in_custom_env_without_goal(self):
        """Test that a custom environment without goal has goal_exists return False"""
        env = Simple2DNavigationEnv(size=5, render_mode=None)
        env.reset()

        # Grid without a goal
        grid = [
            ["#", "#", "#", "#", "#"],
            ["#", "A", "_", "_", "#"],
            ["#", "_", "_", "_", "#"],
            ["#", "_", "_", "_", "#"],
            ["#", "#", "#", "#", "#"],
        ]
        env.set_env_from_list(grid)

        assert goal_exists(env) is False

    def test_goal_exists_with_specific_goal_position(self):
        """Test environment created with specific goal position"""
        env = Simple2DNavigationEnv(size=7, render_mode=None, goal_pos=(3, 3))
        env.reset()

        assert goal_exists(env) is True
        assert env.goal_pos == (3, 3)

    def test_goal_exists_multiple_environments(self):
        """Test that goal_exists works across multiple environment generations"""
        for _ in range(5):
            env = Simple2DNavigationEnv(size=9, complexity=0.5, render_mode=None)
            env.reset()
            assert goal_exists(env) is True

    def test_goal_exists_various_complexities(self):
        """Test that goals exist regardless of complexity level"""
        for complexity in [0.0, 0.3, 0.5, 0.7, 1.0]:
            env = Simple2DNavigationEnv(
                size=11, complexity=complexity, render_mode=None
            )
            env.reset()
            assert goal_exists(env) is True, (
                f"Goal should exist for complexity={complexity}"
            )

    def test_goal_exists_various_sizes(self):
        """Test that goals exist for various environment sizes"""
        for size in [5, 7, 9, 11, 15]:
            env = Simple2DNavigationEnv(size=size, render_mode=None)
            env.reset()
            assert goal_exists(env) is True, f"Goal should exist for size={size}"

    def test_goal_none_after_removal(self):
        """Test that goal_exists returns False when goal_pos is set to None"""
        env = Simple2DNavigationEnv(size=5, render_mode=None)
        env.reset()

        # Verify goal exists initially
        assert goal_exists(env) is True

        # Manually set goal_pos to None
        env.goal_pos = None

        # Now goal_exists should return False
        assert goal_exists(env) is False

    def test_goal_exists_empty_grid_with_goal(self):
        """Test goal exists in an empty room with goal"""
        env = Simple2DNavigationEnv(size=7, render_mode=None)
        env.reset()

        # Empty room with goal
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

        assert goal_exists(env) is True
        assert env.goal_pos is not None

    def test_goal_exists_agent_and_goal_same_position(self):
        """Test that goal exists when agent and goal are at same position"""
        env = Simple2DNavigationEnv(
            size=5, render_mode=None, agent_start_pos=(2, 2), goal_pos=(2, 2)
        )
        env.reset()

        assert goal_exists(env) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
