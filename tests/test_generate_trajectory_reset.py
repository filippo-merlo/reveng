"""Tests for generate_trajectory reset behavior."""

from reveng.commands.get_trajectory import get_trajectory_utils


class FakeBaseEnv:
    """Minimal base env for reset-path tests."""

    def __init__(self):
        self.agent_pos = (1, 2)
        self.goal_pos = (3, 4)
        self.grid = object()
        self.safe_reset_calls = 0
        self.gen_obs_calls = 0

    def safe_reset(self):
        self.safe_reset_calls += 1

    def gen_obs(self):
        self.gen_obs_calls += 1
        return {"obs": "raw"}


class FakeWrappedEnv:
    """Minimal wrapper exposing the methods generate_trajectory uses."""

    def __init__(self):
        self.unwrapped = FakeBaseEnv()
        self.reset_calls = 0
        self.observation_calls = 0
        self.step_calls = 0
        self.grid = self.unwrapped.grid
        self.agent_pos = self.unwrapped.agent_pos
        self.goal_pos = self.unwrapped.goal_pos

    def reset(self):
        self.reset_calls += 1
        return "reset_obs", {}

    def observation(self, raw_obs):
        self.observation_calls += 1
        return f"wrapped:{raw_obs['obs']}"

    def step(self, action):
        self.step_calls += 1
        return "next_obs", 0.0, True, False, {}


class FakeAgent:
    """Placeholder agent; max_steps=0 means action generation is never used."""

    model_name = "fake/model"


def test_generate_trajectory_use_safe_reset_does_not_call_reset(monkeypatch):
    env = FakeWrappedEnv()

    monkeypatch.setattr(get_trajectory_utils, "get_astar_distance", lambda env, obs: 7)

    trajectory = get_trajectory_utils.generate_trajectory(
        env=env,
        agent=FakeAgent(),
        max_steps_per_trajectory=0,
        generation_kwargs={},
        use_safe_reset=True,
    )

    assert env.unwrapped.safe_reset_calls == 1
    assert env.reset_calls == 0
    assert env.unwrapped.gen_obs_calls == 1
    assert env.observation_calls == 1
    assert trajectory.traj_metadata["astar_distance"] == 7


def test_generate_trajectory_skip_reset_uses_current_state(monkeypatch):
    env = FakeWrappedEnv()

    monkeypatch.setattr(get_trajectory_utils, "get_astar_distance", lambda env, obs: 5)

    get_trajectory_utils.generate_trajectory(
        env=env,
        agent=FakeAgent(),
        max_steps_per_trajectory=0,
        generation_kwargs={},
        use_safe_reset=True,
        skip_reset=True,
    )

    assert env.unwrapped.safe_reset_calls == 0
    assert env.reset_calls == 0
    assert env.unwrapped.gen_obs_calls == 1
    assert env.observation_calls == 1


def test_generate_trajectory_regular_reset_path(monkeypatch):
    env = FakeWrappedEnv()

    monkeypatch.setattr(get_trajectory_utils, "get_astar_distance", lambda env, obs: 3)

    get_trajectory_utils.generate_trajectory(
        env=env,
        agent=FakeAgent(),
        max_steps_per_trajectory=0,
        generation_kwargs={},
        use_safe_reset=False,
    )

    assert env.unwrapped.safe_reset_calls == 0
    assert env.reset_calls == 1
    assert env.unwrapped.gen_obs_calls == 0
    assert env.observation_calls == 0
