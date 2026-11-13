from copy import deepcopy

import numpy as np
import pytest
from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT, TICKS_PER_SECOND

from psycho.actions import lookup_table_size
from psycho.config import EnvironmentConfig
from training.environment import build_ppo_environment, build_rlgym_environment


@pytest.fixture
def environment():
    env = build_rlgym_environment()
    try:
        yield env
    finally:
        env.close()


def zero_actions(env) -> dict[str, np.ndarray]:
    return {agent: np.asarray([0], dtype=np.int64) for agent in env.agents}


def test_reset_builds_normalized_finite_1v1_observations(environment) -> None:
    observations = environment.reset()

    assert set(observations) == {"blue-0", "orange-0"}
    assert {car.team_num for car in environment.state.cars.values()} == {0, 1}
    assert environment.state.ball.position[2] == pytest.approx(BALL_RESTING_HEIGHT, abs=0.1)
    assert environment.state.ball.position[:2] == pytest.approx((0.0, 0.0))
    assert {observation.shape for observation in observations.values()} == {(92,)}
    assert all(np.isfinite(observation).all() for observation in observations.values())


def test_discrete_action_space_and_step_are_valid(environment) -> None:
    environment.reset()
    initial_tick = environment.state.tick_count
    action_spaces = environment.action_spaces

    assert lookup_table_size() == 90
    assert all(space == ("discrete", 90) for space in action_spaces.values())

    observations, rewards, terminated, truncated = environment.step(zero_actions(environment))

    assert environment.state.tick_count - initial_tick == 8
    assert all(np.isfinite(observation).all() for observation in observations.values())
    assert all(np.isfinite(reward) for reward in rewards.values())
    assert not any(terminated.values())
    assert not any(truncated.values())


def test_goal_and_timeout_conditions(environment) -> None:
    environment.reset()
    goal_state = deepcopy(environment.state)
    goal_state.goal_scored = True
    goal_state.ball.position[1] = 5300

    goal_done = environment.termination_cond.is_done(
        list(environment.agents), goal_state, environment.shared_info
    )
    assert all(goal_done.values())

    short_config = EnvironmentConfig(
        no_touch_timeout_seconds=10.0,
        episode_timeout_seconds=8 / TICKS_PER_SECOND,
    )
    short_env = build_rlgym_environment(short_config)
    try:
        short_env.reset()
        _, _, _, truncated = short_env.step(zero_actions(short_env))
        assert all(truncated.values())
    finally:
        short_env.close()


def test_multiple_rocketsim_environments_are_independent() -> None:
    environments = [build_rlgym_environment(), build_rlgym_environment()]
    try:
        for environment in environments:
            environment.reset()
        environments[0].step(zero_actions(environments[0]))

        assert environments[0].state.tick_count == 8
        assert environments[1].state.tick_count == 0
    finally:
        for environment in environments:
            environment.close()


def test_rlgym_ppo_wrapper_resets_and_steps() -> None:
    environment = build_ppo_environment()
    try:
        observations = environment.reset()
        actions = np.zeros((len(observations), 1), dtype=np.int64)
        next_observations, rewards, terminated, truncated, info = environment.step(actions)

        assert observations.shape == (2, 92)
        assert next_observations.shape == (2, 92)
        assert np.isfinite(next_observations).all()
        assert all(np.isfinite(reward) for reward in rewards)
        assert not terminated
        assert not truncated
        assert info["state"].tick_count > 0
    finally:
        environment.close()
