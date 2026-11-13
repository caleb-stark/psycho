from typing import Any

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    BALL_MAX_SPEED,
    BLUE_GOAL_BACK,
    BLUE_TEAM,
    CAR_MAX_SPEED,
    ORANGE_GOAL_BACK,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1e-8:
        return np.zeros_like(vector)
    return vector / magnitude


class VelocityToBallReward(RewardFunction[AgentID, GameState, float]):
    """Reward the component of controlled-car velocity directed at the ball."""

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        del agents, initial_state, shared_info

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        del is_terminated, is_truncated, shared_info
        rewards: dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            direction = _unit(state.ball.position - car.physics.position)
            rewards[agent] = float(np.dot(car.physics.linear_velocity, direction) / CAR_MAX_SPEED)
        return rewards


class BallVelocityToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Reward ball velocity toward each agent's opponent goal."""

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        del agents, initial_state, shared_info

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        del is_terminated, is_truncated, shared_info
        rewards: dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            goal = ORANGE_GOAL_BACK if car.team_num == BLUE_TEAM else BLUE_GOAL_BACK
            direction = _unit(np.asarray(goal, dtype=np.float32) - state.ball.position)
            rewards[agent] = float(np.dot(state.ball.linear_velocity, direction) / BALL_MAX_SPEED)
        return rewards
