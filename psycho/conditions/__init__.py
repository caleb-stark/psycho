from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)

from psycho.config import EnvironmentConfig


def build_termination_condition() -> GoalCondition:
    return GoalCondition()


def build_truncation_condition(config: EnvironmentConfig) -> AnyCondition:
    return AnyCondition(
        NoTouchTimeoutCondition(config.no_touch_timeout_seconds),
        TimeoutCondition(config.episode_timeout_seconds),
    )


__all__ = ["build_termination_condition", "build_truncation_condition"]
