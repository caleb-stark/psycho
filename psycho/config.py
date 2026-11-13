from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Small baseline shaping terms; scoring remains the dominant objective."""

    goal: float = 10.0
    touch: float = 0.1
    velocity_to_ball: float = 0.05
    ball_velocity_to_goal: float = 0.1


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Settings shared by RocketSim training and RLBot inference."""

    team_size: int = 1
    spawn_opponents: bool = True
    action_repeat: int = 8
    no_touch_timeout_seconds: float = 30.0
    episode_timeout_seconds: float = 300.0
    rlbot_delay: bool = True
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self) -> None:
        if self.team_size < 1:
            raise ValueError("team_size must be at least 1")
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be at least 1")
        if self.no_touch_timeout_seconds <= 0:
            raise ValueError("no_touch_timeout_seconds must be positive")
        if self.episode_timeout_seconds <= 0:
            raise ValueError("episode_timeout_seconds must be positive")


DEFAULT_ENVIRONMENT = EnvironmentConfig()
