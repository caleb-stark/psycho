from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward

from psycho.config import RewardWeights
from psycho.rewards.velocity import BallVelocityToGoalReward, VelocityToBallReward


def build_reward_function(weights: RewardWeights) -> CombinedReward:
    """Compose a deliberately small reward baseline with an outcome-heavy signal."""

    return CombinedReward(
        (GoalReward(), weights.goal),
        (TouchReward(), weights.touch),
        (VelocityToBallReward(), weights.velocity_to_ball),
        (BallVelocityToGoalReward(), weights.ball_velocity_to_goal),
    )


__all__ = [
    "BallVelocityToGoalReward",
    "VelocityToBallReward",
    "build_reward_function",
]
