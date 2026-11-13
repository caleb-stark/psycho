from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym_ppo.util import RLGymV2GymWrapper

from psycho.actions import build_action_parser
from psycho.conditions import build_termination_condition, build_truncation_condition
from psycho.config import DEFAULT_ENVIRONMENT, EnvironmentConfig
from psycho.observations import build_observation_builder
from psycho.rewards import build_reward_function
from psycho.state_mutators import build_state_mutator


def build_rlgym_environment(config: EnvironmentConfig = DEFAULT_ENVIRONMENT) -> RLGym:
    """Create one 1v1-capable RLGym v2 environment backed by RocketSim."""

    return RLGym(
        state_mutator=build_state_mutator(config),
        obs_builder=build_observation_builder(config),
        action_parser=build_action_parser(config),
        reward_fn=build_reward_function(config.reward_weights),
        termination_cond=build_termination_condition(),
        truncation_cond=build_truncation_condition(config),
        transition_engine=RocketSimEngine(rlbot_delay=config.rlbot_delay),
    )


def build_ppo_environment(config: EnvironmentConfig = DEFAULT_ENVIRONMENT) -> RLGymV2GymWrapper:
    """Adapt the shared v2 environment to the interface expected by RLGym-PPO."""

    return RLGymV2GymWrapper(build_rlgym_environment(config))


__all__ = ["build_ppo_environment", "build_rlgym_environment"]
