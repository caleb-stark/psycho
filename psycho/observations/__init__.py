from rlgym.rocket_league.obs_builders import DefaultObs

from psycho.config import EnvironmentConfig


def build_observation_builder(config: EnvironmentConfig) -> DefaultObs:
    """Build the normalized observation implementation shared with RLBot."""

    return DefaultObs(zero_padding=config.team_size)


__all__ = ["build_observation_builder"]
