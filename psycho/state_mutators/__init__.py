from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from psycho.config import EnvironmentConfig


def build_state_mutator(config: EnvironmentConfig) -> MutatorSequence:
    orange_size = config.team_size if config.spawn_opponents else 0
    return MutatorSequence(
        FixedTeamSizeMutator(blue_size=config.team_size, orange_size=orange_size),
        KickoffMutator(),
    )


__all__ = ["build_state_mutator"]
