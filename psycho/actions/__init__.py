from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction

from psycho.config import EnvironmentConfig


def build_action_parser(config: EnvironmentConfig) -> RepeatAction:
    """Build the discrete parser used without modification in training and live play."""

    return RepeatAction(LookupTableAction(), repeats=config.action_repeat)


def lookup_table_size() -> int:
    return len(LookupTableAction.make_lookup_table())


__all__ = ["build_action_parser", "lookup_table_size"]
