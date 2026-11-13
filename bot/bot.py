import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rlgym_rlbot import RLGymBot, RLGymBotConfig

from bot.runtime import selected_checkpoint
from psycho.actions import build_action_parser
from psycho.checkpoints import load_manifest
from psycho.conditions import build_termination_condition, build_truncation_condition
from psycho.config import environment_config_from_dict
from psycho.models import DiscretePolicy
from psycho.observations import build_observation_builder
from psycho.rewards import build_reward_function
from psycho.state_mutators import build_state_mutator

AGENT_ID = "caleb-stark/psycho"


class PsychoBot(RLGymBot):
    """Thin RLBot v5 host for Psycho's shared RLGym components and policy."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        manifest = load_manifest(checkpoint)
        environment_config = environment_config_from_dict(manifest["environment"])
        self.policy = DiscretePolicy.from_checkpoint(checkpoint, device=device)
        self.observation_size = int(manifest["observation_size"])
        self.checkpoint = checkpoint

        super().__init__(
            obs_builder=build_observation_builder(environment_config),
            action_parser=build_action_parser(environment_config),
            reward_function=build_reward_function(environment_config.reward_weights),
            termination_condition=build_termination_condition(),
            truncation_condition=build_truncation_condition(environment_config),
            state_mutator=build_state_mutator(environment_config),
            default_agent_id=AGENT_ID,
            config=RLGymBotConfig(),
        )

    def get_action(self, obs, game_state, packet) -> np.ndarray:
        del game_state, packet
        observation = np.asarray(obs, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"expected observation shape {(self.observation_size,)}, got {observation.shape}; "
                "run a match with the team size used by this checkpoint"
            )
        action = self.policy.act(observation, deterministic=True)
        return np.asarray([action], dtype=np.int64)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Psycho through RLBot v5.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ppo"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = selected_checkpoint(args.checkpoint, args.checkpoint_dir)
    PsychoBot(checkpoint, device=args.device).run(
        wants_match_communications=False,
        wants_ball_predictions=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
