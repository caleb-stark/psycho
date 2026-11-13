import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from rlgym.rocket_league.common_values import BLUE_TEAM, TICKS_PER_SECOND

from psycho.checkpoints import load_manifest, resolve_checkpoint
from psycho.config import environment_config_from_dict
from psycho.models import DiscretePolicy
from training.environment import build_rlgym_environment


@dataclass(slots=True)
class EvaluationSummary:
    checkpoint: str
    episodes: int
    wins: int
    losses: int
    draws: int
    goals: int
    goals_conceded: int
    touches: int
    average_reward: float
    average_episode_seconds: float


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def evaluate_checkpoint(
    checkpoint: Path,
    episodes: int,
    *,
    deterministic: bool = True,
    device: str = "auto",
    seed: int = 123,
) -> EvaluationSummary:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    manifest = load_manifest(checkpoint)
    environment_config = environment_config_from_dict(manifest["environment"])
    policy = DiscretePolicy.from_checkpoint(checkpoint, device=select_device(device))
    environment = build_rlgym_environment(environment_config)

    wins = losses = draws = goals = goals_conceded = touches = 0
    total_reward = 0.0
    total_decisions = 0
    total_ticks = 0
    controlled_agent = "blue-0"

    try:
        for _ in range(episodes):
            observations = environment.reset()
            episode_start_tick = environment.state.tick_count

            while True:
                actions = {
                    agent: np.asarray(
                        [policy.act(observation, deterministic=deterministic)],
                        dtype=np.int64,
                    )
                    for agent, observation in observations.items()
                }
                observations, rewards, terminated, truncated = environment.step(actions)
                total_reward += float(rewards[controlled_agent])
                total_decisions += 1
                touches += environment.state.cars[controlled_agent].ball_touches

                if any(terminated.values()) or any(truncated.values()):
                    total_ticks += environment.state.tick_count - episode_start_tick
                    if environment.state.goal_scored:
                        if environment.state.scoring_team == BLUE_TEAM:
                            wins += 1
                            goals += 1
                        else:
                            losses += 1
                            goals_conceded += 1
                    else:
                        draws += 1
                    break
    finally:
        environment.close()

    return EvaluationSummary(
        checkpoint=str(checkpoint),
        episodes=episodes,
        wins=wins,
        losses=losses,
        draws=draws,
        goals=goals,
        goals_conceded=goals_conceded,
        touches=touches,
        average_reward=total_reward / max(1, total_decisions),
        average_episode_seconds=total_ticks / TICKS_PER_SECOND / episodes,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Psycho checkpoint without learning.")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ppo"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    summary = evaluate_checkpoint(
        checkpoint,
        args.episodes,
        deterministic=not args.stochastic,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
