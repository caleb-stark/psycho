import argparse
from collections.abc import Sequence
from pathlib import Path

from bot.runtime import select_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the checkpoint used by Psycho in RLBot.")
    parser.add_argument("checkpoint", help="A checkpoint directory or 'latest'.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/ppo"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = select_checkpoint(args.checkpoint, args.checkpoint_dir)
    print(f"Selected Psycho checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
