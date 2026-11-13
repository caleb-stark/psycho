# Psycho

Psycho is a Rocket League machine-learning bot built around RLGym v2,
RocketSim, PPO, and RLBot v5. The project is being rebuilt from its original
rule-based RLBot v4/Gosling implementation.

The core design rule is that training and live play import the same RLGym
observation builder and action parser. The RLBot process does not reproduce
training observations by hand.

## Status

The repository currently contains the clean Python project foundation, pinned
ecosystem dependencies, and a tested 1v1 RLGym v2 environment backed by
RocketSim. PPO training, evaluation, checkpoints, and live-bot workflows are
documented only when verified.

## Requirements

- Windows or Linux for headless RocketSim training
- CPython 3.11.9 (the final Python 3.11 release with an official Windows
  installer)
- `uv` is recommended; standard `venv` and `pip` also work
- Rocket League is required only for later RLBot live play

## Development setup

With `uv`:

```powershell
uv python install 3.11.9
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

With the standard library on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

The environment tests construct, reset, and step RocketSim, exercise two
simultaneous environments, validate finite observations and rewards, and check
the RLGym-PPO adapter.

## Environment baseline

- 1v1 kickoff states from RLGym's standard v2 mutators
- normalized `DefaultObs` with one slot per team (92 floats)
- the standard 90-action `LookupTableAction` wrapped in `RepeatAction`
- configurable action repeat, defaulting to 8 simulation ticks
- goal termination plus no-touch and overall timeout truncation
- a small composed reward: goals, touches, velocity to ball, and ball velocity
  to the opponent goal

`RocketSimEngine(rlbot_delay=True)` is deliberate. The matching live RLBot
configuration uses `rlgym-rlbot`'s default offset of one tick before action end
(`offset=-1`, relative to `ACTION_END`). Changing either side requires changing
and testing both sides together.

## Repository layout

```text
bot/       RLBot v5 entry point and appearance metadata
psycho/    reusable observations, actions, rewards, model, and utilities
training/  RocketSim environment, PPO training, and evaluation workflows
tests/     unit, integration, checkpoint, and smoke tests
```

## Verified dependency direction

- RLGym is consumed through `rlgym-api` and
  `rlgym-rocket-league[sim]`, not the umbrella installation package.
- RLGym-PPO is pinned to an exact Git commit because it is not published on
  PyPI and the official RLGym quick start installs it from GitHub.
- `rlgym-rlbot` provides the RLGym v2 to RLBot v5 bridge.
- WandB is installed transitively by RLGym-PPO but will remain disabled unless
  explicitly requested.

## License

MIT
