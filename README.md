# Psycho

Psycho is a Rocket League machine-learning bot built around RLGym v2,
RocketSim, PPO, and RLBot v5. The project is being rebuilt from its original
rule-based RLBot v4/Gosling implementation.

The core design rule is that training and live play import the same RLGym
observation builder and action parser. The RLBot process does not reproduce
training observations by hand.

## Status

The repository currently contains the clean Python project foundation and
pinned ecosystem dependencies. RocketSim environment, training, evaluation,
checkpoint, and live-bot workflows are added in subsequent milestones and are
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
