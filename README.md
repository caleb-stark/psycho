# Psycho

Psycho is a Rocket League machine-learning bot built around RLGym v2,
RocketSim, PPO, and RLBot v5. The project is being rebuilt from its original
rule-based RLBot v4/Gosling implementation.

The core design rule is that training and live play import the same RLGym
observation builder and action parser. The RLBot process does not reproduce
training observations by hand.

## Status

The repository currently contains the clean Python project foundation, a
tested 1v1 RLGym v2 environment backed by RocketSim, and working RLGym-PPO
training profiles. Evaluation, checkpoint resume/export, and live-bot
workflows are documented only when verified.

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

## Training

Run the one-update smoke profile before a longer session:

```powershell
uv run python -m training.train --profile smoke
```

The smoke profile uses one worker, collects 256 timesteps, and performs one PPO
update on CPU. It checks the multiprocessing, experience collection, policy
optimization, and cleanup path; it is not intended to learn useful gameplay.

Start the development profile:

```powershell
uv run python -m training.train --profile dev
```

Start the larger production baseline:

```powershell
uv run python -m training.train --profile production
```

The production settings are a starting point, not measured optimums. Override
common machine-dependent settings without editing source:

```powershell
uv run python -m training.train --profile dev --processes 4 --device cpu --timesteps 20000
```

Use `--wandb` to opt into Weights & Biases. Training otherwise logs iteration
timesteps, throughput, reward, policy/value losses, entropy, KL divergence, and
update counts to the console without requiring a WandB account.

## Checkpoints and resume

Training writes numeric checkpoint directories under `checkpoints/ppo/` at the
configured interval and once more at clean exit. The development default keeps
the five newest checkpoints. Checkpoints, model files, and training logs are
ignored by Git.

Resume the newest compatible checkpoint:

```powershell
uv run python -m training.train --profile dev --resume latest
```

Or select a numeric checkpoint explicitly:

```powershell
uv run python -m training.train --profile dev --resume checkpoints/ppo/100000
```

Every checkpoint contains RLGym-PPO's policy, value function, optimizers, and
bookkeeping plus `PSYCHO_CHECKPOINT.json`. Psycho's manifest records the Python
and dependency versions, environment timing/action settings, observation and
action sizes, network architecture, training configuration, and timestep.
Psycho rejects resume attempts whose observation, action, environment, or
policy-layer configuration differs from the checkpoint.

Press `q` during an iteration to checkpoint and quit after that iteration.
`Ctrl+C` also triggers a best-effort checkpoint and worker cleanup.

## Evaluation

Evaluate the newest checkpoint without updating any weights:

```powershell
uv run python -m training.evaluate --checkpoint latest --episodes 20
```

Select a checkpoint directly when comparing runs:

```powershell
uv run python -m training.evaluate --checkpoint checkpoints/ppo/100000 --episodes 20
```

Evaluation reconstructs the checkpoint's observation, action, reward, timeout,
and RocketSim timing settings from its manifest. The policy controls both sides
in deterministic self-play by default and reports goals, goals conceded,
touches, wins/losses/draws, mean reward per decision, and mean episode duration.
Use `--stochastic` only when sampling behavior is desired.

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
