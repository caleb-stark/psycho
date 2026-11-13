from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from psycho import __version__
from psycho.actions import lookup_table_size
from psycho.observations import build_observation_builder

if TYPE_CHECKING:
    from training.config import TrainingConfig

MANIFEST_NAME = "PSYCHO_CHECKPOINT.json"
POLICY_NAME = "PPO_POLICY.pt"


class CheckpointError(ValueError):
    pass


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def observation_size(config: TrainingConfig) -> int:
    space_type, size = build_observation_builder(config.environment).get_obs_space("blue-0")
    if space_type != "real" or size < 1:
        raise CheckpointError(f"unsupported observation space: {(space_type, size)}")
    return size


def checkpoint_manifest(config: TrainingConfig, cumulative_timesteps: int) -> dict[str, Any]:
    dependencies = {
        name: version(name)
        for name in (
            "numpy",
            "rlbot",
            "rlgym-api",
            "rlgym-ppo",
            "rlgym-rlbot",
            "rlgym-rocket-league",
            "rocketsim",
            "torch",
        )
    }
    return {
        "schema_version": 1,
        "project": "Psycho",
        "project_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "cumulative_timesteps": cumulative_timesteps,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "dependencies": dependencies,
        "observation_size": observation_size(config),
        "action_count": lookup_table_size(),
        "environment": asdict(config.environment),
        "training": asdict(config),
    }


def write_manifest(
    checkpoint_directory: Path,
    config: TrainingConfig,
    cumulative_timesteps: int,
) -> Path:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_directory / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            checkpoint_manifest(config, cumulative_timesteps),
            indent=2,
            default=_json_value,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_manifest(checkpoint_directory: Path) -> dict[str, Any]:
    manifest_path = checkpoint_directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CheckpointError(f"checkpoint manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def latest_checkpoint(base_directory: Path) -> Path | None:
    if not base_directory.is_dir():
        return None
    candidates = [
        path for path in base_directory.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.name))


def resolve_checkpoint(selection: str | Path, base_directory: Path) -> Path:
    if str(selection).lower() == "latest":
        path = latest_checkpoint(base_directory)
    else:
        path = Path(selection)
    if path is None or not path.is_dir():
        raise CheckpointError(f"checkpoint directory not found: {selection}")
    if not (path / POLICY_NAME).is_file():
        raise CheckpointError(f"policy file not found in checkpoint: {path}")
    load_manifest(path)
    return path


def validate_checkpoint(checkpoint_directory: Path, config: TrainingConfig) -> None:
    manifest = load_manifest(checkpoint_directory)
    expected = checkpoint_manifest(config, int(manifest["cumulative_timesteps"]))
    checks = {
        "observation_size": expected["observation_size"],
        "action_count": expected["action_count"],
        "environment": expected["environment"],
    }
    for key, expected_value in checks.items():
        if manifest.get(key) != expected_value:
            raise CheckpointError(
                f"checkpoint {key} does not match current configuration: "
                f"{manifest.get(key)!r} != {expected_value!r}"
            )
    saved_layers = tuple(manifest["training"]["policy_layer_sizes"])
    if saved_layers != config.policy_layer_sizes:
        raise CheckpointError(
            f"checkpoint policy layers do not match: {saved_layers} != {config.policy_layer_sizes}"
        )


__all__ = [
    "CheckpointError",
    "MANIFEST_NAME",
    "latest_checkpoint",
    "load_manifest",
    "resolve_checkpoint",
    "validate_checkpoint",
    "write_manifest",
]
