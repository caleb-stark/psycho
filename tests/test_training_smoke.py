import subprocess
import sys

import numpy as np

from psycho.checkpoints import load_manifest
from psycho.models import DiscretePolicy


def test_ppo_collects_experience_and_updates(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.train",
            "--profile",
            "smoke",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--checkpoint-interval",
            "256",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Training finished after" in output
    assert "PPO model update(s)" in output

    checkpoint = tmp_path / "checkpoints" / "256"
    manifest = load_manifest(checkpoint)
    assert manifest["cumulative_timesteps"] == 256
    assert manifest["environment"]["action_repeat"] == 8
    assert manifest["observation_size"] == 92
    assert manifest["action_count"] == 90

    observation = np.zeros(92, dtype=np.float32)
    first_policy = DiscretePolicy.from_checkpoint(checkpoint)
    second_policy = DiscretePolicy.from_checkpoint(checkpoint)
    assert first_policy.act(observation) == first_policy.act(observation)
    assert first_policy.act(observation) == second_policy.act(observation)

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.train",
            "--profile",
            "smoke",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--checkpoint-interval",
            "256",
            "--timesteps",
            "512",
            "--resume",
            "latest",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    resumed_output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0, resumed_output
    assert "Loading from checkpoint" in resumed_output
    assert (tmp_path / "checkpoints" / "512" / "PSYCHO_CHECKPOINT.json").is_file()
