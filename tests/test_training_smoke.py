import subprocess
import sys


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
