from argparse import Namespace

import pytest

from training.config import TrainingConfig, get_profile
from training.train import config_from_args


def test_training_profiles_are_internally_consistent() -> None:
    for name in ("dev", "production", "smoke"):
        config = get_profile(name)
        assert config.ppo_minibatch_size <= config.ppo_batch_size
        assert config.ppo_batch_size <= config.experience_buffer_size
        assert config.min_inference_size <= config.n_processes


def test_invalid_training_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_inference_size"):
        TrainingConfig(n_processes=1, min_inference_size=2)


def test_command_line_overrides_do_not_require_source_edits() -> None:
    args = Namespace(
        profile="dev",
        resume="none",
        checkpoint_dir=None,
        checkpoint_interval=None,
        timesteps=512,
        processes=3,
        min_inference_size=None,
        device="cpu",
        seed=7,
        wandb=False,
    )

    config = config_from_args(args)

    assert config.timestep_limit == 512
    assert config.n_processes == 3
    assert config.min_inference_size == 3
    assert config.device == "cpu"
    assert config.random_seed == 7
