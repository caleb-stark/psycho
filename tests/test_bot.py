import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from bot.bot import PsychoBot
from bot.runtime import select_checkpoint, selected_checkpoint
from psycho.checkpoints import POLICY_NAME, write_manifest
from psycho.models import DiscretePolicy
from training.config import smoke_config

REPOSITORY_ROOT = Path(__file__).parents[1]


def make_checkpoint(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "0"
    config = replace(smoke_config(), checkpoint_directory=checkpoint_root)
    policy = DiscretePolicy(92, 90, config.policy_layer_sizes)
    checkpoint.mkdir(parents=True)
    torch.save(policy.state_dict(), checkpoint / POLICY_NAME)
    write_manifest(checkpoint, config, cumulative_timesteps=0)
    return checkpoint_root, checkpoint


def test_model_selection_requires_no_source_edit(tmp_path) -> None:
    checkpoint_root, checkpoint = make_checkpoint(tmp_path)
    selection_file = tmp_path / "selected.json"

    selected = select_checkpoint("latest", checkpoint_root, selection_file)

    assert selected == checkpoint.resolve()
    assert (
        selected_checkpoint(base_directory=checkpoint_root, selection_file=selection_file)
        == selected
    )


def test_rlbot_uses_shared_observation_action_and_timing(tmp_path) -> None:
    _, checkpoint = make_checkpoint(tmp_path)
    bot = PsychoBot(checkpoint)

    assert bot.obs_builder.get_obs_space("blue-0") == ("real", 92)
    assert bot.action_parser.repeats == 8
    assert bot.config.step_offset.offset == -1
    assert bot.config.step_offset.relative_to.name == "ACTION_END"

    action = bot.get_action(np.zeros(92, dtype=np.float32), None, None)
    parsed = bot.action_parser.parser._lookup_table[action]
    assert action.shape == (1,)
    assert 0 <= action.item() < 90
    assert parsed.shape == (1, 8)
    assert np.isfinite(parsed).all()


def test_rlbot_v5_metadata_and_preserved_loadout_are_valid_toml() -> None:
    bot_config = tomllib.loads((REPOSITORY_ROOT / "bot" / "bot.toml").read_text())
    loadout = tomllib.loads((REPOSITORY_ROOT / "bot" / "loadout.toml").read_text())

    assert bot_config["settings"]["agent_id"] == "caleb-stark/psycho"
    assert "python.exe -m bot.bot" in bot_config["settings"]["run_command"]
    assert loadout["blue_loadout"]["car_id"] == 23
    assert loadout["blue_loadout"]["decal_id"] == 3994
    assert loadout["orange_loadout"]["wheels_id"] == 386
