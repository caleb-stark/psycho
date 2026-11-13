import json
import os
from pathlib import Path

from psycho.checkpoints import resolve_checkpoint

SELECTION_FILE = Path(__file__).with_name("selected_checkpoint.json")


def select_checkpoint(
    selection: str | Path,
    base_directory: Path = Path("checkpoints/ppo"),
    selection_file: Path = SELECTION_FILE,
) -> Path:
    checkpoint = resolve_checkpoint(selection, base_directory).resolve()
    selection_file.write_text(
        json.dumps({"checkpoint": str(checkpoint)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return checkpoint


def selected_checkpoint(
    explicit: str | Path | None = None,
    base_directory: Path = Path("checkpoints/ppo"),
    selection_file: Path = SELECTION_FILE,
) -> Path:
    if explicit is not None:
        return resolve_checkpoint(explicit, base_directory)

    environment_selection = os.environ.get("PSYCHO_CHECKPOINT")
    if environment_selection:
        return resolve_checkpoint(environment_selection, base_directory)

    if selection_file.is_file():
        values = json.loads(selection_file.read_text(encoding="utf-8"))
        return resolve_checkpoint(values["checkpoint"], base_directory)

    return resolve_checkpoint("latest", base_directory)


__all__ = ["select_checkpoint", "selected_checkpoint"]
