from importlib.metadata import version

import psycho


def test_project_metadata() -> None:
    assert psycho.__version__ == "0.1.0"


def test_critical_dependencies_are_importable() -> None:
    import rlbot  # noqa: F401
    import rlgym  # noqa: F401
    import rlgym_ppo  # noqa: F401
    import rlgym_rlbot  # noqa: F401
    import RocketSim  # noqa: F401
    import torch  # noqa: F401

    assert version("rlgym-api") == "2.0.0"
    assert version("rlgym-rocket-league") == "2.0.1"
    assert version("rlgym-rlbot") == "0.3.0"
    assert version("rocketsim") == "2.2.1"
