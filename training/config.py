from dataclasses import dataclass, field
from pathlib import Path

from psycho.config import EnvironmentConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """RLGym-PPO settings with explicit development-friendly defaults."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    n_processes: int = 2
    min_inference_size: int = 2
    ppo_batch_size: int = 2_048
    ppo_minibatch_size: int = 512
    experience_buffer_size: int = 4_096
    timesteps_per_iteration: int = 2_048
    policy_layer_sizes: tuple[int, ...] = (128, 128)
    critic_layer_sizes: tuple[int, ...] = (128, 128)
    policy_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    ppo_epochs: int = 2
    entropy_coefficient: float = 0.01
    checkpoint_interval: int = 25_000
    timestep_limit: int = 100_000
    checkpoint_directory: Path = Path("checkpoints/ppo")
    checkpoints_to_keep: int = 5
    device: str = "auto"
    random_seed: int = 123
    log_to_wandb: bool = False

    def __post_init__(self) -> None:
        if self.n_processes < 1:
            raise ValueError("n_processes must be at least 1")
        if not 1 <= self.min_inference_size <= self.n_processes:
            raise ValueError("min_inference_size must be between 1 and n_processes")
        if not 1 <= self.ppo_minibatch_size <= self.ppo_batch_size:
            raise ValueError("ppo_minibatch_size must be between 1 and ppo_batch_size")
        if self.ppo_batch_size > self.experience_buffer_size:
            raise ValueError("ppo_batch_size cannot exceed experience_buffer_size")
        if self.timesteps_per_iteration < 1 or self.timestep_limit < 1:
            raise ValueError("timestep counts must be positive")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be at least 1")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if not self.policy_layer_sizes or not self.critic_layer_sizes:
            raise ValueError("policy and critic must each have at least one hidden layer")


def development_config() -> TrainingConfig:
    return TrainingConfig()


def production_config() -> TrainingConfig:
    """A larger starting point for real learning, not a claim of optimality."""

    return TrainingConfig(
        n_processes=8,
        min_inference_size=7,
        ppo_batch_size=50_000,
        ppo_minibatch_size=10_000,
        experience_buffer_size=150_000,
        timesteps_per_iteration=50_000,
        policy_layer_sizes=(512, 512, 512),
        critic_layer_sizes=(512, 512, 512),
        policy_learning_rate=5e-5,
        critic_learning_rate=5e-5,
        ppo_epochs=2,
        checkpoint_interval=1_000_000,
        timestep_limit=1_000_000_000,
    )


def smoke_config() -> TrainingConfig:
    """One tiny PPO update used to prove worker collection and optimization."""

    return TrainingConfig(
        n_processes=1,
        min_inference_size=1,
        ppo_batch_size=256,
        ppo_minibatch_size=128,
        experience_buffer_size=512,
        timesteps_per_iteration=256,
        policy_layer_sizes=(64, 64),
        critic_layer_sizes=(64, 64),
        ppo_epochs=1,
        checkpoint_interval=1_000_000,
        timestep_limit=256,
        checkpoint_directory=Path("runs/smoke"),
        checkpoints_to_keep=1,
        device="cpu",
    )


def get_profile(name: str) -> TrainingConfig:
    profiles = {
        "dev": development_config,
        "production": production_config,
        "smoke": smoke_config,
    }
    try:
        return profiles[name]()
    except KeyError as error:
        raise ValueError(f"unknown training profile: {name}") from error


__all__ = ["TrainingConfig", "get_profile"]
