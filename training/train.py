import argparse
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

from rlgym_ppo import Learner

from psycho.checkpoints import (
    latest_checkpoint,
    resolve_checkpoint,
    validate_checkpoint,
    write_manifest,
)
from training.config import TrainingConfig, get_profile
from training.environment import build_ppo_environment


class PsychoLearner(Learner):
    def __init__(self, *args, psycho_config: TrainingConfig, **kwargs) -> None:
        self.psycho_config = psycho_config
        super().__init__(*args, **kwargs)

    def save(self, cumulative_timesteps) -> None:
        super().save(cumulative_timesteps)
        checkpoint_directory = self.psycho_config.checkpoint_directory / str(cumulative_timesteps)
        write_manifest(checkpoint_directory, self.psycho_config, int(cumulative_timesteps))

    def learn(self) -> None:
        """Run PPO while preserving a checkpoint on interruption or failure."""

        try:
            self._learn()
        except KeyboardInterrupt:
            print("Training interrupted; saving the current state.")
            self.save(self.agent.cumulative_timesteps)
        except Exception:
            self.save(self.agent.cumulative_timesteps)
            raise
        finally:
            self.cleanup()


def create_learner(config: TrainingConfig, resume: str | None = None) -> PsychoLearner:
    """Create an RLGym-PPO learner without starting its training loop."""

    environment_factory = partial(build_ppo_environment, config.environment)
    resume_path = None
    if resume is not None:
        resume_path = resolve_checkpoint(resume, config.checkpoint_directory)
        validate_checkpoint(resume_path, config)

    return PsychoLearner(
        environment_factory,
        psycho_config=config,
        n_proc=config.n_processes,
        min_inference_size=config.min_inference_size,
        timestep_limit=config.timestep_limit,
        exp_buffer_size=config.experience_buffer_size,
        ts_per_iteration=config.timesteps_per_iteration,
        standardize_returns=True,
        standardize_obs=False,
        policy_layer_sizes=config.policy_layer_sizes,
        critic_layer_sizes=config.critic_layer_sizes,
        ppo_epochs=config.ppo_epochs,
        ppo_batch_size=config.ppo_batch_size,
        ppo_minibatch_size=config.ppo_minibatch_size,
        ppo_ent_coef=config.entropy_coefficient,
        policy_lr=config.policy_learning_rate,
        critic_lr=config.critic_learning_rate,
        log_to_wandb=config.log_to_wandb,
        load_wandb=config.log_to_wandb,
        wandb_project_name="psycho",
        checkpoints_save_folder=str(config.checkpoint_directory),
        add_unix_timestamp=False,
        checkpoint_load_folder=None if resume_path is None else str(resume_path),
        save_every_ts=config.checkpoint_interval,
        random_seed=config.random_seed,
        n_checkpoints_to_keep=config.checkpoints_to_keep,
        device=config.device,
    )


def run_training(config: TrainingConfig, resume: str | None = None) -> int:
    learner = create_learner(config, resume=resume)
    learner.learn()
    updates = int(learner.ppo_learner.cumulative_model_updates)
    if updates < 1:
        raise RuntimeError("training exited without completing a PPO update")
    cumulative_timesteps = int(learner.agent.cumulative_timesteps)
    latest = latest_checkpoint(config.checkpoint_directory)
    if latest is None or latest.name != str(cumulative_timesteps):
        learner.save(cumulative_timesteps)
    print(f"Training finished after {updates} PPO model update(s).")
    return updates


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Psycho with RLGym-PPO.")
    parser.add_argument("--profile", choices=("dev", "production", "smoke"), default="dev")
    parser.add_argument(
        "--resume",
        default="none",
        help="Checkpoint directory, 'latest', or 'none' to start from scratch.",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--min-inference-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = get_profile(args.profile)
    updates = {}
    if args.checkpoint_dir is not None:
        updates["checkpoint_directory"] = args.checkpoint_dir
    if args.checkpoint_interval is not None:
        updates["checkpoint_interval"] = args.checkpoint_interval
    if args.timesteps is not None:
        updates["timestep_limit"] = args.timesteps
    if args.processes is not None:
        updates["n_processes"] = args.processes
        if args.min_inference_size is None:
            updates["min_inference_size"] = max(1, round(args.processes * 0.9))
    if args.min_inference_size is not None:
        updates["min_inference_size"] = args.min_inference_size
    if args.device is not None:
        updates["device"] = args.device
    if args.seed is not None:
        updates["random_seed"] = args.seed
    if args.wandb:
        updates["log_to_wandb"] = True
    return replace(config, **updates)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    resume = None if args.resume.lower() == "none" else args.resume
    run_training(config, resume=resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
