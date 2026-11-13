from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from psycho.checkpoints import POLICY_NAME, load_manifest


class DiscretePolicy(nn.Module):
    """Inference-only equivalent of RLGym-PPO's feed-forward discrete policy."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")

        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.extend((nn.Linear(input_size, action_count), nn.Softmax(dim=-1)))
        self.model = nn.Sequential(*layers)
        self.action_count = action_count

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    @torch.inference_mode()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        tensor = torch.as_tensor(observation, dtype=torch.float32)
        probabilities = self(tensor).reshape(-1, self.action_count)
        if deterministic:
            return int(probabilities.argmax(dim=-1).item())
        return int(torch.multinomial(probabilities, 1).item())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_directory: Path,
        device: str = "cpu",
    ) -> "DiscretePolicy":
        manifest = load_manifest(checkpoint_directory)
        policy = cls(
            observation_size=int(manifest["observation_size"]),
            action_count=int(manifest["action_count"]),
            hidden_sizes=tuple(manifest["training"]["policy_layer_sizes"]),
        ).to(device)
        state_dict = torch.load(
            checkpoint_directory / POLICY_NAME,
            map_location=device,
            weights_only=True,
        )
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy
