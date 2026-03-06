"""Minimal base policy for action chunking decoder."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


@dataclass
class ActionOutput:
    """Container for policy outputs."""

    actions: torch.Tensor
    mean: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None

    @property
    def is_gaussian(self) -> bool:
        return self.mean is not None


class ActionHead(nn.Module):
    """Simple action head that returns ActionOutput."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> ActionOutput:
        return ActionOutput(actions=self.output_layer(x))


class BasePolicy(nn.Module):
    """Minimal base class for policies - compatible with ActionChunkingTransformerPolicy."""

    def __init__(self, cfg: DictConfig, input_dim: int = None, output_dim: int = None):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.action_head = ActionHead(input_dim=input_dim, output_dim=output_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
