from dataclasses import dataclass
from typing import Optional, Tuple

from torch import nn

from .config import QuantPolicy


@dataclass(frozen=True)
class QuantTarget:
    name: str
    module: nn.Module
    prev_layer: Optional[nn.Module] = None


@dataclass(frozen=True)
class ReconstructionUnit:
    name: str
    module: nn.Module
    is_head: bool = False


@dataclass(frozen=True)
class QuantizedBackbone:
    model: nn.Module
    policy: QuantPolicy
    targets: Tuple[QuantTarget, ...]
    units: Tuple[ReconstructionUnit, ...]
