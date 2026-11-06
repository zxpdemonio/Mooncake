"""Caller-owned tensor payload destinations for synchronous direct reads."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mooncake._structured_ranges import _nonoverlap


@dataclass(frozen=True)
class TensorDestination:
    """A pre-registered tensor whose payload, without a header, is writable.

    The caller owns registration and the tensor through physical completion.
    This destination is always strict: unsupported reads must not stage through
    host memory or replace the tensor with a newly allocated result.
    """

    tensor: Any
    alignment: int = 1
    pre_registered: bool = False
    _contract: tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("tensor destination requires a torch.Tensor")
        if self.tensor.layout != torch.strided or not self.tensor.is_contiguous():
            raise ValueError("tensor destination must be a contiguous strided tensor")
        if self.tensor.device.type not in {"cpu", "cuda"}:
            raise ValueError("tensor destination supports CPU and CUDA memory only")
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError(
                "tensor destination alignment must be a positive power of two"
            )
        if self.pre_registered is not True:
            raise ValueError("tensor destination requires pre_registered=True")
        if self.tensor.numel() and self.tensor.data_ptr() % self.alignment:
            raise ValueError("tensor destination data pointer is not aligned")
        object.__setattr__(self, "_contract", self._memory_contract())

    def _memory_contract(self) -> tuple[Any, ...]:
        value = self.tensor
        return (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
            value.layout,
            value.numel(),
        )

    def validate(self, spec: Mapping[str, Any], shape: Sequence[int]) -> int:
        if self._memory_contract() != self._contract:
            raise ValueError("tensor destination changed since registration")
        if spec.get("encoding") != "torch_tensor" or spec.get("nested"):
            raise ValueError("tensor destination requires a dense torch tensor member")
        if str(self.tensor.dtype) != spec.get("dtype"):
            raise ValueError("tensor destination dtype differs from stored member")
        if tuple(self.tensor.shape) != tuple(shape):
            raise ValueError("tensor destination shape differs from selected member")
        if self.tensor.element_size() != spec.get("element_size"):
            raise ValueError(
                "tensor destination element size differs from stored member"
            )
        return math.prod(shape) * self.tensor.element_size()


def tensor_destination(
    tensor: Any, *, alignment: int = 1, pre_registered: bool = False
) -> TensorDestination:
    """Wrap a registered CPU/CUDA tensor for payload-only direct materialization."""
    return TensorDestination(tensor, alignment, pre_registered)


def validate_tensor_destinations(destinations: Mapping[str, Any]) -> None:
    intervals = {}
    for destination in destinations.values():
        if not isinstance(destination, TensorDestination):
            continue
        if destination._memory_contract() != destination._contract:
            raise ValueError("tensor destination changed since registration")
        tensor = destination.tensor
        size = tensor.numel() * tensor.element_size()
        if size:
            intervals.setdefault(tensor.device, []).append(
                (tensor.data_ptr(), tensor.data_ptr() + size)
            )
    for ranges in intervals.values():
        _nonoverlap(ranges)
