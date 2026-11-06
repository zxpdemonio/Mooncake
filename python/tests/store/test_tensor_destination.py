from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mooncake.structured_object_store import (  # noqa: E402
    BundleTransferPolicy,
    tensor_destination,
)
from test_structured_object_store import NoTensorFastPathStore, make_transfer  # noqa: E402


class DirectStore(NoTensorFastPathStore):
    def get_into_ranges(self, *args, require_direct=False):
        assert require_direct is True
        return super().get_into_ranges(*args)


def aligned_tensor(shape, alignment=64):
    size = torch.empty([], dtype=torch.int32).element_size()
    count = 1
    for dim in shape:
        count *= dim
    backing = torch.empty(count * size + alignment, dtype=torch.uint8)
    offset = -backing.data_ptr() % alignment
    return backing[offset : offset + count * size].view(torch.int32).view(shape)


@pytest.mark.parametrize("mode", ["copy", "auto"])
@pytest.mark.parametrize("rows", [None, slice(1, 3), [2, 0, 2], []])
def test_direct_tensor_payload_uses_caller_tensor_without_deserialization(
    mode, rows, monkeypatch
):
    store, transfer = make_transfer(DirectStore())
    source = torch.full((3, 2, 16), -1, dtype=torch.int32)
    source[..., :9] = torch.arange(54, dtype=torch.int32).view(3, 2, 9)
    ref = transfer.put(
        {"value": source}, type="dict", policy=BundleTransferPolicy(copy_mode=mode)
    )
    expected = source if rows is None else source[rows]
    target = aligned_tensor(expected.shape)
    monkeypatch.setattr(
        "mooncake.structured_object_store._deserialize_tensor_payload",
        lambda *_: pytest.fail("direct payload must not be deserialized"),
    )
    result = transfer.get(
        ref,
        type="dict",
        rows=rows,
        destinations={
            "value": tensor_destination(target, alignment=64, pre_registered=True)
        },
    )
    assert result["value"] is target
    torch.testing.assert_close(target, expected)
    assert target.data_ptr() % 64 == 0
    assert store.get_tensor_calls == 0
    assert store.get_into_ranges_calls == (0 if target.numel() == 0 else 1)


def test_tensor_destination_rejects_invalid_contracts():
    target = aligned_tensor((2, 16))
    with pytest.raises(ValueError, match="pre_registered"):
        tensor_destination(target, alignment=64)
    for alignment in (0, 3, True):
        with pytest.raises(ValueError, match="alignment"):
            tensor_destination(target, alignment=alignment, pre_registered=True)
    with pytest.raises(ValueError, match="aligned"):
        tensor_destination(target.view(-1)[1:], alignment=64, pre_registered=True)
    with pytest.raises(ValueError, match="contiguous"):
        tensor_destination(target.t(), pre_registered=True)


@pytest.mark.parametrize("change", ["shape", "dtype", "transport"])
def test_direct_tensor_read_fails_without_staging(change):
    store, transfer = make_transfer(DirectStore())
    source = torch.zeros((2, 16), dtype=torch.int32)
    ref = transfer.put(
        {"value": source}, type="dict", policy=BundleTransferPolicy(copy_mode="copy")
    )
    target = aligned_tensor((2, 16) if change != "shape" else (1, 32))
    if change == "dtype":
        target = target.view(torch.float32)
    if change == "transport":
        transfer._transport._get_into_ranges = None
    with pytest.raises((ValueError, RuntimeError)):
        transfer.get(
            ref,
            type="dict",
            destinations={"value": tensor_destination(target, pre_registered=True)},
        )
    assert store.get_into_ranges_calls == 0


def test_tensor_destination_rejects_mutation_after_registration():
    target = aligned_tensor((2, 16))
    destination = tensor_destination(target, alignment=64, pre_registered=True)
    target.as_strided_((2, 16), (1, 2))
    with pytest.raises(ValueError, match="changed since registration"):
        destination.validate(
            {"encoding": "torch_tensor", "dtype": "torch.int32", "element_size": 4},
            (2, 16),
        )


@pytest.mark.parametrize("offset", [0, 4])
def test_direct_destinations_reject_aliases_before_payload_read(offset):
    store, transfer = make_transfer(DirectStore())
    source = torch.ones((2, 16), dtype=torch.int32)
    ref = transfer.put(
        {"a": source, "b": source * 2},
        type="dict",
        policy=BundleTransferPolicy(copy_mode="copy"),
    )
    allocation = aligned_tensor((40,))
    first = allocation[:32].view(2, 16)
    second = allocation[offset : offset + 32].view(2, 16)
    with pytest.raises(ValueError, match="overlap"):
        transfer.get(
            ref,
            type="dict",
            destinations={
                "a": tensor_destination(first, pre_registered=True),
                "b": tensor_destination(second, pre_registered=True),
            },
        )
    assert store.get_into_ranges_calls == 0
