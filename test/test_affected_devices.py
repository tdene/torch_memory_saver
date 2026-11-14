from types import SimpleNamespace

import pytest
import torch

from torch_memory_saver.entrypoint import _TorchMemorySaverImpl


@pytest.mark.parametrize(
    "devices", [[], [0, 2], list(range(64)), list(range(65))],
    ids=["empty", "sparse", "capacity", "overflow"],
)
def test_affected_devices(devices):
    calls = 0

    def query(tag, output, capacity):
        nonlocal calls
        calls += 1
        assert tag == b"weights"
        for i, device in enumerate(devices[:capacity]):
            output[i] = device
        return len(devices)

    impl = SimpleNamespace(
        _binary_wrapper=SimpleNamespace(cdll=SimpleNamespace(tms_affected_devices=query))
    )
    if len(devices) > 64:
        with pytest.raises(AssertionError, match="Too many affected devices"):
            _TorchMemorySaverImpl._affected_devices(impl, "weights")
    else:
        assert _TorchMemorySaverImpl._affected_devices(impl, "weights") == devices
    assert calls == 1


def _chunk_impl(**cdll):
    return SimpleNamespace(_chunk_cdll=lambda: SimpleNamespace(**cdll))


def test_chunk_table_follows_a_changing_tag():
    chunks = [(0x7F0000000000, 4, 1), (0x7F0000000004, 4, 0), (0x7F1000000000, 2, 1)]
    capacities = []

    def get_layout(tag, addresses, sizes, states, capacity):
        assert tag == b"kv"
        capacities.append(capacity)
        # The tag gains an allocation after the first call.
        visible = chunks if len(capacities) > 1 else chunks[:2]
        for i, (address, size, state) in enumerate(visible[:capacity]):
            addresses[i], sizes[i], states[i] = address, size, state
        return len(visible)

    impl = _chunk_impl(tms_get_chunk_layout=get_layout)
    assert _TorchMemorySaverImpl.chunk_table(impl, "kv") == (
        [address for address, _, _ in chunks],
        [4, 4, 2],
        [True, False, True],
    )
    assert capacities == [0, 2, 3]


@pytest.mark.parametrize(
    "status, error",
    [(0, None), (2, torch.OutOfMemoryError), (-1, IndexError), (-2, RuntimeError)],
)
def test_resume_chunks_status(status, error):
    seen = []

    def resume_chunks(tag, indices, count):
        seen.append((tag, list(indices[:count])))
        return status

    impl = _chunk_impl(tms_resume_chunks=resume_chunks)
    if error is None:
        _TorchMemorySaverImpl.resume_chunks(impl, "kv", [3, 1])
    else:
        with pytest.raises(error, match="nothing was changed") as caught:
            _TorchMemorySaverImpl.resume_chunks(impl, "kv", [3, 1])
        assert caught.type is error
    assert seen == [(b"kv", [3, 1])]
