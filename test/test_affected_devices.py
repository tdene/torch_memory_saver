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


@pytest.mark.parametrize(
    "status, error",
    [(0, None), (2, torch.OutOfMemoryError), (-1, IndexError), (-2, RuntimeError),
     (-3, ValueError), (-4, ValueError), (-5, ValueError)],
)
def test_map_chunks_status(status, error):
    seen = []

    def map_chunks(dst_tag, dst_indices, src_tag, src_indices, count):
        seen.append((dst_tag, list(dst_indices[:count]), src_tag, list(src_indices[:count])))
        return status

    impl = _chunk_impl(tms_map_chunks=map_chunks)
    if error is None:
        _TorchMemorySaverImpl.map_chunks(impl, "kv", [6, 7], "mamba", [1, 0])
    else:
        with pytest.raises(error, match="nothing was changed") as caught:
            _TorchMemorySaverImpl.map_chunks(impl, "kv", [6, 7], "mamba", [1, 0])
        assert caught.type is error
    assert seen == [(b"kv", [6, 7], b"mamba", [1, 0])]

    # Lists of different lengths never reach the hook.
    with pytest.raises(ValueError, match="2 destination chunks for 1 source chunks"):
        _TorchMemorySaverImpl.map_chunks(impl, "kv", [6, 7], "mamba", [1])
    assert len(seen) == 1


@pytest.mark.parametrize("call", ["pause", "resume"])
@pytest.mark.parametrize("status, error", [(0, None), (1, RuntimeError), (2, torch.OutOfMemoryError)])
def test_pause_resume_status(call, status, error):
    if call == "pause" and status == 2:
        error = None  # pause() has no out of memory
    cdll = SimpleNamespace(
        **{f"tms_{call}": lambda tag: status, "tms_affected_devices": lambda *args: 0}
    )
    impl = SimpleNamespace(
        _binary_wrapper=SimpleNamespace(cdll=cdll),
        _is_xpu=False,
        _sync_affected_devices=lambda tag: None,
    )
    method = getattr(_TorchMemorySaverImpl, call)
    if error is None:
        method(impl, "kv")
    else:
        with pytest.raises(error) as caught:
            method(impl, "kv")
        assert caught.type is error
