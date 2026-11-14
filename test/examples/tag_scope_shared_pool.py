"""tag_scope() and create_mem_pool(): one memory pool that holds memory of two tags.

region() enters a memory pool that the package keeps for the tag. tag_scope() sets the
tag and the backup settings of the thread and enters no pool, and create_mem_pool() makes
a pool on the hook's allocator that belongs to the caller. Two tensors are allocated into
one such pool, each inside the scope of its own tag with CPU backup: PyTorch requests one
segment per tensor, and the hook gives each segment the tag of the scope that is open
then. Both segments are segments of the caller's pool in PyTorch's memory snapshot, and
the package has no pool. Pausing one tag returns that tag's segment to the device while
the other tensor holds its data; resuming restores the data.

The pool is the caller's to drop: PyTorch then returns its segments through the hook.
The example drops it inside a scope, where every torch hook frees memory.

In hook_mode="torch" the hook ends the process when it is asked for memory outside a
scope, so the exit status is part of the result and the test runs this file as a script:
  python test/examples/tag_scope_shared_pool.py torch
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG_A = "tag_scope_shared_pool_a"
_TAG_B = "tag_scope_shared_pool_b"
# From 10 MiB on, PyTorch requests a segment of the tensor's size rounded up to 2 MiB: one per tensor.
_TENSOR_BYTES = 256 * 1024**2
_MIN_DELTA_BYTES = 0.9 * _TENSOR_BYTES  # expected rise/drop per segment, with slack for alignment/noise
_MAX_DELTA_BYTES = 1.1 * _TENSOR_BYTES
_MAX_NOISE_BYTES = 0.1 * _TENSOR_BYTES  # how far below the baseline free memory may stay


def _holds(tensor: torch.Tensor, value: int) -> bool:
    # min() and max() read the tensor in place; sum() would first copy it as int64.
    return int(tensor.min()) == value == int(tensor.max())


def _segment_of(tensor: torch.Tensor) -> dict:
    """The segment of PyTorch's memory snapshot that the tensor lies in."""
    address = tensor.data_ptr()
    [segment] = [
        segment
        for segment in get_device_module().memory_snapshot()
        if segment["address"] <= address < segment["address"] + segment["total_size"]
    ]
    return segment


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device = get_device()
    mod = get_device_module()
    device_index = mod.current_device()

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)
    assert _holds(normal_tensor, 100)
    # PyTorch's own cache is empty at the baseline, as it is after the last empty_cache().
    empty_cache()
    free_baseline = free_bytes()

    pool = torch_memory_saver.create_mem_pool()
    impl = torch_memory_saver._impl
    in_scope = impl._binary_wrapper.cdll.tms_get_interesting_region
    assert not in_scope()

    # One pool, two tags: each tensor is allocated inside the scope of its tag.
    with mod.use_mem_pool(pool):
        with torch_memory_saver.tag_scope(tag=_TAG_A, enable_cpu_backup=True):
            assert in_scope()
            tensor_a = torch.full((_TENSOR_BYTES,), 1, dtype=torch.uint8, device=device)
        with torch_memory_saver.tag_scope(tag=_TAG_B, enable_cpu_backup=True):
            tensor_b = torch.full((_TENSOR_BYTES,), 2, dtype=torch.uint8, device=device)
        assert not in_scope()
    assert free_baseline - free_bytes() > 2 * _MIN_DELTA_BYTES
    assert _holds(tensor_a, 1) and _holds(tensor_b, 2)

    # The hook holds memory of both tags, and the package made no pool for either: the two
    # segments are segments of the caller's pool.
    assert impl._affected_devices(_TAG_A) == [device_index]
    assert impl._affected_devices(_TAG_B) == [device_index]
    assert not impl._mem_pools
    segment_a, segment_b = _segment_of(tensor_a), _segment_of(tensor_b)
    assert segment_a["address"] != segment_b["address"]
    assert tuple(segment_a["segment_pool_id"]) == tuple(pool.id)
    assert tuple(segment_b["segment_pool_id"]) == tuple(pool.id)

    # Pausing a tag returns its segment, and only its segment, to the device: the tensor of
    # the other tag holds its data meanwhile. Resuming restores the data from the backup.
    for tag, tensor, value, other, other_value in (
        (_TAG_A, tensor_a, 1, tensor_b, 2),
        (_TAG_B, tensor_b, 2, tensor_a, 1),
    ):
        free_active = free_bytes()
        torch_memory_saver.pause(tag)
        free_paused = free_bytes()
        assert _MIN_DELTA_BYTES < free_paused - free_active < _MAX_DELTA_BYTES
        assert _holds(other, other_value)
        assert _holds(torch_memory_saver.get_cpu_backup(tensor, zero_copy=True), value)

        torch_memory_saver.resume(tag)
        assert _MIN_DELTA_BYTES < free_paused - free_bytes() < _MAX_DELTA_BYTES
        assert _holds(tensor, value) and _holds(other, other_value)

    # The pool keeps the segments of deleted tensors until its owner drops it. Then the
    # hook holds nothing of either tag, and the memory is back at the device.
    del tensor_a, tensor_b, tensor, other
    gc.collect()
    assert impl._affected_devices(_TAG_A) == [device_index]
    assert impl._affected_devices(_TAG_B) == [device_index]
    assert free_baseline - free_bytes() > 2 * _MIN_DELTA_BYTES
    with torch_memory_saver.tag_scope(tag=_TAG_A):
        del pool
        gc.collect()
        assert impl._affected_devices(_TAG_A) == [] and impl._affected_devices(_TAG_B) == []
    assert not in_scope()
    empty_cache()
    assert free_baseline - free_bytes() < _MAX_NOISE_BYTES
    assert _holds(normal_tensor, 100)


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
