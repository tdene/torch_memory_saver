"""release_region_pools() against PyTorch's allocator and the driver.

A region's memory pool keeps the segments of deleted tensors, paused or not, and hands
them to the next region with the same arguments. release_region_pools(tag) drops the
pools of the tag's regions: the memory no tensor occupies returns to the device at once,
the hook holds nothing of the tag any more, and a new region with the same arguments
allocates memory that holds data. Inside an open region of the tag the call raises and
drops nothing; the errors are kept to the end, and no kept error may keep a pool's memory.
The segment of a tensor that is still alive follows once the tensor is deleted and the
cache is emptied.

PyTorch below 2.8 does not return such a segment through the hook: there the call must
raise NotImplementedError and drop nothing.

PyTorch up to 2.9 ends the process when a pool is destroyed while it is in use, so the
exit status is part of the result and the test runs this file as a script:
  python test/examples/release_region_pools.py torch
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG = "release_region_pools"
_OTHER_TAG = "release_region_pools_other"
_TENSOR_BYTES = 256 * 1024**2
_MIN_DELTA_BYTES = 0.9 * _TENSOR_BYTES  # expected rise/drop per tensor, with slack for alignment/noise
_MAX_NOISE_BYTES = 0.1 * _TENSOR_BYTES  # how far below the baseline free memory may stay
_SUPPORTED = tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (2, 8)


def _holds(tensor: torch.Tensor, value: int) -> bool:
    # min() and max() read the tensor in place; sum() would first copy it as int64.
    return int(tensor.min()) == value == int(tensor.max())


def _assert_raises(error, tag):
    try:
        torch_memory_saver.release_region_pools(tag)
    except error as caught:
        # NotImplementedError is a RuntimeError: the type has to be the expected one itself.
        assert type(caught) is error, type(caught)
        print(f"release_region_pools({tag!r}) raised {error.__name__}: {caught}")
        return caught
    raise AssertionError(f"release_region_pools({tag!r}) did not raise {error.__name__}")


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device = get_device()
    mod = get_device_module()
    device_index = mod.current_device()

    def region_tensor(value: int, tag: str = _TAG, **region_kwargs):
        with torch_memory_saver.region(tag=tag, **region_kwargs):
            return torch.full((_TENSOR_BYTES,), value, dtype=torch.uint8, device=device)

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)
    bystander = region_tensor(9, tag=_OTHER_TAG)
    assert _holds(normal_tensor, 100) and _holds(bystander, 9)
    impl = torch_memory_saver._impl

    def tag_pools():
        return [key for key in impl._mem_pools if key[0] == _TAG]

    if not _SUPPORTED:
        plain = region_tensor(2)
        pools = dict(impl._mem_pools)
        assert len(pools) == 2
        _assert_raises(NotImplementedError, _TAG)
        _assert_raises(NotImplementedError, None)
        assert dict(impl._mem_pools) == pools
        assert impl._affected_devices(_TAG) == [device_index]
        assert _holds(plain, 2) and _holds(bystander, 9)

        # Return the regions' memory before the interpreter exits: the tensors, then their pools.
        del plain, bystander, pools
        impl._mem_pools.clear()
        gc.collect()
        empty_cache()
        return

    # PyTorch's own cache is empty at the baseline, as it is after the empty_cache() below:
    # free memory is comparable there too.
    empty_cache()
    free_baseline = free_bytes()

    # Two pools of the tag: a region with CPU backup and a region without.
    backed, plain = region_tensor(1, enable_cpu_backup=True), region_tensor(2)
    assert len(tag_pools()) == 2 and len(impl._mem_pools) == 3
    assert free_baseline - free_bytes() > 2 * _MIN_DELTA_BYTES

    # An open region uses its pool: the call raises, for the tag and for every tag. A caller
    # may keep such an error, which holds the frames of the refused call: these two live to
    # the end, and the pools dropped below must die all the same.
    with torch_memory_saver.region(tag=_TAG):
        refused = [_assert_raises(RuntimeError, _TAG), _assert_raises(RuntimeError, None)]
    assert len(tag_pools()) == 2 and len(impl._mem_pools) == 3
    assert _holds(backed, 1) and _holds(plain, 2)

    # The end of the tag: pause it, delete its tensors, drop its pools. Until they are
    # dropped the pools keep the paused segments.
    torch_memory_saver.pause(_TAG)
    assert free_baseline - free_bytes() < _MAX_NOISE_BYTES
    del backed, plain
    gc.collect()
    assert impl._affected_devices(_TAG) == [device_index]
    assert torch_memory_saver.release_region_pools(_TAG) == 2
    assert impl._affected_devices(_TAG) == []
    assert tag_pools() == [] and len(impl._mem_pools) == 1
    assert free_baseline - free_bytes() < _MAX_NOISE_BYTES
    assert torch_memory_saver.release_region_pools(_TAG) == 0

    # New regions with the same arguments allocate: each gets a new pool and new memory,
    # where a pool that was kept would have handed out its paused segment.
    backed, plain = region_tensor(3, enable_cpu_backup=True), region_tensor(4)
    assert len(tag_pools()) == 2
    assert impl._affected_devices(_TAG) == [device_index]
    assert free_baseline - free_bytes() > 2 * _MIN_DELTA_BYTES
    assert _holds(backed, 3) and _holds(plain, 4)

    # Memory that is not paused stays with the pools of deleted tensors too, and returns
    # to the device when the pools are dropped.
    del backed, plain
    gc.collect()
    assert free_baseline - free_bytes() > 2 * _MIN_DELTA_BYTES
    assert torch_memory_saver.release_region_pools(_TAG) == 2
    assert impl._affected_devices(_TAG) == []
    assert free_baseline - free_bytes() < _MAX_NOISE_BYTES

    # The segment of a live tensor outlives its pool, until the tensor is deleted and the
    # cache is emptied.
    held = region_tensor(5)
    assert torch_memory_saver.release_region_pools(_TAG) == 1 and tag_pools() == []
    assert impl._affected_devices(_TAG) == [device_index]
    assert free_baseline - free_bytes() > _MIN_DELTA_BYTES
    assert _holds(held, 5)
    del held
    gc.collect()
    empty_cache()
    assert impl._affected_devices(_TAG) == []
    assert free_baseline - free_bytes() < _MAX_NOISE_BYTES

    # The other tag was left alone all along. Without a tag the pools of every tag go: its
    # memory, which the baseline includes, returns to the device.
    assert impl._affected_devices(_OTHER_TAG) == [device_index] and _holds(bystander, 9)
    del bystander
    gc.collect()
    assert torch_memory_saver.release_region_pools() == 1
    assert not impl._mem_pools and impl._affected_devices(None) == []
    assert free_bytes() - free_baseline > _MIN_DELTA_BYTES
    assert _holds(normal_tensor, 100)
    # The errors of the refused calls were alive all along, with the frames of those calls.
    assert all(error.__traceback__ is not None for error in refused)


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
