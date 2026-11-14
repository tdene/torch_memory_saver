"""A region allocation that the device has no memory for, in hook_mode="torch".

PyTorch asks the region's pool for a segment and the pool asks the hook. For a tensor
larger than the free device memory the hook has no segment and hands PyTorch a null one,
which is PyTorch's out of memory: the allocation must raise torch.OutOfMemoryError. The
process goes on: the region is left like any other, the hook holds nothing of the refused
request, and a small tensor allocated in a region of the same tag is hook memory that
holds its data.

Before it raises, PyTorch may empty its caches and ask again, depending on its version.
Nothing here depends on that.

The hook used to end the process at such a request, so the exit status is part of the
result and the test runs this file as a script:
  python test/examples/malloc_oom.py torch
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG = "malloc_oom"
_EXCESS_BYTES = 1024**3  # how far beyond the free device memory the refused tensor goes
_SMALL_BYTES = 256 * 1024**2
_MIN_DELTA_BYTES = 0.9 * _SMALL_BYTES  # expected drop, with slack for alignment/noise


def run(hook_mode: str):
    assert hook_mode == "torch", "a null segment is the out of memory of hook_mode='torch'"
    torch_memory_saver.hook_mode = hook_mode

    device = get_device()
    mod = get_device_module()
    device_index = mod.current_device()
    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    too_large = free_bytes() + _EXCESS_BYTES
    try:
        with torch_memory_saver.region(tag=_TAG):
            torch.empty((too_large,), dtype=torch.uint8, device=device)
    except torch.OutOfMemoryError as error:
        print(f"The region allocation raised: {error}")
    else:
        raise AssertionError(f"a region allocation of {too_large} bytes returned")

    impl = torch_memory_saver._impl
    assert not impl._binary_wrapper.cdll.tms_get_interesting_region()
    assert impl._affected_devices(_TAG) == []
    assert normal_tensor[:3].tolist() == [100, 100, 100]

    free_before_small = free_bytes()
    with torch_memory_saver.region(tag=_TAG):
        small_tensor = torch.full((_SMALL_BYTES,), 7, dtype=torch.uint8, device=device)
    assert impl._affected_devices(_TAG) == [device_index]
    assert free_before_small - free_bytes() > _MIN_DELTA_BYTES
    assert bool(torch.all(small_tensor == 7))
    assert normal_tensor[:3].tolist() == [100, 100, 100]

    # Return the region's memory before the interpreter exits: the tensor, then its pool.
    del small_tensor
    impl._mem_pools.clear()
    gc.collect()
    empty_cache()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
