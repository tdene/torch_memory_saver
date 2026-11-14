"""Free a region's memory with no region active, in hook_mode="torch".

PyTorch returns a pool's cached segments through the pool's allocator when the
pool is destroyed and, from 2.8, when a cache is emptied, normally on a thread
that is in no region then. The tensor is deleted before its pool here, so the
segment is returned while the pool is destroyed, on PyTorch 2.6 and 2.7 too.

The exit status is part of the result, so run it as a script:
  python test/examples/free_outside_region.py torch
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import (
    empty_cache,
    get_and_print_gpu_memory,
    get_device,
    get_device_module,
)

_TAG = "free_outside_region"
_ALLOC_BYTES = 1024**3
_MIN_DELTA_BYTES = 0.9 * 1024**3  # expected rise/drop, with slack for alignment/noise


def run(hook_mode: str):
    assert hook_mode == "torch", "free outside a region requires hook_mode='torch'"
    torch_memory_saver.hook_mode = hook_mode

    device = get_device()
    device_index = get_device_module().current_device()
    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)
    mem_before_region = get_and_print_gpu_memory("Before region", gpu_id=device_index)

    with torch_memory_saver.region(tag=_TAG):
        pauseable_tensor = torch.full((_ALLOC_BYTES,), 100, dtype=torch.uint8, device=device)
    mem_after_region = get_and_print_gpu_memory("After region", gpu_id=device_index)
    assert mem_after_region - mem_before_region > _MIN_DELTA_BYTES

    impl = torch_memory_saver._impl
    in_region = impl._binary_wrapper.cdll.tms_get_interesting_region
    assert not in_region()
    assert impl._affected_devices(_TAG) == [device_index]

    del pauseable_tensor
    # MemPool key of a region without backup: (tag, cpu, disk, backend, device).
    impl._mem_pools.pop((_TAG, False, False, "", device_index))
    gc.collect()
    empty_cache()

    assert not in_region()
    assert impl._affected_devices(_TAG) == []
    mem_after_free = get_and_print_gpu_memory("After free outside a region", gpu_id=device_index)
    assert mem_after_region - mem_after_free > _MIN_DELTA_BYTES
    assert normal_tensor[:3].tolist() == [100, 100, 100]


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
