"""resume() when the device has no memory for the paused allocations.

A 1 GiB region with CPU backup is paused, and an ordinary tensor then takes the device
memory. The region is two tensors, hence two allocations, and the ordinary tensor leaves
room for one of them: resume() obtains memory for one, is refused the other and has to give
the first back. It must raise torch.OutOfMemoryError and leave everything as it was: device
memory unchanged, both backups still served. Once the ordinary tensor is gone, the same
call succeeds and both tensors hold their data.

The hook used to end the process there, so the exit status is part of the result and the
test runs this file as a script.
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG = "resume_oom"
_TENSOR_BYTES = 512 * 1024**2
_REGION_BYTES = 2 * _TENSOR_BYTES
_LEFT_FREE_BYTES = 768 * 1024**2  # room for one of the two tensors
_MIN_DELTA_BYTES = 0.9 * _REGION_BYTES  # expected rise/drop, with slack for alignment/noise


def _pattern(offset: int) -> torch.Tensor:
    period = torch.arange(251, dtype=torch.uint8).roll(offset)
    return period.repeat(_TENSOR_BYTES // 251 + 1)[:_TENSOR_BYTES]


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device = get_device()
    mod = get_device_module()
    device_index = mod.current_device()

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    expected = [_pattern(0), _pattern(100)]
    with torch_memory_saver.region(tag=_TAG, enable_cpu_backup=True):
        tensors = [pattern.to(device) for pattern in expected]
    free_active = free_bytes()

    torch_memory_saver.pause(_TAG)
    free_paused = free_bytes()
    assert free_paused - free_active > _MIN_DELTA_BYTES

    filler = torch.empty((free_paused - _LEFT_FREE_BYTES,), dtype=torch.uint8, device=device)
    memory_info_full = mod.mem_get_info(device_index)
    assert memory_info_full[0] < _REGION_BYTES

    try:
        torch_memory_saver.resume(_TAG)
    except torch.OutOfMemoryError as error:
        print(f"resume raised: {error}")
    else:
        raise AssertionError("resume() returned although the device has no room for the region")
    assert mod.mem_get_info(device_index) == memory_info_full
    assert all(
        torch.equal(torch_memory_saver.get_cpu_backup(tensor, zero_copy=True), pattern)
        for tensor, pattern in zip(tensors, expected)
    )

    del filler
    empty_cache()
    free_emptied = free_bytes()
    torch_memory_saver.resume(_TAG)
    assert free_emptied - free_bytes() > _MIN_DELTA_BYTES
    assert all(torch.equal(tensor.cpu(), pattern) for tensor, pattern in zip(tensors, expected))

    # Return the region's memory before the interpreter exits: the tensors, then their pool.
    del tensors
    torch_memory_saver._impl._mem_pools.clear()
    gc.collect()
    empty_cache()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
