"""A region whose allocations reserve address space only.

Inside region(initially_released=True) a tensor made with torch.empty gets an address and
no device memory. resume(tag) gives it memory at that address. From then on the tag is
paused and resumed like any other: with CPU backup the contents survive.
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG = "initially_released"
_TENSOR_BYTES = 256 * 1024**2
_MIN_DELTA_BYTES = 0.9 * _TENSOR_BYTES  # expected rise/drop, with slack for alignment/noise
_MAX_NOISE_BYTES = 0.1 * _TENSOR_BYTES


def _holds(tensor: torch.Tensor, value: int) -> bool:
    return int(tensor.min()) == value == int(tensor.max())


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device = get_device()
    mod = get_device_module()
    device_index = mod.current_device()

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)
    free_before = free_bytes()

    with torch_memory_saver.region(tag=_TAG, enable_cpu_backup=True, initially_released=True):
        tensor = torch.empty((_TENSOR_BYTES,), dtype=torch.uint8, device=device)
    address = tensor.data_ptr()
    assert free_before - free_bytes() < _MAX_NOISE_BYTES

    # The first resume has no backup to restore.
    torch_memory_saver.resume(_TAG)
    assert free_before - free_bytes() > _MIN_DELTA_BYTES
    assert tensor.data_ptr() == address
    tensor.fill_(7)
    assert _holds(tensor, 7)

    torch_memory_saver.pause(_TAG)
    assert free_before - free_bytes() < _MAX_NOISE_BYTES
    assert _holds(torch_memory_saver.get_cpu_backup(tensor, zero_copy=True), 7)
    torch_memory_saver.resume(_TAG)
    assert free_before - free_bytes() > _MIN_DELTA_BYTES
    assert tensor.data_ptr() == address
    assert _holds(tensor, 7) and _holds(normal_tensor, 100)

    # Return the region's memory before the interpreter exits: the tensor, then its pool.
    del tensor
    torch_memory_saver._impl._mem_pools.clear()
    gc.collect()
    empty_cache()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
