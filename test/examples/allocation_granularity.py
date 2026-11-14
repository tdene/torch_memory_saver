"""get_allocation_granularity() against the driver.

Every device reports a positive power of two. It is the unit a device allocation is made
of: in hook_mode="preload" a cudaMalloc of one byte inside a region, called through ctypes
as cuda_vmm_granularity.py does, takes exactly that much device memory and cudaFree gives
it back.

The query is the first statement here that needs the driver, so it has to see to the
driver's initialisation itself, and the hook ends the process when the driver refuses the
query. The exit status is part of the result, so run it as a script:
  python test/examples/allocation_granularity.py torch
"""

import ctypes
import sys

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_device_module

_TAG = "allocation_granularity"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _assert_one_byte_takes_one_granule(mod, device_index: int, granularity: int):
    # The process's cudaMalloc and cudaFree are the preloaded hook's.
    api = ctypes.CDLL(None)
    api.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    api.cudaFree.argtypes = [ctypes.c_void_p]
    impl = torch_memory_saver._impl

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    ptr = ctypes.c_void_p()
    with torch_memory_saver.region(tag=_TAG):
        free_before = free_bytes()
        assert impl._affected_devices(_TAG) == []
        assert api.cudaMalloc(ctypes.byref(ptr), 1) == 0
        free_allocated = free_bytes()
    assert ptr.value is not None
    # The hook made the allocation, not the allocator it forwards to outside a region.
    assert impl._affected_devices(_TAG) == [device_index]
    assert free_before - free_allocated == granularity, (free_before, free_allocated)

    assert api.cudaFree(ptr) == 0
    assert impl._affected_devices(_TAG) == []
    assert free_bytes() == free_before


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode

    granularity = torch_memory_saver.get_allocation_granularity()
    assert type(granularity) is int and _is_power_of_two(granularity), granularity

    mod = get_device_module()
    device_index = mod.current_device()
    device_count = mod.device_count()
    assert torch_memory_saver.get_allocation_granularity(device_index) == granularity
    for index in range(device_count):
        value = torch_memory_saver.get_allocation_granularity(index)
        print(f"Allocation granularity of device {index}: {value} bytes")
        assert _is_power_of_two(value), value
    for unknown in (device_count, -1):
        try:
            torch_memory_saver.get_allocation_granularity(unknown)
        except ValueError as error:
            print(f"Device {unknown} is refused: {error}")
        else:
            raise AssertionError(f"device {unknown} of {device_count} was accepted")

    if hook_mode == "preload":
        _assert_one_byte_takes_one_granule(mod, device_index, granularity)


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
