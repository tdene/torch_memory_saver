"""map_chunks(): one piece of device memory at two addresses.

A source tag owns four chunks. A destination tag reserves six chunks of address space,
backs two of them itself and maps two chunks of the source at its tail. What is written
through one address is read through the other. The destination is paused before the
source and resumed after it; the other order raises and changes nothing. Freeing the
source takes its memory out of the destination.
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG_SRC = "map_chunks_src"
_TAG_DST = "map_chunks_dst"


def _holds(tensor: torch.Tensor, value: int) -> bool:
    return int(tensor.min()) == value == int(tensor.max())


def _assert_raises(error, fn, *args):
    try:
        fn(*args)
    except error as caught:
        assert type(caught) is error, type(caught)
        print(f"{fn.__name__}{args} raised {error.__name__}: {caught}")
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise {error.__name__}")


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device = get_device()
    mod = get_device_module()
    device_index = mod.current_device()
    chunk = 32 * torch_memory_saver.get_allocation_granularity()
    slack = chunk // 2  # alignment/noise allowed in a device memory reading

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    def states(tag):
        return torch_memory_saver.get_chunk_states(tag)

    def part(tensor, index):
        return tensor[index * chunk : (index + 1) * chunk]

    free_start = free_bytes()
    with torch_memory_saver.region(tag=_TAG_SRC, enable_cpu_backup=True, chunk_size=chunk):
        src = torch.empty((4 * chunk,), dtype=torch.uint8, device=device)
    for i in range(4):
        part(src, i).fill_(10 + i)
    with torch_memory_saver.region(
        tag=_TAG_DST, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        dst = torch.empty((6 * chunk,), dtype=torch.uint8, device=device)
    torch_memory_saver.resume_chunks(_TAG_DST, [0, 1])
    dst[: 2 * chunk].fill_(1)
    free_backed = free_bytes()
    assert 6 * chunk <= free_start - free_backed < 6 * chunk + slack

    # Destination chunk 4 is source chunk 3, destination chunk 5 is source chunk 2. The
    # second address takes no device memory.
    torch_memory_saver.map_chunks(_TAG_DST, [4, 5], _TAG_SRC, [3, 2])
    expected = [True, True, False, False, True, True]
    assert states(_TAG_DST) == expected and states(_TAG_SRC) == [True] * 4
    assert abs(free_backed - free_bytes()) < slack
    assert _holds(part(dst, 4), 13) and _holds(part(dst, 5), 12)
    part(dst, 4).fill_(77)
    part(src, 2).fill_(66)
    mod.synchronize()
    assert _holds(part(src, 3), 77) and _holds(part(dst, 5), 66)
    assert _holds(part(src, 0), 10) and _holds(part(src, 1), 11) and _holds(dst[: 2 * chunk], 1)

    # Refused calls change nothing, the first pair of a refused list included.
    for error, args in (
        (ValueError, ([4], _TAG_SRC, [0])),  # the destination is mapped already
        (ValueError, ([2, 0], _TAG_SRC, [0, 1])),  # the second destination holds memory
        (ValueError, ([2], _TAG_DST, [3])),  # the source holds no memory
        (ValueError, ([2], _TAG_DST, [4])),  # the source's memory is not its own
        (ValueError, ([2, 3], _TAG_SRC, [0])),  # lists of different lengths
        (IndexError, ([2], _TAG_SRC, [4])),
    ):
        _assert_raises(error, torch_memory_saver.map_chunks, _TAG_DST, *args)
    assert states(_TAG_DST) == expected and abs(free_backed - free_bytes()) < slack

    # The source cannot be paused while the destination maps its chunks.
    _assert_raises(RuntimeError, torch_memory_saver.pause, _TAG_SRC)
    assert states(_TAG_SRC) == [True] * 4 and _holds(part(dst, 4), 77)

    # Pausing the destination returns its own two chunks; the mapped memory stays with
    # the source, and its contents are in the destination's backup.
    torch_memory_saver.pause(_TAG_DST)
    assert states(_TAG_DST) == [False] * 6
    assert 2 * chunk - slack < free_bytes() - free_backed < 2 * chunk + slack
    assert _holds(part(src, 3), 77) and _holds(part(src, 2), 66)
    assert _holds(torch_memory_saver.get_cpu_backup(part(dst, 4), zero_copy=True), 77)
    torch_memory_saver.pause(_TAG_SRC)
    assert free_start - free_bytes() < slack

    # The destination cannot be resumed while the source holds no memory.
    _assert_raises(RuntimeError, torch_memory_saver.resume, _TAG_DST)
    assert states(_TAG_DST) == [False] * 6 and free_start - free_bytes() < slack
    torch_memory_saver.resume(_TAG_SRC)
    assert all(_holds(part(src, i), value) for i, value in enumerate((10, 11, 66, 77)))
    torch_memory_saver.resume(_TAG_DST)
    assert states(_TAG_DST) == expected
    assert 6 * chunk <= free_start - free_bytes() < 6 * chunk + slack
    assert _holds(dst[: 2 * chunk], 1) and _holds(part(dst, 4), 77) and _holds(part(dst, 5), 66)

    # Still one piece of memory at two addresses: the memory the source holds now.
    part(src, 3).fill_(5)
    mod.synchronize()
    assert _holds(part(dst, 4), 5)

    # Freeing the source takes its memory out of the destination.
    impl = torch_memory_saver._impl
    del src
    for key in [key for key in impl._mem_pools if key[0] == _TAG_SRC]:
        del impl._mem_pools[key]
    gc.collect()
    empty_cache()
    assert impl._affected_devices(_TAG_SRC) == []
    assert states(_TAG_DST) == [True, True, False, False, False, False]
    assert 2 * chunk <= free_start - free_bytes() < 2 * chunk + slack
    assert _holds(dst[: 2 * chunk], 1)

    # Return the region's memory before the interpreter exits: the tensor, then its pool.
    del dst
    impl._mem_pools.clear()
    gc.collect()
    empty_cache()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
