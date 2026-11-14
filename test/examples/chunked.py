"""Chunked regions, addressed by tag.

A region with chunk_size backs each allocation with one device handle per chunk. Part 1
checks the layout of a tag with two allocations, whose chunks are numbered across them in
allocation order, and that pause() and resume() treat the allocations as wholes. Part 2
builds a partly backed tensor: an initially released chunked region reserves the address
range, resume_chunks() gives memory to the chunks that are used, and a pause and resume
restore exactly those chunks with their contents. Part 3 fills the device: resume_chunks()
must raise torch.OutOfMemoryError and change nothing.
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG_FULL = "chunked_full"
_TAG_PART = "chunked_part"
_PART_CHUNKS = 8


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
    granularity = torch_memory_saver.get_allocation_granularity()
    chunk = 32 * granularity
    slack = chunk // 2  # alignment/noise allowed in a device memory reading

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    def states(tag):
        return torch_memory_saver.get_chunk_states(tag)

    # Part 1: an allocation of five chunks and a short sixth, then one of two chunks, all
    # backed. The period of the pattern is no divisor of the chunk size, so a chunk restored
    # to the wrong place shows.
    full_bytes = 5 * chunk + granularity
    pattern = torch.arange(251, dtype=torch.uint8).repeat(full_bytes // 251 + 1)[:full_bytes]
    free_start = free_bytes()
    with torch_memory_saver.region(tag=_TAG_FULL, enable_cpu_backup=True, chunk_size=chunk):
        full = pattern.to(device)
        second = torch.full((2 * chunk,), 5, dtype=torch.uint8, device=device)
    layout = torch_memory_saver.get_chunk_layout(_TAG_FULL)
    assert [size for _, size in layout] == [chunk] * 5 + [granularity] + [chunk] * 2, layout
    assert [address for address, _ in layout] == (
        [full.data_ptr() + i * chunk for i in range(6)]
        + [second.data_ptr() + i * chunk for i in range(2)]
    )
    assert states(_TAG_FULL) == [True] * 8
    assert free_start - free_bytes() > full_bytes + 2 * chunk - slack

    torch_memory_saver.pause(_TAG_FULL)
    assert states(_TAG_FULL) == [False] * 8
    assert free_start - free_bytes() < slack
    assert torch.equal(torch_memory_saver.get_cpu_backup(full, zero_copy=True), pattern)
    torch_memory_saver.resume(_TAG_FULL)
    assert states(_TAG_FULL) == [True] * 8
    assert torch_memory_saver.get_chunk_layout(_TAG_FULL) == layout
    assert torch.equal(full.cpu(), pattern) and _holds(second, 5)

    # Part 2: eight chunks of address space, three of them backed.
    free_before = free_bytes()
    with torch_memory_saver.region(
        tag=_TAG_PART, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        part = torch.empty((_PART_CHUNKS * chunk,), dtype=torch.uint8, device=device)
    assert states(_TAG_PART) == [False] * _PART_CHUNKS
    assert free_before - free_bytes() < slack

    torch_memory_saver.resume_chunks(_TAG_PART, [0, 1, 5])
    expected = [i in (0, 1, 5) for i in range(_PART_CHUNKS)]
    assert states(_TAG_PART) == expected
    assert 3 * chunk <= free_before - free_bytes() < 3 * chunk + slack
    part[: 2 * chunk].fill_(3)
    part[5 * chunk : 6 * chunk].fill_(9)

    # A listed chunk that has memory is passed by; a bad index changes nothing.
    free_backed = free_bytes()
    torch_memory_saver.resume_chunks(_TAG_PART, [0])
    _assert_raises(IndexError, torch_memory_saver.resume_chunks, _TAG_PART, [2, _PART_CHUNKS])
    assert states(_TAG_PART) == expected and free_bytes() == free_backed

    torch_memory_saver.pause(_TAG_PART)
    assert states(_TAG_PART) == [False] * _PART_CHUNKS
    assert free_before - free_bytes() < slack
    _assert_raises(RuntimeError, torch_memory_saver.resume_chunks, _TAG_PART, [2])
    torch_memory_saver.resume(_TAG_PART)
    assert states(_TAG_PART) == expected
    assert 3 * chunk <= free_before - free_bytes() < 3 * chunk + slack
    assert _holds(part[: 2 * chunk], 3) and _holds(part[5 * chunk : 6 * chunk], 9)

    # Part 3: the device has room for one more chunk, and two are asked for.
    filler = torch.empty((free_bytes() - chunk - slack,), dtype=torch.uint8, device=device)
    memory_info_full = mod.mem_get_info(device_index)
    _assert_raises(torch.OutOfMemoryError, torch_memory_saver.resume_chunks, _TAG_PART, [2, 3])
    assert states(_TAG_PART) == expected
    assert mod.mem_get_info(device_index) == memory_info_full
    del filler
    empty_cache()
    torch_memory_saver.resume_chunks(_TAG_PART, [2, 3])
    expected[2] = expected[3] = True
    assert states(_TAG_PART) == expected
    assert _holds(part[: 2 * chunk], 3) and torch.equal(full.cpu(), pattern)

    # Return the regions' memory before the interpreter exits: the tensors, then their pools.
    del full, second, part
    torch_memory_saver._impl._mem_pools.clear()
    gc.collect()
    empty_cache()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
