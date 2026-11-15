"""set_chunk_backup(): which chunks a pause backs up.

Every chunk of a region with CPU backup starts enabled. Part 1 pauses a partly backed
tensor as it is made: each chunk that holds memory is copied, and the copies lie packed in
one host slot. Part 2 turns chunks off: the pause still returns the device memory of every
chunk, copies the enabled ones only, and the resume restores exactly those. Part 3 changes
the selection between pauses, down to none and back to all. Part 4 does so with retained
backups. Part 5 keeps one backup of a map_chunks() pair, on the destination and then on
the source; a destination chunk keeps its choice when it is mapped and when its source is
freed. Part 6: a refused call changes nothing. Part 7 is a tag with two allocations: its
chunks are numbered across them, and a chunk keeps its choice when it gets memory and when
the allocation before it is freed. run_host_bytes() reads the size of the host slot,
retained or not, from the memory of the process.
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG = "chunk_backup"
_TAG_SRC = "chunk_backup_src"
_TAG_DST = "chunk_backup_dst"
_TAG_UNBACKED = "chunk_backup_unbacked"
_TAG_SEGMENTS = "chunk_backup_segments"
_CHUNKS = 8
_HELD = (0, 1, 2, 5, 6)  # the chunks of _TAG that hold memory


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


def _anonymous_bytes() -> int:
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith("RssAnon:"))


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

    def fill(tensor, values):
        for index, value in values.items():
            part(tensor, index).fill_(value)

    def kept(tensor, values):
        return all(_holds(part(tensor, index), value) for index, value in values.items())

    def assert_backup(tensor, values):
        """The backup of the tensor is the chunks of `values` (chunk index: fill value) and
        no other, packed in index order: host memory for those chunks only."""
        addresses = []
        for index in range(tensor.numel() // chunk):
            copy = torch_memory_saver.get_cpu_backup(part(tensor, index), zero_copy=True)
            if index in values:
                assert copy is not None and _holds(copy, values[index]), index
                addresses.append(copy.data_ptr())
            else:
                assert copy is None, index
        assert addresses == [addresses[0] + i * chunk for i in range(len(addresses))]

    # A process's first device operations take device memory of their own: run them on an
    # ordinary tensor before the first reading.
    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)
    assert _holds(normal_tensor, 100)
    free_start = free_bytes()

    # Part 1: eight chunks of address space, five of them backed, paused as they are made.
    with torch_memory_saver.region(
        tag=_TAG, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        cache = torch.empty((_CHUNKS * chunk,), dtype=torch.uint8, device=device)
    torch_memory_saver.resume_chunks(_TAG, _HELD)
    expected = [i in _HELD for i in range(_CHUNKS)]
    held_bytes = len(_HELD) * chunk
    assert held_bytes <= free_start - free_bytes() < held_bytes + slack

    def values(base, chunks=_HELD):
        return {index: base + index for index in chunks}

    def pause_and_resume(tensor, backed_up):
        """Pauses _TAG, which returns the device memory of every chunk, checks that the
        backup of its tensor is the chunks of `backed_up`, and resumes it."""
        free_active = free_bytes()
        torch_memory_saver.pause(_TAG)
        assert states(_TAG) == [False] * _CHUNKS
        assert held_bytes - slack < free_bytes() - free_active < held_bytes + slack
        assert_backup(tensor, backed_up)
        torch_memory_saver.resume(_TAG)
        assert states(_TAG) == expected
        assert abs(free_bytes() - free_active) < slack
        assert kept(tensor, backed_up)

    fill(cache, values(10))
    pause_and_resume(cache, values(10))

    # Part 2: chunks 1 and 5 off, and chunk 7, which holds no memory.
    torch_memory_saver.set_chunk_backup(_TAG, [1, 5, 7], False)
    fill(cache, values(20))
    pause_and_resume(cache, values(20, (0, 2, 6)))

    # Part 3: the choice stays until it is changed.
    fill(cache, values(30))
    pause_and_resume(cache, values(30, (0, 2, 6)))
    torch_memory_saver.set_chunk_backup(_TAG, [1, 5], True)
    torch_memory_saver.set_chunk_backup(_TAG, [0, 2], False)
    fill(cache, values(40))
    pause_and_resume(cache, values(40, (1, 5, 6)))
    # No chunk: the pause takes no host memory.
    torch_memory_saver.set_chunk_backup(_TAG, range(_CHUNKS), False)
    pause_and_resume(cache, {})
    torch_memory_saver.set_chunk_backup(_TAG, range(_CHUNKS), True)
    fill(cache, values(50))
    pause_and_resume(cache, values(50))

    # Part 4: a retained backup is that of the last pause, whatever is chosen since, and
    # the next pause replaces it with its own selection. run_host_bytes() reads its size.
    torch_memory_saver.retain_cpu_backup = True
    torch_memory_saver.set_chunk_backup(_TAG, [0, 1], False)
    fill(cache, values(60))
    pause_and_resume(cache, values(60, (2, 5, 6)))
    torch_memory_saver.set_chunk_backup(_TAG, [5], False)
    assert_backup(cache, values(60, (2, 5, 6)))
    fill(cache, values(70))
    pause_and_resume(cache, values(70, (2, 6)))
    assert_backup(cache, values(70, (2, 6)))
    # Two other chunks are as many bytes: the next pause keeps the slot where it is and
    # writes chunks 0 and 5 where chunks 2 and 6 were.
    slot = torch_memory_saver.get_cpu_backup(part(cache, 2), zero_copy=True).data_ptr()
    torch_memory_saver.set_chunk_backup(_TAG, [2, 6], False)
    torch_memory_saver.set_chunk_backup(_TAG, [0, 5], True)
    fill(cache, values(110))
    pause_and_resume(cache, values(110, (0, 5)))
    assert_backup(cache, values(110, (0, 5)))
    assert torch_memory_saver.get_cpu_backup(part(cache, 0), zero_copy=True).data_ptr() == slot
    torch_memory_saver.set_chunk_backup(_TAG, range(_CHUNKS), False)
    pause_and_resume(cache, {})
    assert_backup(cache, {})
    torch_memory_saver.retain_cpu_backup = False
    torch_memory_saver.set_chunk_backup(_TAG, range(_CHUNKS), True)

    # Part 5: a source of four chunks, and a destination that backs two chunks itself and
    # maps source chunks 3 and 2 as its chunks 4 and 5. One backup of a pair is enough.
    free_pair = free_bytes()
    with torch_memory_saver.region(tag=_TAG_SRC, enable_cpu_backup=True, chunk_size=chunk):
        src = torch.empty((4 * chunk,), dtype=torch.uint8, device=device)
    with torch_memory_saver.region(
        tag=_TAG_DST, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        dst = torch.empty((6 * chunk,), dtype=torch.uint8, device=device)
    torch_memory_saver.resume_chunks(_TAG_DST, [0, 1])
    torch_memory_saver.map_chunks(_TAG_DST, [4, 5], _TAG_SRC, [3, 2])
    dst_states = [True, True, False, False, True, True]

    # On the destination only: the source's side is released without a copy and gets new
    # memory, into which the destination's backup is restored. The tags keep their order.
    torch_memory_saver.set_chunk_backup(_TAG_SRC, [2, 3], False)
    fill(src, {0: 10, 1: 11})
    fill(dst, {0: 1, 1: 2, 4: 77, 5: 66})
    _assert_raises(RuntimeError, torch_memory_saver.pause, _TAG_SRC)
    torch_memory_saver.pause(_TAG_DST)
    assert_backup(dst, {0: 1, 1: 2, 4: 77, 5: 66})
    torch_memory_saver.pause(_TAG_SRC)
    assert_backup(src, {0: 10, 1: 11})
    assert free_pair - free_bytes() < slack
    _assert_raises(RuntimeError, torch_memory_saver.resume, _TAG_DST)
    torch_memory_saver.resume(_TAG_SRC)
    assert states(_TAG_SRC) == [True] * 4 and kept(src, {0: 10, 1: 11})
    torch_memory_saver.resume(_TAG_DST)
    assert states(_TAG_DST) == dst_states
    assert 6 * chunk <= free_pair - free_bytes() < 6 * chunk + slack
    assert kept(dst, {0: 1, 1: 2, 4: 77, 5: 66})
    assert kept(src, {0: 10, 1: 11, 2: 66, 3: 77})

    # On the source only: the destination's side is unmapped without a copy and mapped
    # again at the resume, where it shows what the source's backup restored.
    torch_memory_saver.set_chunk_backup(_TAG_SRC, [2, 3], True)
    torch_memory_saver.set_chunk_backup(_TAG_DST, [4, 5], False)
    fill(dst, {4: 55})
    fill(src, {2: 44})
    torch_memory_saver.pause(_TAG_DST)
    assert_backup(dst, {0: 1, 1: 2})
    assert kept(src, {0: 10, 1: 11, 2: 44, 3: 55})
    torch_memory_saver.pause(_TAG_SRC)
    assert_backup(src, {0: 10, 1: 11, 2: 44, 3: 55})
    assert free_pair - free_bytes() < slack
    torch_memory_saver.resume(_TAG_SRC)
    torch_memory_saver.resume(_TAG_DST)
    assert states(_TAG_DST) == dst_states
    assert 6 * chunk <= free_pair - free_bytes() < 6 * chunk + slack
    assert kept(dst, {0: 1, 1: 2, 4: 55, 5: 44})
    assert kept(src, {0: 10, 1: 11, 2: 44, 3: 55})
    # Still one piece of memory at two addresses.
    part(src, 3).fill_(5)
    mod.synchronize()
    assert _holds(part(dst, 4), 5)

    # A chunk keeps its choice when map_chunks() gives it memory. Destination chunk 3 is
    # turned off and then maps source chunk 1; chunk 2, which maps source chunk 0, is on.
    torch_memory_saver.set_chunk_backup(_TAG_DST, [3], False)
    torch_memory_saver.map_chunks(_TAG_DST, [2, 3], _TAG_SRC, [0, 1])
    assert states(_TAG_DST) == [True] * 6
    torch_memory_saver.pause(_TAG_DST)
    assert_backup(dst, {0: 1, 1: 2, 2: 10})
    # It keeps it when it loses its source too. The source is freed while the destination
    # is paused, which takes chunk 2 out of that backup.
    impl = torch_memory_saver._impl
    del src
    for key in [key for key in impl._mem_pools if key[0] == _TAG_SRC]:
        del impl._mem_pools[key]
    gc.collect()
    empty_cache()
    assert torch_memory_saver.get_chunk_layout(_TAG_SRC) == []
    assert_backup(dst, {0: 1, 1: 2})
    torch_memory_saver.resume(_TAG_DST)
    assert states(_TAG_DST) == [True, True, False, False, False, False]
    assert kept(dst, {0: 1, 1: 2})
    # With memory of their own, chunk 2 is still on and chunk 3 still off.
    torch_memory_saver.resume_chunks(_TAG_DST, [2, 3])
    fill(dst, {2: 22, 3: 33})
    torch_memory_saver.pause(_TAG_DST)
    assert_backup(dst, {0: 1, 1: 2, 2: 22})
    torch_memory_saver.resume(_TAG_DST)
    assert states(_TAG_DST) == [True, True, True, True, False, False]
    assert kept(dst, {0: 1, 1: 2, 2: 22})

    # Part 6: refused calls. The first chunk of a refused list is not changed either.
    with torch_memory_saver.region(tag=_TAG_UNBACKED, chunk_size=chunk):
        unbacked = torch.empty((chunk,), dtype=torch.uint8, device=device)
    for error, tag, indices, enabled in (
        (IndexError, _TAG, [0, _CHUNKS], False),
        (IndexError, _TAG, [2**64], False),  # as a size_t it would be chunk 0
        (IndexError, "chunk_backup_no_such_tag", [0], False),
        (ValueError, _TAG_UNBACKED, [0], False),  # the region has no CPU backup
        (ValueError, _TAG_UNBACKED, [0], True),
    ):
        _assert_raises(error, torch_memory_saver.set_chunk_backup, tag, indices, enabled)
    fill(cache, values(80))
    torch_memory_saver.pause(_TAG)
    _assert_raises(RuntimeError, torch_memory_saver.set_chunk_backup, _TAG, [0], False)
    torch_memory_saver.resume(_TAG)
    pause_and_resume(cache, values(80))

    # Part 7: a tag with two allocations, like the segments of one pool. Chunks 0 and 1 are
    # the first's, chunks 2 to 5 the second's.
    with torch_memory_saver.region(tag=_TAG_SEGMENTS, enable_cpu_backup=True, chunk_size=chunk):
        first = torch.empty((2 * chunk,), dtype=torch.uint8, device=device)
    [first_pool] = [key for key in impl._mem_pools if key[0] == _TAG_SEGMENTS]
    with torch_memory_saver.region(
        tag=_TAG_SEGMENTS, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        second = torch.empty((4 * chunk,), dtype=torch.uint8, device=device)
    assert [address for address, _ in torch_memory_saver.get_chunk_layout(_TAG_SEGMENTS)] == (
        [first.data_ptr() + i * chunk for i in range(2)]
        + [second.data_ptr() + i * chunk for i in range(4)]
    )
    torch_memory_saver.resume_chunks(_TAG_SEGMENTS, [2, 3])

    # Off: a chunk of each allocation, and chunk 4, which holds no memory yet and stays off
    # when it gets some.
    torch_memory_saver.set_chunk_backup(_TAG_SEGMENTS, [1, 3, 4], False)
    torch_memory_saver.resume_chunks(_TAG_SEGMENTS, [4])
    assert states(_TAG_SEGMENTS) == [True] * 5 + [False]
    fill(first, {0: 90, 1: 91})
    fill(second, {0: 92, 1: 93, 2: 94})
    torch_memory_saver.pause(_TAG_SEGMENTS)
    assert states(_TAG_SEGMENTS) == [False] * 6
    assert_backup(first, {0: 90})
    assert_backup(second, {0: 92})
    torch_memory_saver.resume(_TAG_SEGMENTS)
    assert states(_TAG_SEGMENTS) == [True] * 5 + [False]
    assert kept(first, {0: 90}) and kept(second, {0: 92})

    # Freeing the first allocation numbers the chunks of the second from 0. Each keeps its
    # choice, and the call takes the new numbers.
    del first, impl._mem_pools[first_pool]
    gc.collect()
    empty_cache()
    assert torch_memory_saver.get_chunk_layout(_TAG_SEGMENTS) == [
        (second.data_ptr() + i * chunk, chunk) for i in range(4)
    ]
    assert states(_TAG_SEGMENTS) == [True, True, True, False]
    _assert_raises(IndexError, torch_memory_saver.set_chunk_backup, _TAG_SEGMENTS, [0, 4], False)
    fill(second, {0: 95, 1: 96, 2: 97})
    torch_memory_saver.pause(_TAG_SEGMENTS)
    assert_backup(second, {0: 95})
    torch_memory_saver.resume(_TAG_SEGMENTS)
    assert kept(second, {0: 95})
    torch_memory_saver.set_chunk_backup(_TAG_SEGMENTS, [1], True)
    fill(second, {1: 98})
    torch_memory_saver.pause(_TAG_SEGMENTS)
    assert_backup(second, {0: 95, 1: 98})
    torch_memory_saver.resume(_TAG_SEGMENTS)
    assert states(_TAG_SEGMENTS) == [True, True, True, False]
    assert kept(second, {0: 95, 1: 98})
    assert _holds(normal_tensor, 100)

    # Return the regions' memory before the interpreter exits: the tensors, then their pools.
    del cache, dst, unbacked, second
    torch_memory_saver._impl._mem_pools.clear()
    gc.collect()
    empty_cache()


def run_host_bytes(hook_mode: str):
    """The host slot of the mmap backend is anonymous memory of the process: a pause takes
    as much of it as the enabled chunks hold, and the resume gives it back unless backups
    are retained."""
    torch_memory_saver.hook_mode = hook_mode
    device = get_device()
    mod = get_device_module()
    chunk = 32 * torch_memory_saver.get_allocation_granularity()
    slack = chunk // 2

    with torch_memory_saver.region(
        tag=_TAG, enable_cpu_backup=True, cpu_backup_backend="mmap", chunk_size=chunk
    ):
        tensor = torch.full((_CHUNKS * chunk,), 7, dtype=torch.uint8, device=device)

    def host_bytes_of_pause():
        mod.synchronize()
        active = _anonymous_bytes()
        torch_memory_saver.pause(_TAG)
        taken = _anonymous_bytes() - active
        torch_memory_saver.resume(_TAG)
        mod.synchronize()
        assert abs(_anonymous_bytes() - active) < slack
        return taken

    # What the process takes once for its first copy to the host is not measured.
    torch_memory_saver.pause(_TAG)
    torch_memory_saver.resume(_TAG)
    assert abs(host_bytes_of_pause() - _CHUNKS * chunk) < slack
    assert _holds(tensor, 7)
    torch_memory_saver.set_chunk_backup(_TAG, [0, 1, 3, 4, 7], False)
    assert abs(host_bytes_of_pause() - 3 * chunk) < slack
    assert all(_holds(tensor[i * chunk : (i + 1) * chunk], 7) for i in (2, 5, 6))
    torch_memory_saver.set_chunk_backup(_TAG, [2, 5, 6], False)
    assert abs(host_bytes_of_pause()) < slack

    # A retained slot outlives the resume. The next pause makes it the size of its own
    # selection, smaller as well as larger, keeps it for other chunks of the same size, and
    # frees it when no chunk is enabled.
    torch_memory_saver.retain_cpu_backup = True
    mod.synchronize()
    without_slot = _anonymous_bytes()

    def retained_bytes(enabled, value):
        tensor.fill_(value)
        torch_memory_saver.set_chunk_backup(_TAG, range(_CHUNKS), False)
        torch_memory_saver.set_chunk_backup(_TAG, enabled, True)
        torch_memory_saver.pause(_TAG)
        torch_memory_saver.resume(_TAG)
        mod.synchronize()
        assert all(_holds(tensor[i * chunk : (i + 1) * chunk], value) for i in enabled)
        return _anonymous_bytes() - without_slot

    assert abs(retained_bytes([2, 5, 6], 1) - 3 * chunk) < slack
    assert abs(retained_bytes([2, 6], 2) - 2 * chunk) < slack
    assert abs(retained_bytes([0, 5], 5) - 2 * chunk) < slack
    assert abs(retained_bytes([0, 1, 2, 6], 3) - 4 * chunk) < slack
    assert abs(retained_bytes([], 4)) < slack

    del tensor
    torch_memory_saver._impl._mem_pools.clear()
    gc.collect()
    empty_cache()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
