"""pause() and resume() copy a chunk backup beside their mapping calls.

A pause copies the chunks that it backs up on a thread of its own, at most 1 GiB a copy,
while the calling thread unmaps and releases: a chunk goes once its copy has returned, a
chunk without a copy at once. A resume maps while that thread copies back what is mapped
already. The bytes must come through as they do with one thread.

A source tag owns a line of chunks: a run that takes two copies, with one chunk in both of
them, and two short runs, each behind a chunk that the source does not back up. A
destination tag backs three chunks itself and maps three chunks of the source: two that the
source backs up and the destination does not, and one, next to a chunk of its own, that
only the destination backs up. A cycle writes new bytes through both tags, pauses them,
reads both backups, resumes them and reads the bytes through both addresses. It takes the
tags one by one, the destination before the source and back in the other order, or both in
one call, which then copies for two allocations. Two cycles free the backups at the resume
and two retain them; in the last cycle the device is full at the first resume of the
source, which raises torch.OutOfMemoryError and leaves the source paused with its backup.

A fault under a copy ends the process, so the exit status is part of the result and the
test runs this file as a script.
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device, get_device_module

_TAG_SRC = "chunk_backup_copy_src"
_TAG_DST = "chunk_backup_copy_dst"
_COPY_BYTES = 1 << 30  # kBackupCopyPieceBytes in csrc/core.cpp: the most that one copy moves
_DST_CHUNKS = 16
_DST_OWN = (0, 1, 4)
_DST_BACKED = (*_DST_OWN, 5)


def _pattern(nbytes: int, start: int) -> torch.Tensor:
    """Byte i is (start + i) % 251: bytes that land a chunk or a copy too far do not match."""
    period = torch.arange(251, dtype=torch.uint8).roll(-(start % 251))
    return period.repeat(nbytes // 251 + 1)[:nbytes]


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device = get_device()
    mod = get_device_module()
    device_index = mod.current_device()
    granularity = torch_memory_saver.get_allocation_granularity()
    chunk = 12 * granularity  # no divisor of a copy: one chunk of the long run lies in two
    slack = 16 * granularity  # alignment/noise allowed in a device memory reading

    # The source's chunks: the long run, then a chunk that only the destination backs up,
    # a short run that the destination maps, a chunk without a backup, a short run.
    long_run = _COPY_BYTES // chunk + 2
    only_dst, mapped, no_backup = long_run, (long_run + 1, long_run + 2), long_run + 3
    src_chunks = long_run + 6
    src_backed = [i for i in range(src_chunks) if i not in (only_dst, no_backup)]
    dst_states = [i < 6 for i in range(_DST_CHUNKS)]

    def free_bytes():
        return mod.mem_get_info(device_index)[0]

    def states(tag):
        return torch_memory_saver.get_chunk_states(tag)

    def part(tensor, index):
        return tensor[index * chunk : (index + 1) * chunk]

    def holds(tensor, index, expected):
        return torch.equal(part(tensor, index).cpu(), expected)

    def assert_backup(tensor, indices, expected):
        """The backup of the tensor is the chunks `indices`, chunk i with the bytes
        expected(i), and no other, packed in index order."""
        addresses = []
        for index in range(tensor.numel() // chunk):
            copy = torch_memory_saver.get_cpu_backup(part(tensor, index), zero_copy=True)
            if index in indices:
                assert copy is not None and torch.equal(copy, expected(index)), index
                addresses.append(copy.data_ptr())
            else:
                assert copy is None, index
        assert addresses == [addresses[0] + i * chunk for i in range(len(addresses))]

    # A process's first device operations take device memory of their own: run them on an
    # ordinary tensor before the first reading.
    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device=device)
    assert int(normal_tensor.min()) == 100
    free_start = free_bytes()
    with torch_memory_saver.region(tag=_TAG_SRC, enable_cpu_backup=True, chunk_size=chunk):
        src = torch.empty((src_chunks * chunk,), dtype=torch.uint8, device=device)
    with torch_memory_saver.region(
        tag=_TAG_DST, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        dst = torch.empty((_DST_CHUNKS * chunk,), dtype=torch.uint8, device=device)
    torch_memory_saver.resume_chunks(_TAG_DST, _DST_OWN)
    torch_memory_saver.map_chunks(_TAG_DST, [2, 3, 5], _TAG_SRC, [*mapped, only_dst])
    torch_memory_saver.set_chunk_backup(_TAG_SRC, [only_dst, no_backup], False)
    torch_memory_saver.set_chunk_backup(_TAG_DST, [2, 3], False)
    assert states(_TAG_SRC) == [True] * src_chunks and states(_TAG_DST) == dst_states
    held_bytes = (src_chunks + len(_DST_OWN)) * chunk
    assert held_bytes <= free_start - free_bytes() < held_bytes + slack

    def cycle(number, retained, together=False, device_full=False):
        """Writes the bytes of this cycle, pauses and resumes both tags, in one call each
        with `together`, and checks the backups and the bytes at both addresses. With
        `device_full` the first resume of the source runs out of device memory."""
        torch_memory_saver.retain_cpu_backup = retained

        def src_bytes(index):
            return _pattern(chunk, 17 * number + index * chunk)

        def dst_bytes(index):
            return _pattern(chunk, 17 * number + 101 + index * chunk)

        def shared_bytes_at_both_addresses():
            return all(
                holds(src, src_index, expected) and holds(dst, dst_index, expected)
                for src_index, dst_index, expected in (
                    (only_dst, 5, dst_bytes(5)),
                    *((index, 2 + i, src_bytes(index)) for i, index in enumerate(mapped)),
                )
            )

        for index in range(src_chunks):
            part(src, index).copy_(src_bytes(index))
        for index in _DST_BACKED:
            part(dst, index).copy_(dst_bytes(index))
        mod.synchronize()
        assert shared_bytes_at_both_addresses()
        free_active = free_bytes()

        if together:
            torch_memory_saver.pause()
        else:
            # The destination's own memory returns. What it maps stays with the source,
            # which reads it at its own address.
            torch_memory_saver.pause(_TAG_DST)
            assert states(_TAG_DST) == [False] * _DST_CHUNKS
            returned = len(_DST_OWN) * chunk
            assert returned - slack < free_bytes() - free_active < returned + slack
            assert holds(src, only_dst, dst_bytes(5))
            assert all(holds(src, index, src_bytes(index)) for index in mapped)
            torch_memory_saver.pause(_TAG_SRC)
        assert states(_TAG_SRC) == [False] * src_chunks
        assert states(_TAG_DST) == [False] * _DST_CHUNKS
        assert free_start - free_bytes() < slack
        assert_backup(src, src_backed, src_bytes)
        assert_backup(dst, _DST_BACKED, dst_bytes)

        if device_full:
            # Room for half of the source: resume() is given memory for some of its chunks,
            # is refused the others and gives the first back.
            filler = torch.empty(
                (free_bytes() - src_chunks * chunk // 2,), dtype=torch.uint8, device=device
            )
            memory_info_full = mod.mem_get_info(device_index)
            assert memory_info_full[0] < src_chunks * chunk
            try:
                torch_memory_saver.resume(_TAG_SRC)
            except torch.OutOfMemoryError as error:
                print(f"resume raised: {error}")
            else:
                raise AssertionError("resume() returned although the device has no room")
            assert mod.mem_get_info(device_index) == memory_info_full
            assert states(_TAG_SRC) == [False] * src_chunks
            assert_backup(src, src_backed, src_bytes)
            del filler
            empty_cache()

        if together:
            torch_memory_saver.resume()
        else:
            torch_memory_saver.resume(_TAG_SRC)
            assert states(_TAG_SRC) == [True] * src_chunks
            assert all(holds(src, index, src_bytes(index)) for index in src_backed)
            torch_memory_saver.resume(_TAG_DST)
        assert states(_TAG_SRC) == [True] * src_chunks and states(_TAG_DST) == dst_states
        assert abs(free_bytes() - free_active) < slack
        # The destination's backup went into the source's new memory, and the destination
        # shows what the source's backup restored.
        assert shared_bytes_at_both_addresses()
        assert all(holds(src, index, src_bytes(index)) for index in src_backed)
        assert all(holds(dst, index, dst_bytes(index)) for index in _DST_BACKED)
        # A retained backup is still served after the resume; the others are freed.
        assert_backup(src, src_backed if retained else (), src_bytes)
        assert_backup(dst, _DST_BACKED if retained else (), dst_bytes)

    def slot():
        return torch_memory_saver.get_cpu_backup(part(src, 0), zero_copy=True).data_ptr()

    cycle(1, retained=False)
    cycle(2, retained=False, together=True)
    cycle(3, retained=True, together=True)
    retained_slot = slot()
    # The next pause writes the same chunks into the slot that was retained.
    cycle(4, retained=True, device_full=True)
    assert slot() == retained_slot
    torch_memory_saver.retain_cpu_backup = False
    assert int(normal_tensor.min()) == 100 == int(normal_tensor.max())

    # Return the regions' memory before the interpreter exits: the tensors, then their pools.
    del src, dst
    torch_memory_saver._impl._mem_pools.clear()
    gc.collect()
    empty_cache()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
