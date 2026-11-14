"""What does the device report as free at each step of a chunked region's life?

The chunked and map_chunks examples compare `mem_get_info` readings against a baseline
taken before the first device operation of the process. This probe prints the reading
after every step, twice: once from a cold process (the examples' situation) and once
again in the same process, where everything the first pass loaded is already loaded.
A constant offset in the first pass that is absent from the second is memory the
process took for itself (kernels, handles), not memory of the region.

Usage: python chunk_memory_readings.py [torch|preload]
Run from the repository root of a built checkout.
"""

import gc
import sys

import torch

from torch_memory_saver import torch_memory_saver

MiB = 1 << 20


def free_bytes():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info(0)[0]


def line(name, baseline, expected):
    used = baseline - free_bytes()
    print(
        f"  {name:44s} used {used / MiB:9.1f} MiB   expected {expected / MiB:9.1f}   "
        f"off by {(used - expected) / MiB:+8.1f}   torch reserved {torch.cuda.memory_reserved(0) / MiB:8.1f}",
        flush=True,
    )


def drop(tag, *tensors):
    impl = torch_memory_saver._impl
    for key in [key for key in impl._mem_pools if key[0] == tag]:
        del impl._mem_pools[key]
    gc.collect()
    torch.cuda.empty_cache()


def one_pass(label, chunk):
    print(f"{label}: chunk {chunk // MiB} MiB", flush=True)
    full_tag, part_tag, src_tag, dst_tag = (f"{label}_{name}" for name in ("full", "part", "src", "dst"))

    # The chunked example, part 1: backed at once, paused, resumed.
    baseline = free_bytes()
    line("baseline", baseline, 0)
    with torch_memory_saver.region(tag=full_tag, enable_cpu_backup=True, chunk_size=chunk):
        full = torch.empty((5 * chunk,), dtype=torch.uint8, device="cuda")
    line("region of 5 chunks, nothing written", baseline, 5 * chunk)
    full.fill_(7)
    line("after the first fill_", baseline, 5 * chunk)
    torch_memory_saver.pause(full_tag)
    line("after pause", baseline, 0)
    torch_memory_saver.resume(full_tag)
    line("after resume", baseline, 5 * chunk)
    assert int(full.min()) == 7 == int(full.max())
    line("after min/max", baseline, 5 * chunk)
    torch_memory_saver.pause(full_tag)
    line("after the second pause", baseline, 0)
    torch_memory_saver.resume(full_tag)

    # Part 2: address space only, then three chunks.
    with torch_memory_saver.region(
        tag=part_tag, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        part = torch.empty((8 * chunk,), dtype=torch.uint8, device="cuda")
    line("plus 8 chunks of address space", baseline, 5 * chunk)
    torch_memory_saver.resume_chunks(part_tag, [0, 1, 5])
    line("plus 3 chunks backed", baseline, 8 * chunk)
    part[: 2 * chunk].fill_(3)
    line("after a fill_ of two of them", baseline, 8 * chunk)

    # The map_chunks example: 4 source chunks, 2 own destination chunks, 2 mapped.
    with torch_memory_saver.region(tag=src_tag, enable_cpu_backup=True, chunk_size=chunk):
        src = torch.empty((4 * chunk,), dtype=torch.uint8, device="cuda")
    src.fill_(1)
    with torch_memory_saver.region(
        tag=dst_tag, enable_cpu_backup=True, chunk_size=chunk, initially_released=True
    ):
        dst = torch.empty((6 * chunk,), dtype=torch.uint8, device="cuda")
    torch_memory_saver.resume_chunks(dst_tag, [0, 1])
    dst[: 2 * chunk].fill_(2)
    line("plus source 4 + destination 2", baseline, 14 * chunk)
    torch_memory_saver.map_chunks(dst_tag, [4, 5], src_tag, [3, 2])
    line("after map_chunks of 2 (no new memory)", baseline, 14 * chunk)
    seen = int(dst[4 * chunk : 5 * chunk].min())
    line(f"after reading through the mapping (saw {seen})", baseline, 14 * chunk)
    torch_memory_saver.pause(dst_tag)
    line("after pause of the destination", baseline, 12 * chunk)
    torch_memory_saver.pause(src_tag)
    line("after pause of the source", baseline, 8 * chunk)
    torch_memory_saver.resume(src_tag)
    torch_memory_saver.resume(dst_tag)
    line("after resume of both", baseline, 14 * chunk)

    del full, part, src, dst
    for tag in (dst_tag, src_tag, part_tag, full_tag):
        drop(tag)
    line("everything freed", baseline, 0)
    return baseline - free_bytes()


def main():
    torch_memory_saver.hook_mode = sys.argv[1] if len(sys.argv) > 1 else "torch"
    granularity = torch_memory_saver.get_allocation_granularity()
    chunk = 32 * granularity
    print(
        f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, hook mode "
        f"{torch_memory_saver.hook_mode}, granularity {granularity} bytes",
        flush=True,
    )
    cold = one_pass("cold", chunk)
    warm = one_pass("warm", chunk)
    print(f"left behind by the cold pass: {cold / MiB:.1f} MiB; by the warm pass: {warm / MiB:.1f} MiB")


if __name__ == "__main__":
    main()
