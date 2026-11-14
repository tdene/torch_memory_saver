"""Does one piece of device memory behave correctly when it is mapped at two addresses?

The CUDA programming guide documents this as "virtual aliasing". This probe checks it on the
real driver, the real PyTorch and inside captured CUDA graphs, which is how it would be used:
a graph writes through one address, something else reads or writes through the other, never
in the same operation.

Usage: python two_address_test.py [--device 0] [--mib 64]
Prints one PASS/FAIL line per check and exits 1 if any check failed.
"""

import argparse
import sys

import torch

from vmm import Driver, MiB

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--mib", type=int, default=64)
    args = ap.parse_args()

    drv = Driver(args.device)
    dev = f"cuda:{args.device}"
    g = drv.granularity()
    n = max(4, args.mib * MiB // g)
    size = n * g
    per = g // 4  # float32 elements per piece
    print(f"device {args.device}: granularity {g} bytes, {n} pieces, {size // MiB} MiB per address range", flush=True)

    a, b = drv.reserve(size), drv.reserve(size)
    handles = [drv.create(g) for _ in range(n)]
    for i, h in enumerate(handles):
        drv.map(a + i * g, g, h)
    drv.set_access(a, size)
    # The second address range holds the same pieces in REVERSE order, so a pass cannot come
    # from two ranges that merely happen to be laid out alike.
    for i, h in enumerate(handles):
        drv.map(b + (n - 1 - i) * g, g, h)
    drv.set_access(b, size)

    ta = drv.tensor(a, size).view(n, per)
    tb = drv.tensor(b, size).view(n, per)
    rev = torch.arange(n - 1, -1, -1, device=dev)

    # 1. eager, write through A, read through B
    pattern = torch.arange(n * per, device=dev, dtype=torch.float32).view(n, per)
    ta.copy_(pattern)
    torch.cuda.synchronize()
    check("eager: written through A, read through B", torch.equal(tb, pattern[rev]))

    # 2. eager, write through B, read through A
    tb.copy_(-pattern)
    torch.cuda.synchronize()
    check("eager: written through B, read through A", torch.equal(ta, -pattern[rev]))

    # 3. a captured graph writes through A; after the replay B shows the result
    x = torch.zeros(n, per, device=dev)
    writer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(writer):
        ta.copy_(x)
        ta.mul_(2.0)
        total = ta.sum(dtype=torch.float64)
    ok = True
    for step in range(3):
        x.copy_(pattern + step)
        writer.replay()
        torch.cuda.synchronize()
        ok &= torch.equal(tb, (2.0 * (pattern + step))[rev])
        ok &= bool(torch.isclose(total, (2.0 * (pattern + step)).sum(dtype=torch.float64)))
    check("graph: replay writes through A, B shows it (3 replays)", ok)

    # 4. one graph writes through A, a second graph reads through B (two operations, in order)
    reader = torch.cuda.CUDAGraph()
    with torch.cuda.graph(reader):
        seen = tb + 1.0
    x.copy_(pattern * 3.0)
    writer.replay()
    reader.replay()
    torch.cuda.synchronize()
    check("graph: writer through A, then reader through B", torch.equal(seen, (6.0 * pattern)[rev] + 1.0))

    # 5. the other user overwrites the memory between two replays; the graph is unaffected,
    #    because it rewrites what it reads (the property the design relies on)
    tb.fill_(float("nan"))
    x.copy_(pattern + 7.0)
    writer.replay()
    torch.cuda.synchronize()
    check(
        "graph: correct after the other address scribbled NaN in between",
        bool(torch.isclose(total, (2.0 * (pattern + 7.0)).sum(dtype=torch.float64))) and not bool(torch.isnan(ta).any()),
    )

    # 6. a third address for the same pieces
    c = drv.reserve(size)
    for i, h in enumerate(handles):
        drv.map(c + i * g, g, h)
    drv.set_access(c, size)
    tc = drv.tensor(c, size).view(n, per)
    torch.cuda.synchronize()
    check("three addresses: C equals A", torch.equal(tc, ta))
    del tc
    for i in range(n):
        drv.unmap(c + i * g, g)
    drv.address_free(c, size)

    # 7. unmapping one address leaves the other intact, and the address can be mapped again
    before = ta.clone()
    del tb, seen, reader
    for i in range(n):
        drv.unmap(b + i * g, g)
    torch.cuda.synchronize()
    check("unmap B: A keeps its contents", torch.equal(ta, before))
    for i, h in enumerate(handles):
        drv.map(b + i * g, g, h)  # same order as A this time
    drv.set_access(b, size)
    tb = drv.tensor(b, size).view(n, per)
    check("remap B in A's order: B equals A", torch.equal(tb, ta))

    # 8. the handles can be released while mappings exist; the memory lives until the last unmap
    free_before = torch.cuda.mem_get_info(args.device)[0]
    for h in handles:
        drv.release(h)
    ta.fill_(5.0)
    torch.cuda.synchronize()
    check("handles released, mappings alive: still usable", bool((tb == 5.0).all()))
    del ta, tb, before, writer, total
    for i in range(n):
        drv.unmap(a + i * g, g)
    held = free_before - torch.cuda.mem_get_info(args.device)[0]
    for i in range(n):
        drv.unmap(b + i * g, g)
    freed = torch.cuda.mem_get_info(args.device)[0] - free_before
    check(
        "memory returns only after the LAST address is unmapped",
        abs(held) < g and freed >= size - g,
        f"after unmapping A only: {held} bytes change; after B too: {freed} bytes freed of {size}",
    )
    drv.address_free(a, size)
    drv.address_free(b, size)

    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}", flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
