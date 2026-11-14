"""What do the driver's memory calls cost, as a function of the piece size?

For each piece size the same total amount of memory is built from pieces of that size at one
address range, mapped a second time at another range, and torn down again. Reported per call
and per GiB. This decides the piece size a chunked region uses.

Usage: python driver_call_timing.py [--device 0] [--gib 4] [--mib 2 8 32 128 1024] [--repeat 3]
"""

import argparse
import json
import time

import torch

from vmm import Driver, GiB, MiB


def timed(fn, count):
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for i in range(count):
        fn(i)
    return (time.perf_counter_ns() - t0) / 1e3  # microseconds


def one_size(drv, total, piece):
    n = total // piece
    size = n * piece
    a, b = drv.reserve(size), drv.reserve(size)
    handles = [0] * n

    def create(i):
        handles[i] = drv.create(piece)

    out = {"piece_mib": piece // MiB, "pieces": n, "gib": size / GiB}
    out["create"] = timed(create, n)
    out["map"] = timed(lambda i: drv.map(a + i * piece, piece, handles[i]), n)
    out["set_access_per_piece"] = timed(lambda i: drv.set_access(a + i * piece, piece), n)
    out["map_second"] = timed(lambda i: drv.map(b + i * piece, piece, handles[i]), n)
    out["set_access_whole_range"] = timed(lambda i: drv.set_access(b, size), 1)
    # touch both ranges once, so the teardown is of memory that was really used
    drv.tensor(a, size, torch.uint8).fill_(1)
    ok = bool((drv.tensor(b, size, torch.uint8) == 1).all())
    torch.cuda.synchronize()
    out["unmap_second"] = timed(lambda i: drv.unmap(b + i * piece, piece), n)
    out["unmap"] = timed(lambda i: drv.unmap(a + i * piece, piece), n)
    out["release"] = timed(lambda i: drv.release(handles[i]), n)
    drv.address_free(a, size)
    drv.address_free(b, size)
    out["second_address_saw_the_data"] = ok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gib", type=float, default=4.0)
    ap.add_argument("--mib", type=int, nargs="+", default=[2, 8, 32, 128, 1024])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    drv = Driver(args.device)
    g = drv.granularity()
    free, cap = torch.cuda.mem_get_info(args.device)
    total = int(args.gib * GiB)
    print(f"device {args.device}: {torch.cuda.get_device_name(args.device)}, granularity {g} bytes, "
          f"free {free / GiB:.1f} of {cap / GiB:.1f} GiB, building {total / GiB:.1f} GiB per size", flush=True)
    assert total < free, "not enough free memory for --gib"

    calls = ["create", "map", "set_access_per_piece", "map_second", "unmap_second", "unmap", "release"]
    rows = []
    for mib in args.mib:
        piece = mib * MiB
        if piece % g or piece > total:
            print(f"skip {mib} MiB (not a multiple of the granularity, or larger than the total)")
            continue
        runs = [one_size(drv, total, piece) for _ in range(args.repeat)]
        best = {k: min(r[k] for r in runs) for k in calls + ["set_access_whole_range"]}
        best.update(piece_mib=mib, pieces=runs[0]["pieces"], gib=runs[0]["gib"],
                    ok=all(r["second_address_saw_the_data"] for r in runs))
        rows.append(best)
        n, gib = best["pieces"], best["gib"]
        print(f"\npiece {mib} MiB x {n} (best of {args.repeat}; second address saw the data: {best['ok']})")
        print(f"  {'call':22s} {'us per call':>12s} {'ms per GiB':>12s}")
        for k in calls:
            print(f"  {k:22s} {best[k] / n:12.1f} {best[k] / 1e3 / gib:12.2f}")
        print(f"  {'set_access, one call':22s} {best['set_access_whole_range']:12.1f} "
              f"{best['set_access_whole_range'] / 1e3 / gib:12.2f}")
        build = best["create"] + best["map"] + best["set_access_whole_range"]
        second = best["map_second"] + best["set_access_whole_range"]
        teardown = best["unmap"] + best["release"]
        print(f"  build (create + map + one set_access): {build / 1e3 / gib:8.2f} ms per GiB")
        print(f"  second address (map + one set_access): {second / 1e3 / gib:8.2f} ms per GiB")
        print(f"  tear down (unmap + release):           {teardown / 1e3 / gib:8.2f} ms per GiB", flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
