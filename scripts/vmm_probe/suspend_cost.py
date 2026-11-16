"""What do a suspend and a resume of a chunked region cost in driver calls, and do the
processes of a host pay for them one at a time?

A region is built the way the inference engine's is: pieces of device memory, each mapped at
an address of its own, and a share of them mapped at a second address as well. A suspend
unmaps every second address, then unmaps and releases every piece; a resume creates every
piece again and maps it, then maps the second addresses, with the access set after each map.
These are the driver calls of pause() and resume() in csrc/core.cpp for a chunked allocation
with map_chunks mappings, in their order. Nothing is written to the memory, read from it or
copied to the host.

Every piece size covers the same --line-gib, so a larger piece means fewer calls. With
--processes N there is one process per GPU, and each suspend and each resume starts at the
same moment in all of them: if the driver serves one process at a time, the times per mapping
are N times those of --processes 1. "on CPU %" is the share of a cycle a process spent
running; a process that waits for another one inside the driver sleeps, unless the driver spins.

The library sets the GPUDirect RDMA flag on every handle where the device supports it; vmm.py
and driver_call_timing.py leave it clear. --rdma-flag clear times the same calls on handles
without it, so the flag's share of the cost can be told from the processes' share.

Usage: python suspend_cost.py [--processes 1] [--piece-mib 4 16 64] [--line-gib 168]
           [--second-fraction 0.70] [--cycles 3] [--rdma-flag library|clear]
           [--out suspend_cost_1p.json]
Exits 1 on a driver error, with the failing call named, and 2 on a refused configuration.
"""

import argparse
import ctypes
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import traceback

import torch

from vmm import Driver, GiB, MiB

MAPPING_LIMIT = 1 << 19  # about what the driver allows a host, all processes together
FIT_MARGIN = GiB  # device memory left free beside the line
START_LEAD = 1.0  # seconds from sending a start time to the start
MESSAGE_TIMEOUT = 1800.0  # seconds the parent waits for the next message of a process
MESSAGE_TAG = "@suspend_cost "
RDMA_WITH_VMM_SUPPORTED = 110  # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED

COLUMNS = (
    f"{'piece MiB':>9s} {'handles':>8s} {'mappings':>9s} {'process':>7s} {'cycle':>6s} {'suspend s':>10s} "
    f"{'resume s':>9s} {'suspend us/map':>15s} {'resume us/map':>14s} {'pair us/map':>12s} {'on CPU %':>9s}"
)


class Refused(Exception):
    """The configuration cannot be measured. Nothing was timed."""


class Failed(Exception):
    """A process of the measurement failed."""


class LibraryDriver(Driver):
    """Creates a handle the way the library's cu_mem_create does (csrc/utils.h).

    Before every cuMemCreate the library asks the device whether it supports the GPUDirect
    RDMA flag, and sets the flag if it does. vmm.py leaves the flag clear, and so does this
    class with `library_flag` False: it then creates a handle as vmm.py does, without the
    question.
    """

    def __init__(self, device=0, library_flag=True):
        super().__init__(device)
        self.library_flag = library_flag
        c_int = ctypes.c_int
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(c_int), c_int]
        self.lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(c_int), c_int, c_int]
        self.lib.cuDeviceGet.restype = self.lib.cuDeviceGetAttribute.restype = c_int
        handle = c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(handle), device), "cuDeviceGet")
        self._device_handle = handle.value
        self._rdma = c_int()

    def rdma_capable(self):
        self._check(
            self.lib.cuDeviceGetAttribute(ctypes.byref(self._rdma), RDMA_WITH_VMM_SUPPORTED, self._device_handle),
            "cuDeviceGetAttribute",
        )
        return bool(self._rdma.value)

    def create(self, size):
        if self.library_flag:
            self._prop.allocFlags.gpuDirectRDMACapable = 1 if self.rdma_capable() else 0
        return super().create(size)


class Region:
    """`handles` pieces at one address range, the first `second` of them at a second range too."""

    def __init__(self, drv, piece, handles, second):
        self.drv, self.piece, self.count, self.second = drv, piece, handles, second
        self.own = drv.reserve(handles * piece)
        self.other = drv.reserve(second * piece) if second else 0
        self.handles = [0] * handles

    def resume(self):
        """resume(): create_all, then map_all, which maps the own pieces before the second addresses."""
        drv, piece, handles, own, other = self.drv, self.piece, self.handles, self.own, self.other
        cpu, t0 = time.thread_time_ns(), time.perf_counter_ns()
        for i in range(self.count):
            handles[i] = drv.create(piece)
        t1 = time.perf_counter_ns()
        for i in range(self.count):
            drv.map(own + i * piece, piece, handles[i])
            drv.set_access(own + i * piece, piece)
        t2 = time.perf_counter_ns()
        for i in range(self.second):
            drv.map(other + i * piece, piece, handles[i])
            drv.set_access(other + i * piece, piece)
        t3 = time.perf_counter_ns()
        return {
            "seconds": (t3 - t0) / 1e9,
            "cpu_seconds": (time.thread_time_ns() - cpu) / 1e9,
            "create_seconds": (t1 - t0) / 1e9,
            "map_own_seconds": (t2 - t1) / 1e9,
            "map_second_seconds": (t3 - t2) / 1e9,
        }

    def suspend(self):
        """pause(): unmap_borrowed_chunks, then release_own_chunks, which unmaps and releases piece by piece."""
        drv, piece, handles, own, other = self.drv, self.piece, self.handles, self.own, self.other
        cpu, t0 = time.thread_time_ns(), time.perf_counter_ns()
        for i in range(self.second):
            drv.unmap(other + i * piece, piece)
        t1 = time.perf_counter_ns()
        for i in range(self.count):
            drv.unmap(own + i * piece, piece)
            drv.release(handles[i])
        t2 = time.perf_counter_ns()
        return {
            "seconds": (t2 - t0) / 1e9,
            "cpu_seconds": (time.thread_time_ns() - cpu) / 1e9,
            "unmap_second_seconds": (t1 - t0) / 1e9,
            "unmap_release_own_seconds": (t2 - t1) / 1e9,
        }

    def free(self):
        self.drv.address_free(self.own, self.count * self.piece)
        if self.second:
            self.drv.address_free(self.other, self.second * self.piece)


def emit(event, **fields):
    print(MESSAGE_TAG + json.dumps({"event": event, **fields}), flush=True)


def used_bytes():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    return total - free


def wait_for_start():
    """Wait until the wall-clock time the parent sends on stdin. Returns how late the start is, in ms."""
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("the parent closed the start pipe")
    start = float(line)
    while (left := start - time.time()) > 0:
        if left > 0.002:
            time.sleep(left - 0.002)
    return (time.time() - start) * 1e3


def child(args, cases):
    """One process of the measurement, on the only device it sees.

    It reports to the parent in tagged JSON lines on stdout and takes each start time from stdin.
    """
    drv = LibraryDriver(0, library_flag=args.rdma_flag == "library")
    granularity = drv.granularity()
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.mem_get_info(0)[1]
    used_at_start = used_bytes()
    free = total - used_at_start
    uneven = [case["piece_mib"] for case in cases if case["piece_mib"] * MiB % granularity]
    line = max(case["bytes"] for case in cases)
    if uneven:
        emit("refuse", message=f"pieces of {uneven} MiB are not multiples of the granularity, {granularity} bytes")
        return 2
    if line + FIT_MARGIN > free:
        room = max(0, free - FIT_MARGIN) // GiB
        emit(
            "refuse",
            message=f"a line of {line / GiB:.1f} GiB does not fit {name}, which has {free / GiB:.1f} GiB free: "
            f"with the margin of {FIT_MARGIN / GiB:.0f} GiB there is room for --line-gib {room}",
        )
        return 2
    emit(
        "hello",
        gpu=name,
        granularity=granularity,
        total_bytes=total,
        used_at_start=used_at_start,
        gpu_direct_rdma_flag=drv.library_flag and drv.rdma_capable(),
    )
    for case in cases:
        region = Region(drv, case["piece_mib"] * MiB, case["handles"], case["second"])
        emit("built", **region.resume(), used=used_bytes())
        for _ in range(args.cycles):
            for phase in (region.suspend, region.resume):
                emit("ready")
                late = wait_for_start()
                emit("done", **phase(), start_delay_ms=late)
        teardown = region.suspend()
        region.free()
        emit("torn_down", **teardown, used=used_bytes())
    emit("bye", used_at_exit=used_bytes())
    return 0


def read_messages(index, proc, inbox):
    for line in proc.stdout:
        if line.startswith(MESSAGE_TAG):
            inbox.put(json.loads(line[len(MESSAGE_TAG):]))
        else:
            sys.stderr.write(f"[process {index}] {line}")
    inbox.put({"event": "eof"})


def start_children(args, gpus):
    children = []
    for index in range(args.processes):
        proc = subprocess.Popen(
            [
                sys.executable, os.path.abspath(__file__), "--child",
                "--piece-mib", *(str(mib) for mib in args.piece_mib),
                "--line-gib", repr(args.line_gib),
                "--second-fraction", repr(args.second_fraction),
                "--cycles", str(args.cycles),
                "--rdma-flag", args.rdma_flag,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[index]},
        )
        inbox = queue.Queue()
        threading.Thread(target=read_messages, args=(index, proc, inbox), daemon=True).start()
        children.append((proc, inbox))
    return children


def receive(children, index, event):
    """The next message of one process, which has to be `event`."""
    proc, inbox = children[index]
    try:
        message = inbox.get(timeout=MESSAGE_TIMEOUT)
    except queue.Empty:
        raise Failed(f"process {index} sent no '{event}' in {MESSAGE_TIMEOUT:.0f} s") from None
    got = message.pop("event")
    if got == "refuse":
        raise Refused(f"process {index}: {message['message']}")
    if got == "error":
        raise Failed(f"process {index}: {message['message']}")
    if got == "eof":
        raise Failed(f"process {index} ended with exit code {proc.wait()} before its '{event}'")
    if got != event:
        raise Failed(f"process {index} sent '{got}' where '{event}' was due")
    return message


def together(children):
    """One suspend or one resume, started at the same moment in every process."""
    for index in range(len(children)):
        receive(children, index, "ready")
    start = time.time() + START_LEAD
    for proc, _ in children:
        try:
            proc.stdin.write(f"{start!r}\n")
            proc.stdin.flush()
        except BrokenPipeError:
            pass  # the process is gone; receive() below reports how it ended
    return [receive(children, index, "done") for index in range(len(children))]


def per_mapping(suspend, resume, mappings):
    return {
        "suspend_seconds": suspend,
        "resume_seconds": resume,
        "suspend_us_per_mapping": suspend * 1e6 / mappings,
        "resume_us_per_mapping": resume * 1e6 / mappings,
        "pair_us_per_mapping": (suspend + resume) * 1e6 / mappings,
    }


def row(case, process, label, times):
    return (
        f"{case['piece_mib']:9d} {case['handles']:8d} {case['mappings']:9d} {process:7d} {label:>6s} "
        f"{times['suspend_seconds']:10.3f} {times['resume_seconds']:9.3f} {times['suspend_us_per_mapping']:15.1f} "
        f"{times['resume_us_per_mapping']:14.1f} {times['pair_us_per_mapping']:12.1f} {100 * times['cpu_share']:9.0f}"
    )


def measure(children, gpus, args, cases, result):
    count = len(children)
    for index in range(count):
        hello = receive(children, index, "hello")
        result["processes"].append({"process": index, "cuda_visible_devices": gpus[index], **hello})
        print(
            f"process {index}: GPU {gpus[index]}, {hello['gpu']}, granularity {hello['granularity']} bytes, "
            f"{hello['used_at_start'] / GiB:.2f} of {hello['total_bytes'] / GiB:.1f} GiB in use, "
            f"GPUDirect RDMA flag on the handles: {'yes' if hello['gpu_direct_rdma_flag'] else 'no'}",
            flush=True,
        )

    for case in cases:
        record = {**case, "processes": [{"process": index, "cycles": []} for index in range(count)]}
        result["cases"].append(record)
        for index in range(count):
            record["processes"][index]["build"] = receive(children, index, "built")
        print(f"\n{COLUMNS}", flush=True)
        for cycle in range(args.cycles):
            suspends = together(children)
            resumes = together(children)
            for index in range(count):
                suspend, resume = suspends[index], resumes[index]
                times = per_mapping(suspend["seconds"], resume["seconds"], case["mappings"])
                times["cpu_share"] = (suspend["cpu_seconds"] + resume["cpu_seconds"]) / (
                    suspend["seconds"] + resume["seconds"]
                )
                record["processes"][index]["cycles"].append({**times, "suspend": suspend, "resume": resume})
                print(row(case, index, str(cycle + 1), times), flush=True)
        for index in range(count):
            cycles = record["processes"][index]["cycles"]
            median = per_mapping(
                statistics.median(cycle["suspend_seconds"] for cycle in cycles),
                statistics.median(cycle["resume_seconds"] for cycle in cycles),
                case["mappings"],
            )
            median["cpu_share"] = statistics.median(cycle["cpu_share"] for cycle in cycles)
            record["processes"][index]["median"] = median
            print(row(case, index, "median", median), flush=True)
        for index in range(count):
            record["processes"][index]["teardown"] = receive(children, index, "torn_down")
        late = max(
            cycle[phase]["start_delay_ms"]
            for process in record["processes"]
            for cycle in process["cycles"]
            for phase in ("suspend", "resume")
        )
        print(f"the starts of these suspends and resumes were at most {late:.2f} ms late", flush=True)

    print()
    for index in range(count):
        process = result["processes"][index]
        process.update(receive(children, index, "bye"))
        kept = process["used_at_exit"] - process["used_at_start"]
        print(
            f"process {index}: device memory in use at start {process['used_at_start'] / MiB:.1f} MiB, "
            f"at exit {process['used_at_exit'] / MiB:.1f} MiB: "
            f"{'all of it returned' if kept == 0 else f'{kept / MiB:+.1f} MiB NOT RETURNED'}",
            flush=True,
        )
    for index, (proc, _) in enumerate(children):
        try:
            code = proc.wait(timeout=MESSAGE_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise Failed(f"process {index} did not exit after its last message") from None
        if code:
            raise Failed(f"process {index} exited with code {code} after its last message")


def driver_version():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.split("\n")[0].strip() or None


def write_json(path, result):
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {path}", flush=True)


def parent(args, cases):
    budget = MAPPING_LIMIT * 3 // 4
    for case in cases:
        if args.processes * case["mappings"] > budget:
            raise Refused(
                f"{args.processes} processes x {case['mappings']} mappings at {case['piece_mib']} MiB is more than "
                f"{budget}, three quarters of the {MAPPING_LIMIT} mappings the driver allows a host"
            )
    cpus = len(os.sched_getaffinity(0))
    if cpus < args.processes:
        raise Refused(f"{args.processes} processes on {cpus} CPUs would time the CPUs, not the driver")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(index) for index in range(args.processes)]
    if len(gpus) < args.processes:
        raise Refused(f"CUDA_VISIBLE_DEVICES={visible} names fewer than {args.processes} GPUs")

    result = {
        "arguments": {name: value for name, value in vars(args).items() if name != "child"},
        "driver_version": driver_version(),
        "cpus": cpus,
        "mapping_limit": MAPPING_LIMIT,
        "fit_margin_bytes": FIT_MARGIN,
        "start_lead_seconds": START_LEAD,
        "processes": [],
        "cases": [],
        "error": None,
    }
    print(
        f"processes: {args.processes}, line: {args.line_gib:g} GiB, second address for {args.second_fraction:g} "
        f"of the pieces, cycles: {args.cycles}, handles: {args.rdma_flag}, driver: {result['driver_version']}, "
        f"CPUs: {cpus}",
        flush=True,
    )
    children = start_children(args, gpus)
    try:
        measure(children, gpus, args, cases, result)
    except Failed as failure:
        result["error"] = str(failure)
        write_json(args.out, result)
        raise
    finally:
        for proc, _ in children:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    write_json(args.out, result)


def mib_list(text):
    return [int(part) for part in text.split(",") if part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=1, help="process i runs on GPU i")
    ap.add_argument("--piece-mib", type=mib_list, nargs="+", default=[[4, 16, 64]], help="4 16 64 or 4,16,64")
    ap.add_argument("--line-gib", type=float, default=168.0, help="device memory every piece size covers, per GPU")
    ap.add_argument("--second-fraction", type=float, default=0.70, help="share of the pieces with a second address")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument(
        "--rdma-flag", choices=("library", "clear"), default="library",
        help="library: the GPUDirect RDMA flag on every handle where the device supports it; clear: never",
    )
    ap.add_argument("--out", default=None, help="JSON file (default: suspend_cost_<processes>p[_clear].json)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.piece_mib = [mib for group in args.piece_mib for mib in group]
    if args.out is None:
        args.out = f"suspend_cost_{args.processes}p{'_clear' if args.rdma_flag == 'clear' else ''}.json"
    if args.processes < 1 or args.cycles < 1 or args.line_gib <= 0 or min(args.piece_mib, default=0) < 1:
        ap.error("--processes, --cycles, --line-gib and every --piece-mib have to be positive")
    if not 0.0 <= args.second_fraction <= 1.0:
        ap.error("--second-fraction is a share between 0 and 1")

    cases = []
    for mib in args.piece_mib:
        handles = int(args.line_gib * GiB) // (mib * MiB)
        if handles == 0:
            ap.error(f"a piece of {mib} MiB is larger than the line of {args.line_gib:g} GiB")
        second = int(handles * args.second_fraction)
        cases.append(
            {
                "piece_mib": mib,
                "handles": handles,
                "second": second,
                "mappings": handles + second,
                "bytes": handles * mib * MiB,
            }
        )

    if args.child:
        try:
            sys.exit(child(args, cases))
        except Exception as error:  # vmm.Driver names the driver call that failed
            traceback.print_exc()
            emit("error", message=str(error))
            sys.exit(1)
    try:
        parent(args, cases)
    except Refused as refusal:
        print(f"refused: {refusal}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Failed as failure:
        print(f"FAILED: {failure}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
