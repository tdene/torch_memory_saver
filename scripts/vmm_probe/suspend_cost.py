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

Three further questions, one option each; without them the measurement is as described above.

--in-turn: the processes take each suspend and each resume one after another instead of at the
same moment. If the sum of their times is below what one process takes with all starting
together, waiting for each other costs more than taking turns would.

--keep-fraction F: a suspend leaves that share of the pieces alone, mapped and holding their
memory, the pieces with a second address first; only the others are unmapped, released and
made again. The times per mapping are over the mappings a cycle touches: equal to those of
F = 0 if the mappings that stay cost the others nothing. The device memory in use after a
suspend shows that exactly the released share came back.

--copy-during: a suspend copies the line to pinned host memory beside its calls, as pause()'s
backup does before them, and a slab's pieces go once the slab is copied; a resume makes a
slab's pieces and copies the slab back while it makes the next. One pinned buffer of
SLAB_BYTES is reused for every slab, so the host side is lighter than pause()'s, which pins
the whole line. "suspend s" and "resume s" are then the time until both the copy and the
calls are done: near the larger of the copy alone (timed once after the build) and the calls
alone (a run without the option) if the driver runs both at once, near their sum if not.

Usage: python suspend_cost.py [--processes 1] [--piece-mib 4 16 64] [--line-gib 168]
           [--second-fraction 0.70] [--cycles 3] [--rdma-flag library|clear]
           [--in-turn] [--keep-fraction 0.0 | --copy-during] [--out suspend_cost_1p.json]
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
SLAB_BYTES = GiB  # what one copy of --copy-during moves, and what its calls follow

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


class Frontier:
    """How far along the line one thread has come; the other thread waits for it."""

    def __init__(self):
        self._changed = threading.Condition()
        self._bytes = 0
        self._error = None

    def advance(self, to_bytes):
        with self._changed:
            self._bytes = to_bytes
            self._changed.notify_all()

    def fail(self, error):
        with self._changed:
            self._error = error
            self._changed.notify_all()

    def wait_for(self, to_bytes):
        """Returns the seconds waited. Raises if the other thread failed."""
        t0 = time.perf_counter_ns()
        with self._changed:
            while self._bytes < to_bytes and self._error is None:
                self._changed.wait()
            if self._error is not None:
                raise RuntimeError(f"the other thread of this suspend or resume failed: {self._error}")
        return (time.perf_counter_ns() - t0) / 1e9


class HostCopy:
    """Copies a line to pinned host memory or back, a slab at a time, through one pinned buffer."""

    def __init__(self, drv, slab_bytes):
        self.drv, self.slab_bytes = drv, slab_bytes
        vp, sz, u64 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong
        signatures = {
            "cuCtxGetCurrent": [ctypes.POINTER(vp)],
            "cuCtxSetCurrent": [vp],
            "cuMemAllocHost_v2": [ctypes.POINTER(vp), sz],
            "cuMemFreeHost": [vp],
            "cuMemcpyDtoH_v2": [vp, u64, sz],
            "cuMemcpyHtoD_v2": [u64, vp, sz],
        }
        for name, argtypes in signatures.items():
            fn = getattr(drv.lib, name)
            fn.argtypes, fn.restype = argtypes, ctypes.c_int
        self._context, self._host = vp(), vp()
        drv._check(drv.lib.cuCtxGetCurrent(ctypes.byref(self._context)), "cuCtxGetCurrent")
        drv._check(drv.lib.cuMemAllocHost_v2(ctypes.byref(self._host), slab_bytes), "cuMemAllocHost")

    def free(self):
        self.drv._check(self.drv.lib.cuMemFreeHost(self._host), "cuMemFreeHost")

    def run(self, base, size, to_host, copied=None, mapped=None):
        """Copy [base, base + size) slab by slab on the calling thread, which may be a new one.

        `copied` is advanced past every slab that is done; a slab waits until `mapped` is past it.
        Returns the seconds in the copies and the seconds waited for `mapped`.
        """
        drv, lib, host = self.drv, self.drv.lib, self._host
        drv._check(lib.cuCtxSetCurrent(self._context), "cuCtxSetCurrent")
        copying = waiting = 0.0
        for offset in range(0, size, self.slab_bytes):
            count = min(self.slab_bytes, size - offset)
            if mapped is not None:
                waiting += mapped.wait_for(offset + count)
            t0 = time.perf_counter_ns()
            if to_host:
                drv._check(lib.cuMemcpyDtoH_v2(host, base + offset, count), "cuMemcpyDtoH")
            else:
                drv._check(lib.cuMemcpyHtoD_v2(base + offset, host, count), "cuMemcpyHtoD")
            copying += (time.perf_counter_ns() - t0) / 1e9
            if copied is not None:
                copied.advance(offset + count)
        return copying, waiting


class CopyThread(threading.Thread):
    """HostCopy.run on a thread of its own. `ahead` is the Frontier the copy advances or follows."""

    def __init__(self, copy, base, size, to_host, ahead):
        super().__init__(daemon=True)
        self.copy, self.base, self.size, self.to_host, self.ahead = copy, base, size, to_host, ahead
        self.copying = self.waiting = 0.0
        self.error = None

    def run(self):
        try:
            if self.to_host:
                self.copying, self.waiting = self.copy.run(self.base, self.size, True, copied=self.ahead)
            else:
                self.copying, self.waiting = self.copy.run(self.base, self.size, False, mapped=self.ahead)
        except Exception as error:  # the calls' thread reports it: it waits on `ahead` or joins
            self.error = error
            self.ahead.fail(error)

    def finish(self):
        self.join()
        if self.error is not None:
            raise self.error


class Region:
    """`handles` pieces at one address range, the first `second` of them at a second range too."""

    def __init__(self, drv, piece, handles, second):
        self.drv, self.piece, self.count, self.second = drv, piece, handles, second
        self.own = drv.reserve(handles * piece)
        self.other = drv.reserve(second * piece) if second else 0
        self.handles = [0] * handles

    def resume(self, kept=0):
        """resume(): create_all, then map_all, which maps the own pieces before the second addresses.

        The first `kept` pieces are left as they are: a suspend(kept) did not touch them.
        """
        drv, piece, handles, own, other = self.drv, self.piece, self.handles, self.own, self.other
        cpu, t0 = time.thread_time_ns(), time.perf_counter_ns()
        for i in range(kept, self.count):
            handles[i] = drv.create(piece)
        t1 = time.perf_counter_ns()
        for i in range(kept, self.count):
            drv.map(own + i * piece, piece, handles[i])
            drv.set_access(own + i * piece, piece)
        t2 = time.perf_counter_ns()
        for i in range(min(kept, self.second), self.second):
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

    def suspend(self, kept=0):
        """pause(): unmap_borrowed_chunks, then release_own_chunks, which unmaps and releases piece by piece.

        The first `kept` pieces stay mapped, at both addresses where they have two, and keep their memory.
        """
        drv, piece, handles, own, other = self.drv, self.piece, self.handles, self.own, self.other
        cpu, t0 = time.thread_time_ns(), time.perf_counter_ns()
        for i in range(min(kept, self.second), self.second):
            drv.unmap(other + i * piece, piece)
        t1 = time.perf_counter_ns()
        for i in range(kept, self.count):
            drv.unmap(own + i * piece, piece)
            drv.release(handles[i])
        t2 = time.perf_counter_ns()
        return {
            "seconds": (t2 - t0) / 1e9,
            "cpu_seconds": (time.thread_time_ns() - cpu) / 1e9,
            "unmap_second_seconds": (t1 - t0) / 1e9,
            "unmap_release_own_seconds": (t2 - t1) / 1e9,
        }

    def _slabs(self, slab_bytes):
        """(first piece, piece after the last) of every slab of the line."""
        step = slab_bytes // self.piece
        return [(first, min(first + step, self.count)) for first in range(0, self.count, step)]

    def suspend_beside_copy(self, copy):
        """The calls of suspend(), a slab's once the slab is on the host: its second addresses, then its pieces."""
        drv, piece, handles, own, other = self.drv, self.piece, self.handles, self.own, self.other
        copied = Frontier()
        thread = CopyThread(copy, own, self.count * piece, True, copied)
        waited = 0.0
        cpu, t0 = time.thread_time_ns(), time.perf_counter_ns()
        thread.start()
        for first, end in self._slabs(copy.slab_bytes):
            waited += copied.wait_for(end * piece)
            for i in range(first, min(end, self.second)):
                drv.unmap(other + i * piece, piece)
            for i in range(first, end):
                drv.unmap(own + i * piece, piece)
                drv.release(handles[i])
        t1 = time.perf_counter_ns()
        thread.finish()
        t2 = time.perf_counter_ns()
        return {
            "seconds": (t2 - t0) / 1e9,
            "cpu_seconds": (time.thread_time_ns() - cpu) / 1e9,
            "calls_seconds": (t1 - t0) / 1e9 - waited,
            "calls_waited_seconds": waited,
            "copy_seconds": thread.copying,
        }

    def resume_beside_copy(self, copy):
        """The calls of resume() slab by slab, each slab copied back from the host while the next is made."""
        drv, piece, handles, own, other = self.drv, self.piece, self.handles, self.own, self.other
        mapped = Frontier()
        thread = CopyThread(copy, own, self.count * piece, False, mapped)
        cpu, t0 = time.thread_time_ns(), time.perf_counter_ns()
        thread.start()
        try:
            for first, end in self._slabs(copy.slab_bytes):
                for i in range(first, end):
                    handles[i] = drv.create(piece)
                for i in range(first, end):
                    drv.map(own + i * piece, piece, handles[i])
                    drv.set_access(own + i * piece, piece)
                mapped.advance(end * piece)
                for i in range(first, min(end, self.second)):
                    drv.map(other + i * piece, piece, handles[i])
                    drv.set_access(other + i * piece, piece)
        except Exception as error:
            mapped.fail(error)  # the copy waits for pieces that will not come
            raise
        t1 = time.perf_counter_ns()
        thread.finish()
        t2 = time.perf_counter_ns()
        return {
            "seconds": (t2 - t0) / 1e9,
            "cpu_seconds": (time.thread_time_ns() - cpu) / 1e9,
            "calls_seconds": (t1 - t0) / 1e9,
            "copy_seconds": thread.copying,
            "copy_waited_seconds": thread.waiting,
        }

    def copy_alone(self, copy):
        """The line copied to the host and back with no call beside it. Every piece is mapped."""
        size = self.count * self.piece
        return {
            "to_host_seconds": copy.run(self.own, size, True)[0],
            "to_device_seconds": copy.run(self.own, size, False)[0],
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
        built = region.resume()
        if args.copy_during:
            copy = HostCopy(drv, case["slab_bytes"])
            built["copy_alone"] = region.copy_alone(copy)
            phases = (lambda: region.suspend_beside_copy(copy), lambda: region.resume_beside_copy(copy))
        elif case["kept"]:
            phases = (lambda: region.suspend(case["kept"]), lambda: region.resume(case["kept"]))
        else:
            phases = (region.suspend, region.resume)
        emit("built", **built, used=used_bytes())
        for _ in range(args.cycles):
            for phase in phases:
                emit("ready")
                late = wait_for_start()
                times = phase()
                if case["kept"]:
                    times["used"] = used_bytes()
                emit("done", **times, start_delay_ms=late)
        teardown = region.suspend()
        if args.copy_during:
            copy.free()
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
                "--keep-fraction", repr(args.keep_fraction),
                *(["--copy-during"] if args.copy_during else []),
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


def send_start(proc):
    try:
        proc.stdin.write(f"{time.time() + START_LEAD!r}\n")
        proc.stdin.flush()
    except BrokenPipeError:
        pass  # the process is gone; receive() reports how it ended


def together(children, in_turn=False):
    """One suspend or one resume in every process: started at the same moment, or, `in_turn`, each
    one after the one before it is done."""
    for index in range(len(children)):
        receive(children, index, "ready")
    if in_turn:
        done = []
        for index, (proc, _) in enumerate(children):
            send_start(proc)
            done.append(receive(children, index, "done"))
        return done
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
        if case["kept"]:
            print(
                f"\na suspend leaves {case['kept']} of the {case['handles']} pieces of {case['piece_mib']} MiB "
                f"alone; a cycle touches {case['cycled_mappings']} of the {case['mappings']} mappings, and the "
                f"times per mapping are over those",
                flush=True,
            )
        print(f"\n{COLUMNS}", flush=True)
        for cycle in range(args.cycles):
            suspends = together(children, args.in_turn)
            resumes = together(children, args.in_turn)
            for index in range(count):
                suspend, resume = suspends[index], resumes[index]
                times = per_mapping(suspend["seconds"], resume["seconds"], case["cycled_mappings"])
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
                case["cycled_mappings"],
            )
            median["cpu_share"] = statistics.median(cycle["cpu_share"] for cycle in cycles)
            record["processes"][index]["median"] = median
            print(row(case, index, "median", median), flush=True)
        medians = [process["median"] for process in record["processes"]]
        if args.in_turn:
            print(
                f"one after another, all {count} processes: "
                f"{sum(median['suspend_seconds'] for median in medians):.3f} s of suspends and "
                f"{sum(median['resume_seconds'] for median in medians):.3f} s of resumes (sums of the medians)",
                flush=True,
            )
        for index in range(count):
            process = record["processes"][index]
            cycles = process["cycles"]
            if case["kept"]:
                built = process["build"]["used"]
                after = statistics.median(cycle["suspend"]["used"] for cycle in cycles)
                print(
                    f"process {index}: device memory in use {built / GiB:.2f} GiB built, {after / GiB:.2f} GiB "
                    f"after a suspend: {(built - after) / GiB:.2f} GiB came back, "
                    f"{(case['handles'] - case['kept']) * case['piece_mib'] / 1024:.2f} GiB were released",
                    flush=True,
                )
            if args.copy_during:
                alone = process["build"]["copy_alone"]

                def middle(phase, key):
                    return statistics.median(cycle[phase][key] for cycle in cycles)

                print(
                    f"process {index}: the copy alone takes {alone['to_host_seconds']:.3f} s to the host and "
                    f"{alone['to_device_seconds']:.3f} s back. Beside the calls: suspend "
                    f"{middle('suspend', 'seconds'):.3f} s (copies {middle('suspend', 'copy_seconds'):.3f} s, "
                    f"calls {middle('suspend', 'calls_seconds'):.3f} s, which waited "
                    f"{middle('suspend', 'calls_waited_seconds'):.3f} s for the copy); resume "
                    f"{middle('resume', 'seconds'):.3f} s (calls {middle('resume', 'calls_seconds'):.3f} s, "
                    f"copies {middle('resume', 'copy_seconds'):.3f} s, which waited "
                    f"{middle('resume', 'copy_waited_seconds'):.3f} s for the calls)",
                    flush=True,
                )
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
    modes = [
        "the processes take turns" if args.in_turn else "",
        f"a suspend leaves {args.keep_fraction:g} of the pieces alone" if args.keep_fraction else "",
        f"the line is copied to the host and back beside the calls, {SLAB_BYTES // MiB} MiB at a time"
        if args.copy_during
        else "",
    ]
    if any(modes):
        print("; ".join(mode for mode in modes if mode), flush=True)
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
    ap.add_argument("--in-turn", action="store_true", help="one process after another, not all at the same moment")
    ap.add_argument(
        "--keep-fraction", type=float, default=0.0,
        help="share of the pieces a suspend leaves mapped and holding memory, those with a second address first",
    )
    ap.add_argument(
        "--copy-during", action="store_true",
        help="copy the line to pinned host memory beside a suspend's calls, and back beside a resume's",
    )
    ap.add_argument(
        "--out", default=None,
        help="JSON file (default: suspend_cost_<processes>p[_clear][_turn][_keep<percent>][_copy].json)",
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.piece_mib = [mib for group in args.piece_mib for mib in group]
    if args.out is None:
        args.out = (
            f"suspend_cost_{args.processes}p{'_clear' if args.rdma_flag == 'clear' else ''}"
            f"{'_turn' if args.in_turn else ''}"
            f"{f'_keep{round(100 * args.keep_fraction)}' if args.keep_fraction else ''}"
            f"{'_copy' if args.copy_during else ''}.json"
        )
    if args.processes < 1 or args.cycles < 1 or args.line_gib <= 0 or min(args.piece_mib, default=0) < 1:
        ap.error("--processes, --cycles, --line-gib and every --piece-mib have to be positive")
    if not 0.0 <= args.second_fraction <= 1.0:
        ap.error("--second-fraction is a share between 0 and 1")
    if not 0.0 <= args.keep_fraction < 1.0:
        ap.error("--keep-fraction is a share from 0 to less than 1: a suspend has to release something")
    if args.keep_fraction and args.copy_during:
        ap.error("--keep-fraction and --copy-during are two measurements: one of them a run")

    cases = []
    for mib in args.piece_mib:
        handles = int(args.line_gib * GiB) // (mib * MiB)
        if handles == 0:
            ap.error(f"a piece of {mib} MiB is larger than the line of {args.line_gib:g} GiB")
        second = int(handles * args.second_fraction)
        kept = int(handles * args.keep_fraction)
        cases.append(
            {
                "piece_mib": mib,
                "handles": handles,
                "second": second,
                "mappings": handles + second,
                "bytes": handles * mib * MiB,
                "kept": kept,
                "cycled_mappings": handles - kept + max(0, second - kept),
                "slab_bytes": max(1, SLAB_BYTES // (mib * MiB)) * mib * MiB,
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
