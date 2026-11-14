import atexit
import ctypes
import operator

import numpy as np
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Literal, Optional, Sequence
import torch

from .binary_wrapper import BinaryWrapper
from .hooks.base import HookUtilBase, HookMode
from .utils import is_musa, is_xpu

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"
CpuBackupBackend = Literal["mmap", "pinned"]
# torch.OutOfMemoryError, which older PyTorch has only as torch.cuda.OutOfMemoryError.
_OutOfMemoryError = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
# Up to PyTorch 2.7 a segment that a tensor still occupies when its pool is dropped is later
# returned with cudaFree, not through the pool's allocator, which torch-mode memory needs.
_MIN_TORCH_FOR_POOL_RELEASE = (2, 8)


# TmsChunkStatus in csrc/core.h: why a chunk call was refused.
_CHUNK_ERRORS = {
    2: (_OutOfMemoryError, "the device has no memory for the chunks"),
    -1: (IndexError, "a chunk index is out of range for the tag"),
    -2: (RuntimeError, "an allocation of the tag is paused"),
}


def _raise_for_chunk_status(status: int, call: str) -> None:
    if status != 0:
        error, reason = _CHUNK_ERRORS[status]
        raise error(f"torch_memory_saver.{call}: {reason}; nothing was changed")


def _is_cuda_build() -> bool:
    return not (torch.version.hip or is_xpu() or is_musa())


def _torch_version() -> tuple:
    return tuple(int(part) for part in torch.__version__.split(".")[:2])


class TorchMemorySaver:
    def __init__(self):
        self._impl_ctor_kwargs = {}
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(
        self,
        tag: str = _TAG_DEFAULT,
        enable_cpu_backup: bool = False,
        enable_disk_backup: bool = False,
        cpu_backup_backend: Optional[CpuBackupBackend] = None,
        chunk_size: int = 0,
        initially_released: bool = False,
    ):
        """Context manager for memory saving with optional tag.

        enable_disk_backup spills paused memory to files instead of a CPU
        buffer; mutually exclusive with enable_cpu_backup. The target directory is
        process-global (TMS_DISK_BACKUP_DIR or set_disk_backup_dir()), not
        region-scoped, and must be a real disk mount (not tmpfs).

        cpu_backup_backend selects the host shadow when enable_cpu_backup is True:
        "pinned" (default; cudaMallocHost/hipHostMalloc/musaMallocHost) or "mmap"
        (reclaimable RSS on CUDA/MUSA). ROCm is pinned-only. XPU uses pageable
        malloc and does not take cpu_backup_backend.

        chunk_size (CUDA only): each allocation of the region is one address range
        backed by one device handle per chunk_size bytes; the last chunk is shorter
        when the size is no multiple. It must be a multiple of
        get_allocation_granularity() and applies to every allocation made in the
        region, PyTorch's pool segments included. pause(tag) and resume(tag) treat
        such an allocation as a whole and pass chunks without memory by;
        get_chunk_layout(), get_chunk_states() and resume_chunks() address its
        chunks by tag. No disk backup.

        initially_released (CUDA only): the region's allocations reserve address
        space only. Each is born paused, with no device memory and no backup, and
        resume(tag) gives it memory. With chunk_size it is born active with no
        chunk holding memory, and resume_chunks() gives memory to the chunks that
        are used. Create tensors in such a region with torch.empty: nothing may
        write to them before they have memory.
        """
        self._validate_cpu_backup_backend(enable_cpu_backup, cpu_backup_backend)
        self._ensure_initialized()
        assert not (enable_cpu_backup and enable_disk_backup), \
            "enable_cpu_backup and enable_disk_backup are mutually exclusive"
        with self._impl.region(
            tag=tag,
            enable_cpu_backup=enable_cpu_backup,
            enable_disk_backup=enable_disk_backup,
            cpu_backup_backend=cpu_backup_backend,
            chunk_size=chunk_size,
            initially_released=initially_released,
        ):
            yield

    @contextmanager
    def cuda_graph(
            self,
            cuda_graph, pool=None, stream=None, capture_error_mode='global',
            tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False,
            cpu_backup_backend: Optional[CpuBackupBackend] = None,
    ):
        """Similar to `torch.cuda.graph`, but ensures memory in it to be pauseable."""
        self._validate_cpu_backup_backend(enable_cpu_backup, cpu_backup_backend)
        self._ensure_initialized()
        with self._impl.cuda_graph(
                cuda_graph=cuda_graph,
                pool=pool, stream=stream, capture_error_mode=capture_error_mode,
                tag=tag, enable_cpu_backup=enable_cpu_backup,
                cpu_backup_backend=cpu_backup_backend,
        ):
            yield

    @contextmanager
    def disable(self):
        self._ensure_initialized()
        with self._impl.disable():
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        self._ensure_initialized()
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None

        On CUDA, raises torch.OutOfMemoryError when the device has no memory for
        the paused allocations: all of them stay paused with their backups, and
        the call can be repeated once device memory is free.
        """
        self._ensure_initialized()
        self._impl.resume(tag=tag)

    def get_allocation_granularity(self, device: Optional[int] = None) -> int:
        """Granularity of device memory in bytes: the size of an allocation on the
        device (None: the current device) is rounded up to a multiple of it.

        CUDA only: NotImplementedError on other backends. Raises ValueError for a
        device outside [0, device_count()).
        """
        self._ensure_initialized()
        return self._impl.get_allocation_granularity(device)

    def get_chunk_layout(self, tag: str) -> list[tuple[int, int]]:
        """(address, size) of every chunk of the tag's chunked allocations.

        A tag names one chunked allocation, or the segments of one pool: its chunks
        are numbered across its allocations in the order those were made, and
        these numbers are the chunk indices of the other chunk calls. CUDA only.
        """
        self._ensure_initialized()
        addresses, sizes, _ = self._impl.chunk_table(tag)
        return list(zip(addresses, sizes))

    def get_chunk_states(self, tag: str) -> list[bool]:
        """One flag per chunk of the tag, in get_chunk_layout() order: True for a
        chunk that holds device memory now. CUDA only."""
        self._ensure_initialized()
        return self._impl.chunk_table(tag)[2]

    def resume_chunks(self, tag: str, chunk_indices: Sequence[int]) -> None:
        """Give device memory to the listed chunks of the tag that have none.

        All or nothing: raises torch.OutOfMemoryError, with no chunk changed, when
        the device has no memory for all of them. The contents of the new memory
        are undefined. Raises IndexError for an index outside the tag's chunks and
        RuntimeError while the tag is paused. CUDA only.
        """
        self._ensure_initialized()
        self._impl.resume_chunks(tag, chunk_indices)

    def release_region_pools(self, tag: Optional[str] = None) -> int:
        """Drop the memory pools of the regions of one tag (None: of every tag) and
        return how many were dropped.

        A pool keeps the memory of deleted tensors, paused or not, for the next
        region with the same arguments. Memory that no tensor occupies is freed at
        once; memory of live tensors follows when they are deleted and the cache is
        emptied. A later region() with the same arguments gets a new pool.

        Raises RuntimeError, dropping nothing, while one of the pools is in use by an
        open region or a CUDA graph. CUDA with PyTorch >= 2.8 only:
        NotImplementedError otherwise.
        """
        if not _is_cuda_build():
            raise NotImplementedError("release_region_pools() is only supported on CUDA")
        if _torch_version() < _MIN_TORCH_FOR_POOL_RELEASE:
            raise NotImplementedError(
                "release_region_pools() needs PyTorch >= "
                f"{'.'.join(map(str, _MIN_TORCH_FOR_POOL_RELEASE))}, found {torch.__version__}"
            )
        self._ensure_initialized()
        return self._impl.release_region_pools(tag)

    @contextmanager
    def tag_scope(
        self,
        tag: str = _TAG_DEFAULT,
        enable_cpu_backup: bool = False,
        enable_disk_backup: bool = False,
        cpu_backup_backend: Optional[CpuBackupBackend] = None,
        chunk_size: int = 0,
    ):
        """region() without its memory pool: only the tag, the backup settings and
        chunk_size of the calling thread, with the meaning they have for region().

        Memory that reaches torch_memory_saver while the scope is open gets them,
        whichever pool it is allocated into. Entering a pool is left to the
        caller: with a pool from create_mem_pool(), one pool can hold memory of
        several tags, each paused and resumed by its tag. Without a pool,
        hook_mode="preload" tags segments of PyTorch's default pool, which later
        tensors reuse.
        """
        self._validate_cpu_backup_backend(enable_cpu_backup, cpu_backup_backend)
        self._ensure_initialized()
        assert not (enable_cpu_backup and enable_disk_backup), \
            "enable_cpu_backup and enable_disk_backup are mutually exclusive"
        with self._impl._with_region_config(
            tag=tag,
            enable_cpu_backup=enable_cpu_backup,
            enable_disk_backup=enable_disk_backup,
            cpu_backup_backend=cpu_backup_backend,
            chunk_size=chunk_size,
        ):
            yield

    def create_mem_pool(self) -> "torch.cuda.MemPool":
        """A new memory pool on torch_memory_saver's allocator, on the current
        device. It belongs to the caller: torch_memory_saver keeps no reference.

        Allocate into it only inside tag_scope() or region(): outside of both, a
        request for a new segment ends the process in hook_mode="torch" and gets
        memory that cannot be paused in hook_mode="preload". Delete what uses the
        pool before the pool: up to PyTorch 2.9 destroying a pool that a CUDA
        graph still uses aborts, and up to 2.7 a segment that outlives its pool
        is returned with cudaFree, not through the pool's allocator. On ROCm and
        MUSA, hook_mode="torch" ends the process at a free outside of both as
        well: drop the pool inside one.
        """
        self._ensure_initialized()
        return self._impl.create_mem_pool()

    # for compatibility
    @property
    def enabled(self):
        return True

    @property
    def hook_mode(self):
        raise AttributeError

    @hook_mode.setter
    def hook_mode(self, hook_mode: HookMode):
        assert self._impl_ctor_kwargs is not None, "Cannot configure after initialization"
        self._impl_ctor_kwargs["hook_mode"] = hook_mode

    @property
    def memory_margin_bytes(self):
        raise NotImplementedError("Only setter is supported")

    @memory_margin_bytes.setter
    def memory_margin_bytes(self, value: int):
        self._ensure_initialized()
        if self._impl._is_xpu:
            raise NotImplementedError(
                "TorchMemorySaver.memory_margin_bytes is not supported on Intel "
                "XPU: the OOM-margin guard needs a device-wide free-bytes "
                "reading, which the Intel GPU driver does not provide reliably "
                "(free-bytes telemetry is frozen). Manage device memory headroom "
                "outside torch_memory_saver instead."
            )
        self._impl._binary_wrapper.cdll.set_memory_margin_bytes(value)

    @property
    def retain_cpu_backup(self) -> bool:
        self._ensure_initialized()
        return self._impl._binary_wrapper.cdll.tms_get_retain_cpu_backup()

    @retain_cpu_backup.setter
    def retain_cpu_backup(self, value: bool):
        self._ensure_initialized()
        self._impl._binary_wrapper.cdll.tms_set_retain_cpu_backup(value)

    def get_cpu_backup(self, x: torch.Tensor, zero_copy: bool = False):
        self._ensure_initialized()
        return self._impl.get_cpu_backup(x, zero_copy=zero_copy)

    def set_disk_backup_dir(self, path: str):
        """Set the directory for disk backup files (created if needed)."""
        self._ensure_initialized()
        os.makedirs(path, exist_ok=True)
        self._impl._binary_wrapper.cdll.tms_set_disk_backup_dir(path.encode("utf-8"))

    def _validate_cpu_backup_backend(
        self,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
    ) -> None:
        if cpu_backup_backend is None:
            return
        if not enable_cpu_backup:
            raise ValueError("cpu_backup_backend requires enable_cpu_backup=True")
        if is_xpu():
            raise ValueError(
                "cpu_backup_backend is not supported on XPU; host shadows use malloc"
            )
        if cpu_backup_backend == "mmap" and torch.version.hip:
            raise ValueError("cpu_backup_backend='mmap' is not supported on ROCm")

    def _ensure_initialized(self):
        if self._impl is not None:
            return
        self._impl = _TorchMemorySaverImpl(**self._impl_ctor_kwargs)
        del self._impl_ctor_kwargs


class _TorchMemorySaverImpl:
    def __init__(self, hook_mode: HookMode = "preload"):
        self._is_xpu = is_xpu()
        self._is_musa = is_musa()
        if self._is_xpu:
            assert hook_mode == "torch", (
                "XPU only supports hook_mode='torch' (in-process pluggable "
                "allocator); preload/LD_PRELOAD is CUDA-only."
            )
        self._hook_mode = hook_mode
        self._hook_util = HookUtilBase.create(hook_mode=hook_mode)
        self._binary_wrapper = BinaryWrapper(path_binary=self._hook_util.get_path_binary())

        self._device_module = torch.get_device_module()

        if self._is_xpu:
            # Prewarm per-device SYCL contexts before registering the allocator.
            if hasattr(self._binary_wrapper.cdll, "tms_xpu_prewarm_devices"):
                self._binary_wrapper.cdll.tms_xpu_prewarm_devices(
                    self._device_module.device_count()
                )

        self._mem_pools = defaultdict(
            lambda: self._device_module.MemPool(allocator=self._hook_util.get_allocator())
        )
        if self._is_musa:
            import threading

            # MusaMemPool.use() temporarily replaces the process-global
            # empty_cache, so overlapping disable() scopes are unsupported.
            self._musa_disable_lock = threading.Lock()
            self._musa_region_depth_local = threading.local()

        _sanity_checks()
        if torch.version.hip or self._is_xpu:
            # HIP/SYCL runtime static dtors may run before MemPool's at exit
            # (destruction-order fiasco); free pools via atexit while runtime alive.
            atexit.register(self._mem_pools.clear)

    @property
    def _musa_region_depth(self) -> int:
        """region() nesting depth on the calling thread (MUSA only)."""
        return getattr(self._musa_region_depth_local, "value", 0)

    @_musa_region_depth.setter
    def _musa_region_depth(self, depth: int) -> None:
        self._musa_region_depth_local.value = depth

    @contextmanager
    def region(
        self,
        tag: str,
        enable_cpu_backup: bool,
        enable_disk_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
        chunk_size: int = 0,
        initially_released: bool = False,
    ):
        if self._is_xpu and enable_disk_backup:
            raise NotImplementedError(
                "TorchMemorySaver disk backup (enable_disk_backup=True) is not "
                "supported on Intel XPU; use enable_cpu_backup=True instead."
            )
        if initially_released:
            if not _is_cuda_build():
                raise NotImplementedError("initially_released is only supported on CUDA")
            if (
                torch.are_deterministic_algorithms_enabled()
                and torch.utils.deterministic.fill_uninitialized_memory
            ):
                raise RuntimeError(
                    "An initially_released region cannot be used while deterministic "
                    "algorithms fill uninitialized memory: torch.empty would write to "
                    "address space that has no device memory"
                )
        # See https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # Key by device too: a MemPool is bound to its creation device, so a
        # multi-device process must not reuse one device's pool on another.
        # Include backend so torch-mode pool reuse cannot stick the first kind.
        device = self._current_device()
        resolved_cpu_backup_backend = ""
        if enable_cpu_backup:
            resolved_cpu_backup_backend = (
                cpu_backup_backend
                or self._binary_wrapper.cdll.tms_get_cpu_backup_backend().decode("utf-8")
            )
        key = (
            tag,
            enable_cpu_backup,
            enable_disk_backup,
            resolved_cpu_backup_backend,
            device,
        )
        if chunk_size or initially_released:
            # Pools of its own; the key of an ordinary region stays as it was.
            key += (chunk_size, initially_released)
        mem_pool = self._mem_pools[key]
        with self._device_module.use_mem_pool(mem_pool):
            with self._with_region_config(
                tag=tag,
                enable_cpu_backup=enable_cpu_backup,
                enable_disk_backup=enable_disk_backup,
                cpu_backup_backend=cpu_backup_backend,
                chunk_size=chunk_size,
                initially_released=initially_released,
            ):
                yield

    def create_mem_pool(self):
        return self._device_module.MemPool(allocator=self._hook_util.get_allocator())

    def _current_device(self) -> int:
        return self._device_module.current_device()

    @contextmanager
    def cuda_graph(
        self,
        cuda_graph,
        pool,
        stream,
        capture_error_mode,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
    ):
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"

        with self._device_module.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
            with self._with_region_config(
                tag=tag,
                enable_cpu_backup=enable_cpu_backup,
                cpu_backup_backend=cpu_backup_backend,
            ):
                yield

    @contextmanager
    def _with_region_config(
        self,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
        enable_disk_backup: bool = False,
        chunk_size: int = 0,
        initially_released: bool = False,
    ):
        if chunk_size:
            if not _is_cuda_build():
                raise NotImplementedError("chunk_size is only supported on CUDA")
            if enable_disk_backup:
                raise ValueError("chunk_size cannot be combined with enable_disk_backup")
            granularity = self.get_allocation_granularity(None)
            if chunk_size < 0 or chunk_size % granularity:
                raise ValueError(
                    f"chunk_size={chunk_size} is not a positive multiple of the device "
                    f"allocation granularity ({granularity} bytes)"
                )
        cdll = self._binary_wrapper.cdll
        orig_tag = cdll.tms_get_current_tag().decode("utf-8")
        orig_interesting_region = cdll.tms_get_interesting_region()
        orig_enable_cpu_backup = cdll.tms_get_enable_cpu_backup()
        orig_cpu_backup_backend = cdll.tms_get_cpu_backup_backend().decode("utf-8")
        orig_enable_disk_backup = cdll.tms_get_enable_disk_backup()
        orig_initially_released = _is_cuda_build() and cdll.tms_get_initially_released()
        orig_chunk_size = cdll.tms_get_chunk_size() if _is_cuda_build() else 0
        expected_cpu_backup_backend = cpu_backup_backend or orig_cpu_backup_backend

        self._binary_wrapper.set_config(
            tag=tag,
            interesting_region=True,
            enable_cpu_backup=enable_cpu_backup,
            cpu_backup_backend=cpu_backup_backend,
            enable_disk_backup=enable_disk_backup,
            initially_released=initially_released,
            chunk_size=chunk_size,
        )

        if self._is_musa:
            # Track nesting on MUSA so disable() rejects a region-enclosed scope instead of nesting two MemPool scopes.
            musa_region_depth = self._musa_region_depth
        try:
            if self._is_musa:
                self._musa_region_depth = musa_region_depth + 1
            yield
        finally:
            if self._is_musa:
                self._musa_region_depth = musa_region_depth
            assert cdll.tms_get_interesting_region()
            assert cdll.tms_get_enable_cpu_backup() == enable_cpu_backup
            assert (
                cdll.tms_get_cpu_backup_backend().decode("utf-8")
                == expected_cpu_backup_backend
            )
            assert cdll.tms_get_enable_disk_backup() == enable_disk_backup
            assert cdll.tms_get_current_tag().decode("utf-8") == tag
            self._binary_wrapper.set_config(
                tag=orig_tag,
                interesting_region=orig_interesting_region,
                enable_cpu_backup=orig_enable_cpu_backup,
                cpu_backup_backend=orig_cpu_backup_backend,
                enable_disk_backup=orig_enable_disk_backup,
                initially_released=orig_initially_released,
                chunk_size=orig_chunk_size,
            )

    @contextmanager
    def disable(self, dispose_mem_pool_after_use: bool = True):
        if self._is_xpu:
            raise NotImplementedError(
                "TorchMemorySaver.disable() is not supported on Intel XPU. "
                "Allocate outside a region() instead of using disable()."
            )
        if self._is_musa:
            with self._musa_disable(dispose_mem_pool_after_use):
                yield
            return

        assert dispose_mem_pool_after_use, "Only dispose_mem_pool_after_use=true is supported now"
        assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

        self._binary_wrapper.cdll.tms_set_interesting_region(False)
        try:
            # We can either reuse the pool or delete it immediately, and we implement the latter currently since Slime uses it.
            # About why we need a pool: https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
            pool = self._device_module.MemPool()
            with self._device_module.use_mem_pool(pool):
                yield
            del pool
        finally:
            self._binary_wrapper.cdll.tms_set_interesting_region(True)

    @contextmanager
    def _musa_disable(self, dispose_mem_pool_after_use: bool = True):
        """disable() on MUSA, where the pool needs an explicit release and
        MusaMemPool.use() temporarily replaces the process-global empty_cache."""
        from .musa_pool import MusaMemPool

        if self._musa_region_depth > 0:
            raise RuntimeError(
                "TorchMemorySaver.disable() cannot be used inside a region() on MUSA: "
                "the two MemPool scopes cannot be nested"
            )
        # The empty_cache shim is process-global, so overlapping scopes would
        # restore it in the wrong order.
        if not self._musa_disable_lock.acquire(blocking=False):
            raise RuntimeError(
                "TorchMemorySaver.disable() cannot be nested or called concurrently on MUSA"
            )
        try:
            assert dispose_mem_pool_after_use, "Only dispose_mem_pool_after_use=true is supported now"
            assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

            self._binary_wrapper.cdll.tms_set_interesting_region(False)
            try:
                musa_pool = MusaMemPool()
                try:
                    with musa_pool.use():
                        yield
                finally:
                    musa_pool.release()
            finally:
                self._binary_wrapper.cdll.tms_set_interesting_region(True)
        finally:
            self._musa_disable_lock.release()

    def pause(self, tag: Optional[str]):
        self._sync_affected_devices(tag)
        tag_bytes = tag.encode("utf-8") if tag else None
        ret = self._binary_wrapper.cdll.tms_pause(tag_bytes)
        if self._is_xpu and ret != 0:
            raise RuntimeError(
                f"torch_memory_saver.pause(tag={tag!r}) partially failed "
                f"(code {ret}); some allocations could not be released and are "
                "retained. Retry after resolving the failure."
            )

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        ret = self._binary_wrapper.cdll.tms_resume(tag_bytes)
        if self._is_xpu:
            if ret != 0:
                raise RuntimeError(
                    f"torch_memory_saver.resume(tag={tag!r}) partially failed "
                    f"(code {ret}); some allocations remain paused. Retry after "
                    "resolving the failure (e.g. free device memory)."
                )
            self._sync_affected_devices(tag)
        elif ret != 0:
            raise _OutOfMemoryError(
                f"torch_memory_saver.resume(tag={tag!r}) ran out of device memory "
                f"(code {ret}). Every matching allocation is still paused and keeps "
                "its backup; free device memory and call resume() again."
            )

    def get_allocation_granularity(self, device: Optional[int]) -> int:
        cdll = self._binary_wrapper.cdll
        if not hasattr(cdll, "tms_get_allocation_granularity"):
            raise NotImplementedError("get_allocation_granularity() is only supported on CUDA")
        # Reading the current device also initialises the driver, which the query needs.
        current_device = self._current_device()
        device = current_device if device is None else operator.index(device)
        device_count = self._device_module.device_count()
        # The hook ends the process when the driver does not know the device.
        if not 0 <= device < device_count:
            raise ValueError(f"device {device} is outside [0, {device_count})")
        return cdll.tms_get_allocation_granularity(device)

    def _chunk_cdll(self):
        if not _is_cuda_build():
            raise NotImplementedError("Chunked regions are only supported on CUDA")
        return self._binary_wrapper.cdll

    def chunk_table(self, tag: str) -> tuple[list[int], list[int], list[bool]]:
        """Address, size and state of every chunk of the tag."""
        get_layout = self._chunk_cdll().tms_get_chunk_layout
        tag_bytes = tag.encode("utf-8")
        count = get_layout(tag_bytes, None, None, None, 0)
        while True:
            addresses = (ctypes.c_uint64 * count)()
            sizes = (ctypes.c_size_t * count)()
            states = (ctypes.c_uint8 * count)()
            found = get_layout(tag_bytes, addresses, sizes, states, count)
            # Otherwise the tag gained or lost an allocation between the two calls.
            if found == count:
                return list(addresses), list(sizes), [bool(state) for state in states]
            count = found

    def resume_chunks(self, tag: str, chunk_indices: Sequence[int]) -> None:
        indices = (ctypes.c_size_t * len(chunk_indices))(*chunk_indices)
        status = self._chunk_cdll().tms_resume_chunks(tag.encode("utf-8"), indices, len(indices))
        _raise_for_chunk_status(status, f"resume_chunks(tag={tag!r})")

    def release_region_pools(self, tag: Optional[str]) -> int:
        # A pool is named inside the comprehension only: the traceback of the error below keeps
        # this frame, and a pool bound in it would live as long as that error, through a later
        # call that drops the pool.
        # list(): a region on another thread may add its pool meanwhile.
        selected = [
            (key, pool.use_count() != 1)
            for key, pool in list(self._mem_pools.items())
            if tag is None or key[0] == tag
        ]
        in_use = [key for key, used in selected if used]
        if in_use:
            raise RuntimeError(
                f"torch_memory_saver.release_region_pools(tag={tag!r}): the memory pools "
                f"{in_use} are in use by an open region or a CUDA graph; no pool was released"
            )
        dropped = 0
        for key, _ in selected:
            # pop(): a call on another thread may have dropped the pool meanwhile.
            if self._mem_pools.pop(key, None) is not None:
                dropped += 1
        return dropped

    def _sync_affected_devices(self, tag: Optional[str]):
        for device in self._affected_devices(tag):
            self._device_module.synchronize(device)

    def _affected_devices(self, tag: Optional[str]) -> list[int]:
        """Read tag-filtered device ids from the backend's locked allocation map."""
        device_ids = (ctypes.c_int * 64)()
        count = self._binary_wrapper.cdll.tms_affected_devices(
            tag.encode("utf-8") if tag else None,
            device_ids,
            len(device_ids),
        )
        assert count <= len(device_ids), "Too many affected devices"
        return device_ids[:count]

    def get_cpu_backup(self, x: torch.Tensor, zero_copy: bool = False):
        assert x.is_cuda or x.is_xpu or (self._is_musa and x.device.type == "musa"), f"{x.device=}"
        assert x.is_contiguous(), f"{x.shape=} {x.stride()=} {x.dtype=}"

        nbytes = x.nbytes
        gpu_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_uint8))
        cpu_ptr = self._binary_wrapper.cdll.tms_get_cpu_backup_pointer(gpu_ptr, nbytes)
        if not cpu_ptr:
            return None

        np_untyped = np.ctypeslib.as_array(cpu_ptr, shape=(nbytes,))
        assert np_untyped.dtype == np.uint8, f"{np_untyped.dtype=} {np_untyped.shape=}"

        ans_untyped = torch.from_numpy(np_untyped)
        ans = ans_untyped.view(x.dtype).view(x.shape)

        # For simplicity and safety
        if not zero_copy:
            ans = ans.clone()

        assert ans.device == torch.device("cpu"), f"{ans.device=}"
        assert ans.dtype == x.dtype, f"{ans.dtype=} {x.dtype=}"
        assert ans.shape == x.shape, f"{ans.shape=} {x.shape=}"
        assert ans.stride() == x.stride(), f"{ans.stride()=} {x.stride()=}"
        return ans

def _sanity_checks():
    alloc_confs = ["PYTORCH_CUDA_ALLOC_CONF"]
    if is_musa():
        alloc_confs.append("PYTORCH_MUSA_ALLOC_CONF")
    if any("expandable_segments:True" in os.environ.get(name, "") for name in alloc_confs):
        raise RuntimeError(
            "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet."
        )
