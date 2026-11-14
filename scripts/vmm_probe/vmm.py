"""ctypes bindings for the CUDA driver's virtual memory calls, for the probes in this directory.

Cluster scratch, not part of the library. PyTorch owns the context: every probe touches the
device through torch first, so the driver calls below run in torch's primary context.
"""

import ctypes

import torch

MiB = 1 << 20
GiB = 1 << 30

_ALLOCATION_TYPE_PINNED = 1
_LOCATION_TYPE_DEVICE = 1
_ACCESS_READWRITE = 3
_GRANULARITY_MINIMUM = 0


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


class _Raw:
    """A device address range, presented to torch through the CUDA array interface."""

    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 2,
        }


class Driver:
    """The driver calls a chunked region needs, on one device."""

    def __init__(self, device=0):
        self.device = device
        torch.zeros(1, device=f"cuda:{device}")  # makes torch's context current on this thread
        self.lib = ctypes.CDLL("libcuda.so.1")
        u64, sz, vp = ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_void_p
        signatures = {
            "cuCtxGetCurrent": [ctypes.POINTER(vp)],
            "cuMemGetAllocationGranularity": [ctypes.POINTER(sz), ctypes.POINTER(_AllocationProp), ctypes.c_int],
            "cuMemAddressReserve": [ctypes.POINTER(u64), sz, sz, u64, u64],
            "cuMemAddressFree": [u64, sz],
            "cuMemCreate": [ctypes.POINTER(u64), sz, ctypes.POINTER(_AllocationProp), u64],
            "cuMemRelease": [u64],
            "cuMemMap": [u64, sz, sz, u64, u64],
            "cuMemUnmap": [u64, sz],
            "cuMemSetAccess": [u64, sz, ctypes.POINTER(_AccessDesc), sz],
        }
        for name, argtypes in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = argtypes, ctypes.c_int
        ctx = vp()
        self._check(self.lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            raise RuntimeError("no current CUDA context after touching the device through torch")
        self._prop = _AllocationProp()
        self._prop.type = _ALLOCATION_TYPE_PINNED
        self._prop.location = _Location(_LOCATION_TYPE_DEVICE, device)
        self._access = _AccessDesc(_Location(_LOCATION_TYPE_DEVICE, device), _ACCESS_READWRITE)

    @staticmethod
    def _check(code, what):
        if code != 0:
            raise RuntimeError(f"{what} failed with CUDA driver error {code}")

    def granularity(self):
        out = ctypes.c_size_t()
        self._check(
            self.lib.cuMemGetAllocationGranularity(ctypes.byref(out), ctypes.byref(self._prop), _GRANULARITY_MINIMUM),
            "cuMemGetAllocationGranularity",
        )
        return out.value

    def reserve(self, size):
        ptr = ctypes.c_ulonglong()
        self._check(self.lib.cuMemAddressReserve(ctypes.byref(ptr), size, 0, 0, 0), "cuMemAddressReserve")
        return ptr.value

    def address_free(self, ptr, size):
        self._check(self.lib.cuMemAddressFree(ptr, size), "cuMemAddressFree")

    def create(self, size):
        handle = ctypes.c_ulonglong()
        self._check(self.lib.cuMemCreate(ctypes.byref(handle), size, ctypes.byref(self._prop), 0), "cuMemCreate")
        return handle.value

    def release(self, handle):
        self._check(self.lib.cuMemRelease(handle), "cuMemRelease")

    def map(self, ptr, size, handle):
        self._check(self.lib.cuMemMap(ptr, size, 0, handle, 0), "cuMemMap")

    def unmap(self, ptr, size):
        self._check(self.lib.cuMemUnmap(ptr, size), "cuMemUnmap")

    def set_access(self, ptr, size):
        self._check(self.lib.cuMemSetAccess(ptr, size, ctypes.byref(self._access), 1), "cuMemSetAccess")

    def tensor(self, ptr, nbytes, dtype=torch.float32):
        """A tensor over [ptr, ptr + nbytes). The caller keeps the range mapped while it lives."""
        typestr = {torch.float32: "<f4", torch.uint8: "|u1", torch.int64: "<i8"}[dtype]
        count = nbytes // torch.empty(0, dtype=dtype).element_size()
        return torch.as_tensor(_Raw(ptr, (count,), typestr), device=f"cuda:{self.device}")
