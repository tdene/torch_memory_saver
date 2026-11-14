#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"
#if defined(USE_MUSA)
#include "hardware_musa_support.h"
#endif

#include <algorithm>
#include <set>

#if defined(USE_CUDA)
namespace {

// cuMemAddressReserve and cuMemMap document CUDA_ERROR_OUT_OF_MEMORY like cuMemCreate does.
// Only that result is handed back, as cu_mem_create does; anything else stays fatal.
cudaError_t try_reserve(void **ptr, size_t size) {
    const CUresult ret = cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0);
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        std::cerr << "[torch_memory_saver.cpp] cuMemAddressReserve CUDA_ERROR_OUT_OF_MEMORY" << std::endl;
        return cudaErrorMemoryAllocation;
    }
    CURESULT_CHECK(ret);
    return cudaSuccess;
}

cudaError_t try_map(void *ptr, size_t size, CUmemGenericAllocationHandle handle, CUdevice device) {
    const CUresult ret = cuMemMap((CUdeviceptr) ptr, size, 0, handle, 0);
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        std::cerr << "[torch_memory_saver.cpp] cuMemMap CUDA_ERROR_OUT_OF_MEMORY" << std::endl;
        return cudaErrorMemoryAllocation;
    }
    CURESULT_CHECK(ret);
    CUDAUtils::cu_mem_set_access(ptr, size, device);
    return cudaSuccess;
}

// One address range that a call has to back with a handle of its own.
struct PendingMapping {
    void *address;
    size_t size;
    CUdevice device;
    // Where the allocation keeps the handle. While an allocation or a chunk has no memory
    // the value there is stale, so create_all() writes the new one in place.
    CUmemGenericAllocationHandle *handle;
    bool created = false;
    bool mapped = false;
};

// All or nothing: on out of memory the mappings made by this call are removed again.
cudaError_t map_all(std::vector<PendingMapping> &pending) {
    for (PendingMapping &mapping : pending) {
        const cudaError_t ret = try_map(mapping.address, mapping.size, *mapping.handle, mapping.device);
        if (ret != cudaSuccess) {
            for (PendingMapping &done : pending) {
                if (done.mapped) {
                    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) done.address, done.size));
                    done.mapped = false;
                }
            }
            return ret;
        }
        mapping.mapped = true;
    }
    return cudaSuccess;
}

void release_created(const std::vector<PendingMapping> &pending) {
    for (const PendingMapping &mapping : pending) {
        if (mapping.created) {
            CURESULT_CHECK(cuMemRelease(*mapping.handle));
        }
    }
}

// All or nothing: on out of memory the handles made by this call are released again.
cudaError_t create_all(std::vector<PendingMapping> &pending) {
    for (PendingMapping &mapping : pending) {
        const cudaError_t ret = CUDAUtils::cu_mem_create(mapping.handle, mapping.size, mapping.device);
        if (ret != cudaSuccess) {
            release_created(pending);
            return ret;
        }
        mapping.created = true;
    }
    return cudaSuccess;
}

// Only the last chunk can be shorter than chunk_size.
size_t chunk_bytes(const AllocationMetadata &metadata, size_t index) {
    return std::min(metadata.chunk_size, metadata.allocation_size - index * metadata.chunk_size);
}

void *chunk_ptr(void *ptr, const AllocationMetadata &metadata, size_t index) {
    return static_cast<uint8_t *>(ptr) + index * metadata.chunk_size;
}

void release_own_chunks(void *ptr, const AllocationMetadata &metadata) {
    for (size_t i = 0; i < metadata.chunks.size(); ++i) {
        if (metadata.chunks[i].own) {
            CURESULT_CHECK(cuMemUnmap(
                (CUdeviceptr) chunk_ptr(ptr, metadata, i), chunk_bytes(metadata, i)));
            CURESULT_CHECK(cuMemRelease(metadata.chunks[i].handle));
        }
    }
}

// One copy per run of neighbouring chunks that are in the cpu_backup slot: they are
// neighbours there too.
void copy_backed_up_chunks(void *ptr, const AllocationMetadata &metadata, bool to_host) {
    const size_t count = metadata.chunks.size();
    for (size_t first = 0; first < count;) {
        if (metadata.chunks[first].backup_offset == kNoChunkBackup) {
            ++first;
            continue;
        }
        size_t end = first;
        size_t bytes = 0;
        for (; end < count && metadata.chunks[end].backup_offset != kNoChunkBackup; ++end) {
            bytes += chunk_bytes(metadata, end);
        }
        CpuBackupSlot run{
            metadata.cpu_backup.kind,
            static_cast<uint8_t *>(metadata.cpu_backup.data) + metadata.chunks[first].backup_offset,
            bytes};
        void *run_ptr = chunk_ptr(ptr, metadata, first);
        if (to_host) {
            cpu_backuper::offload(run_ptr, bytes, run);
        } else {
            cpu_backuper::onload(run_ptr, bytes, metadata.device, run);
        }
        first = end;
    }
}

// Packs the chunks that hold memory into the cpu_backup slot in index order: an allocation
// that is partly backed costs host memory for what it holds only.
void offload_chunks(void *ptr, AllocationMetadata &metadata) {
    size_t total = 0;
    for (size_t i = 0; i < metadata.chunks.size(); ++i) {
        Chunk &chunk = metadata.chunks[i];
        chunk.backup_offset = chunk.own ? total : kNoChunkBackup;
        if (chunk.own) {
            total += chunk_bytes(metadata, i);
        }
    }
    if (metadata.cpu_backup.size != total) {
        cpu_backuper::release(metadata.cpu_backup);
    }
    if (total == 0) {
        return;
    }
    if (metadata.cpu_backup.data == nullptr) {
        cpu_backuper::allocate(total, metadata.cpu_backup);
    }
    copy_backed_up_chunks(ptr, metadata, true);
}

}  // namespace
#endif

TorchMemorySaver::TorchMemorySaver()
    : disk_backend_(compute_disk_backup_dir_from_env(), compute_disk_chunk_bytes_from_env()) {
#if TMS_SUPPORTS_RETAIN_CPU_BACKUP
    retain_cpu_backup_.store(get_bool_env_var("TMS_RETAIN_CPU_BACKUP"));
#endif
}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(
    void **ptr,
    CUdevice device,
    size_t raw_size,
    const std::string& tag,
    const bool enable_cpu_backup,
    const CpuBackupKind cpu_backup_kind,
    const bool enable_disk_backup,
    const bool initially_released,
    const size_t chunk_size) {
    // Enforce here, not only in the Python layer: an assert is stripped under
    // python -O and bypassed by direct C-API / env-var use.
    SIMPLE_CHECK(!(enable_cpu_backup && enable_disk_backup),
                 "cpu_backup and disk_backup are mutually exclusive");
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(!enable_disk_backup, "disk backup is not supported on the ROCm 6.x legacy chunked path");
    return ROCmHIPImplementation::rocm_malloc(
        ptr,
        device,
        raw_size,
        tag,
        enable_cpu_backup,
        cpu_backup_kind,
        allocation_metadata_,
        allocator_metadata_mutex_);

#elif defined(USE_XPU)
    SIMPLE_CHECK(!enable_disk_backup, "disk backup is not supported on Intel XPU");
    return XPUImplementation::xpu_malloc(ptr, device, raw_size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#else
    const size_t allocation_size = CUDAUtils::cu_mem_get_allocation_size(raw_size, device);
#if defined(USE_CUDA)
    if (chunk_size > 0) {
        AllocationMetadata metadata{
            raw_size, device, tag, AllocationState::ACTIVE,
            enable_cpu_backup, CpuBackupSlot{cpu_backup_kind, nullptr, 0},
            enable_disk_backup, DiskBackupSlot{}, allocation_size, 0};
        metadata.chunk_size = chunk_size;
        return malloc_chunked(ptr, std::move(metadata), initially_released);
    }
    if (initially_released) {
        // Address space only: no device memory is requested, so the margin does not apply.
        void *address = nullptr;
        const cudaError_t reserve_ret = try_reserve(
            &address, CUDAUtils::cu_mem_get_reserve_size(allocation_size, device));
        if (reserve_ret != cudaSuccess) {
            return reserve_ret;
        }
        *ptr = address;
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            address,
            AllocationMetadata{
                raw_size, device, tag, AllocationState::PAUSED,
                enable_cpu_backup, CpuBackupSlot{cpu_backup_kind, nullptr, 0},
                enable_disk_backup, DiskBackupSlot{}, allocation_size, 0, true});
        return cudaSuccess;
    }
#endif
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + allocation_size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " allocation_size=" << allocation_size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, allocation_size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

#if defined(USE_CUDA)
    // Out of memory from the reserve or the map is returned like out of memory from the
    // create: what the request obtained is given back and *ptr is not written.
    const size_t reserve_size = CUDAUtils::cu_mem_get_reserve_size(allocation_size, device);
    void *address = nullptr;
    ret = try_reserve(&address, reserve_size);
    if (ret != cudaSuccess) {
        CURESULT_CHECK(cuMemRelease(allocHandle));
        return ret;
    }
    ret = try_map(address, allocation_size, allocHandle, device);
    if (ret != cudaSuccess) {
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) address, reserve_size));
        CURESULT_CHECK(cuMemRelease(allocHandle));
        return ret;
    }
    *ptr = address;
#else
    CURESULT_CHECK(cuMemAddressReserve(
        (CUdeviceptr *) ptr, CUDAUtils::cu_mem_get_reserve_size(allocation_size, device), 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, allocation_size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, allocation_size, device);
#endif

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{
                raw_size, device, tag, AllocationState::ACTIVE,
                enable_cpu_backup, CpuBackupSlot{cpu_backup_kind, nullptr, 0},
                enable_disk_backup, DiskBackupSlot{}, allocation_size, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " raw_size=" << raw_size
              << " allocation_size=" << allocation_size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

#if defined(USE_CUDA)
cudaError_t TorchMemorySaver::malloc_chunked(
    void **ptr, AllocationMetadata metadata, const bool initially_released) {
    SIMPLE_CHECK(!metadata.enable_disk_backup, "disk backup is not supported for chunked regions");
    const size_t granularity = CUDAUtils::cu_mem_get_allocation_size(1, metadata.device);
    SIMPLE_CHECK(metadata.chunk_size % granularity == 0,
                 "chunk_size must be a multiple of the device allocation granularity"
                 << " chunk_size=" << metadata.chunk_size << " granularity=" << granularity);
    metadata.chunks.resize(
        (metadata.allocation_size + metadata.chunk_size - 1) / metadata.chunk_size);

    // Every handle, then the address range, then every mapping: out of memory at any of them
    // gives back what the request obtained and leaves *ptr unwritten, as in malloc. An
    // initially released allocation is born active with no chunk holding memory.
    std::vector<PendingMapping> pending;
    if (!initially_released) {
        for (size_t i = 0; i < metadata.chunks.size(); ++i) {
            metadata.chunks[i].own = true;
            pending.push_back(PendingMapping{
                nullptr, chunk_bytes(metadata, i), metadata.device, &metadata.chunks[i].handle});
        }
    }
    cudaError_t ret = create_all(pending);
    if (ret != cudaSuccess) {
        return ret;
    }
    const size_t reserve_size =
        CUDAUtils::cu_mem_get_reserve_size(metadata.allocation_size, metadata.device);
    void *address = nullptr;
    ret = try_reserve(&address, reserve_size);
    if (ret != cudaSuccess) {
        release_created(pending);
        return ret;
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        pending[i].address = chunk_ptr(address, metadata, i);
    }
    ret = map_all(pending);
    if (ret != cudaSuccess) {
        release_created(pending);
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) address, reserve_size));
        return ret;
    }
    *ptr = address;

    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    metadata.sequence = next_sequence_++;
    allocation_metadata_.emplace(address, std::move(metadata));
    return cudaSuccess;
}
#endif

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_XPU)
    return XPUImplementation::xpu_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

#if defined(USE_CUDA)
    if (metadata.chunk_size != 0) {
        if (AllocationState::ACTIVE == metadata.state) {
            release_own_chunks(ptr, metadata);
        }
    } else
#endif
    if (AllocationState::ACTIVE == metadata.state) {
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    }
    CURESULT_CHECK(cuMemAddressFree(
        (CUdeviceptr) ptr, CUDAUtils::cu_mem_get_reserve_size(metadata.allocation_size, metadata.device)));

    cpu_backuper::release(metadata.cpu_backup);

    if (metadata.enable_disk_backup) {
        disk_backend_.release(metadata.disk);
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
              << " metadata.allocation_size=" << metadata.allocation_size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

#if defined(USE_XPU)
uint64_t TorchMemorySaver::xpu_committed_bytes(int device_id) {
    return XPUImplementation::xpu_committed_bytes(
        device_id, allocation_metadata_, allocator_metadata_mutex_);
}

uint64_t TorchMemorySaver::xpu_leaked_bytes(int device_id) {
    return XPUImplementation::xpu_leaked_bytes(
        device_id, allocation_metadata_, allocator_metadata_mutex_);
}

uint64_t TorchMemorySaver::xpu_tracked_bytes(int device_id) {
    return XPUImplementation::xpu_tracked_bytes(
        device_id, allocation_metadata_, allocator_metadata_mutex_);
}
#endif

uint32_t TorchMemorySaver::affected_devices(const char* tag, int* out_device_ids, uint32_t capacity) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const bool all = (tag == nullptr || tag[0] == '\0');
    std::set<int> devices;
    for (const auto &kv : allocation_metadata_) {
        const AllocationMetadata &metadata = kv.second;
        if (!all && metadata.tag != tag)
            continue;
        devices.insert(static_cast<int>(metadata.device));
    }
    const auto count = static_cast<uint32_t>(devices.size());
    if (out_device_ids != nullptr) {
        std::copy_n(devices.begin(), std::min(capacity, count), out_device_ids);
    }
    return count;
}

cudaError_t TorchMemorySaver::pause(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);
    return cudaSuccess;

#elif defined(USE_XPU)
    return XPUImplementation::xpu_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

#if defined(USE_CUDA)
        if (metadata.chunk_size != 0) {
            if (metadata.enable_cpu_backup) {
                offload_chunks(ptr, metadata);
            }
            release_own_chunks(ptr, metadata);
            metadata.state = AllocationState::PAUSED;
            continue;
        }
#endif

        if (metadata.enable_cpu_backup) {
            cpu_backuper::offload(ptr, metadata.raw_size, metadata.cpu_backup);
        } else if (metadata.enable_disk_backup) {
            disk_backend_.offload(ptr, metadata.raw_size, metadata.disk);
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
    return cudaSuccess;
#endif
}

cudaError_t TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);
    return cudaSuccess;

#elif defined(USE_XPU)
    return XPUImplementation::xpu_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
    const bool retain_cpu_backup = retain_cpu_backup_.load();

#if defined(USE_CUDA)
    // Every handle is created and mapped before a backup or a state is touched: out of memory
    // leaves every matching allocation paused with its backup, and the call can be repeated.
    std::vector<PendingMapping> pending;
    for (auto &kv : allocation_metadata_) {
        AllocationMetadata &metadata = kv.second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        // Checked for all of them before memory is requested, so that a tag in a wrong state
        // is not reported as out of memory when memory is short.
        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)kv.first)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.chunk_size != 0) {
            for (size_t i = 0; i < metadata.chunks.size(); ++i) {
                if (metadata.chunks[i].own) {
                    pending.push_back(PendingMapping{
                        chunk_ptr(kv.first, metadata, i), chunk_bytes(metadata, i),
                        metadata.device, &metadata.chunks[i].handle});
                }
            }
            continue;
        }

        pending.push_back(PendingMapping{
            kv.first, metadata.allocation_size, metadata.device, &metadata.allocHandle});
    }
    cudaError_t pending_ret = create_all(pending);
    if (pending_ret != cudaSuccess) {
        return pending_ret;
    }
    pending_ret = map_all(pending);
    if (pending_ret != cudaSuccess) {
        release_created(pending);
        return pending_ret;
    }
#endif

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

#if defined(USE_CUDA)
        if (metadata.chunk_size != 0) {
            if (metadata.cpu_backup.data != nullptr) {
                copy_backed_up_chunks(ptr, metadata, false);
                if (!retain_cpu_backup) {
                    cpu_backuper::release(metadata.cpu_backup);
                }
            }
            metadata.state = AllocationState::ACTIVE;
            continue;
        }
#endif

        CUmemGenericAllocationHandle newAllocHandle;
#if defined(USE_CUDA)
        newAllocHandle = metadata.allocHandle;
#else
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(
            &newAllocHandle, metadata.allocation_size, metadata.device));

        CURESULT_CHECK(cuMemMap(
            (CUdeviceptr) ptr, metadata.allocation_size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.allocation_size, metadata.device);
#endif

#if defined(USE_MUSA)
        // MUSA may hand back non-zero pages for a fresh mapping, so zero a
        // region that has no backup to restore. Backed regions are overwritten
        // by onload, and zeroing them first would double the resume traffic.
        if (!metadata.enable_cpu_backup && !metadata.enable_disk_backup) {
            MUSAImplementation::zero_region(
                ptr, metadata.allocation_size, static_cast<int>(metadata.device));
        }
#endif

        if (metadata.born_released) {
            metadata.born_released = false;
        } else if (metadata.enable_cpu_backup) {
            cpu_backuper::onload(
                ptr, metadata.raw_size, metadata.device, metadata.cpu_backup);
            if (!retain_cpu_backup) {
                cpu_backuper::release(metadata.cpu_backup);
            }
        } else if (metadata.enable_disk_backup) {
            disk_backend_.onload(ptr, metadata.raw_size, metadata.disk);
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size
                  << " (old)metadata.allocHandle=" << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
    return cudaSuccess;
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED || defined(USE_XPU)
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.raw_size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            // Disk-backed allocations have no CPU-resident copy; callers must use the resumed GPU tensor.
            if (metadata.enable_disk_backup) {
                return nullptr;
            }
#if defined(USE_CUDA)
            if (metadata.chunk_size != 0) {
                // The slot packs the backed-up chunks in index order, so a range has one host
                // address when every chunk it touches is in the slot.
                if (metadata.cpu_backup.data == nullptr) {
                    return nullptr;
                }
                const size_t last_chunk = metadata.chunks.size() - 1;
                const size_t first = std::min<size_t>(offset / metadata.chunk_size, last_chunk);
                const size_t last = std::min<size_t>(
                    (offset + std::max<uint64_t>(query_size, 1) - 1) / metadata.chunk_size, last_chunk);
                for (size_t i = first; i <= last; ++i) {
                    if (metadata.chunks[i].backup_offset == kNoChunkBackup) {
                        return nullptr;
                    }
                }
                return (uint8_t*) metadata.cpu_backup.data + metadata.chunks[first].backup_offset
                    + (offset - first * metadata.chunk_size);
            }
#endif
            if (metadata.state == AllocationState::ACTIVE) {
                return metadata.cpu_backup.data == nullptr
                    ? nullptr
                    : (uint8_t*) metadata.cpu_backup.data + offset;
            } else {
                SIMPLE_CHECK(nullptr != metadata.cpu_backup.data,
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist, do you forget to enable cpu backup");
                return (uint8_t*) metadata.cpu_backup.data + offset;
            }
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}

#if defined(USE_CUDA)
std::vector<TorchMemorySaver::ChunkRef> TorchMemorySaver::chunks_of_tag(const std::string& tag) {
    std::vector<std::pair<void*, AllocationMetadata*>> allocations;
    for (auto &kv : allocation_metadata_) {
        if (kv.second.chunk_size != 0 && kv.second.tag == tag) {
            allocations.emplace_back(kv.first, &kv.second);
        }
    }
    std::sort(allocations.begin(), allocations.end(), [](const auto &a, const auto &b) {
        return a.second->sequence < b.second->sequence;
    });
    std::vector<ChunkRef> refs;
    for (const auto &allocation : allocations) {
        for (size_t i = 0; i < allocation.second->chunks.size(); ++i) {
            refs.push_back(ChunkRef{allocation.first, allocation.second, i});
        }
    }
    return refs;
}

size_t TorchMemorySaver::get_chunk_layout(
    const std::string& tag, uint64_t* out_addresses, size_t* out_sizes, uint8_t* out_states,
    size_t capacity) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const std::vector<ChunkRef> refs = chunks_of_tag(tag);
    for (size_t i = 0; i < std::min(capacity, refs.size()); ++i) {
        const AllocationMetadata &metadata = *refs[i].metadata;
        out_addresses[i] = (uint64_t) chunk_ptr(refs[i].base, metadata, refs[i].index);
        out_sizes[i] = chunk_bytes(metadata, refs[i].index);
        out_states[i] =
            metadata.state == AllocationState::ACTIVE && metadata.chunks[refs[i].index].own;
    }
    return refs.size();
}

int TorchMemorySaver::resume_chunks(const std::string& tag, const size_t* indices, size_t count) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const std::vector<ChunkRef> refs = chunks_of_tag(tag);
    const std::set<size_t> wanted(indices, indices + count);

    std::vector<PendingMapping> pending;
    for (const size_t index : wanted) {
        if (index >= refs.size()) {
            return TMS_CHUNK_INDEX_OUT_OF_RANGE;
        }
        AllocationMetadata &metadata = *refs[index].metadata;
        if (metadata.state != AllocationState::ACTIVE) {
            return TMS_CHUNK_ALLOCATION_PAUSED;
        }
        Chunk &chunk = metadata.chunks[refs[index].index];
        if (!chunk.own) {
            pending.push_back(PendingMapping{
                chunk_ptr(refs[index].base, metadata, refs[index].index),
                chunk_bytes(metadata, refs[index].index), metadata.device, &chunk.handle});
        }
    }
    // Every handle first, then every mapping: out of memory leaves the chunks as they were.
    if (create_all(pending) != cudaSuccess) {
        return TMS_CHUNK_OUT_OF_MEMORY;
    }
    if (map_all(pending) != cudaSuccess) {
        release_created(pending);
        return TMS_CHUNK_OUT_OF_MEMORY;
    }
    for (const size_t index : wanted) {
        refs[index].metadata->chunks[refs[index].index].own = true;
    }
    return TMS_CHUNK_OK;
}
#endif
