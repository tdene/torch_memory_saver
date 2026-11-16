#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"
#if defined(USE_MUSA)
#include "hardware_musa_support.h"
#endif

#include <algorithm>
#include <condition_variable>
#include <set>
#include <thread>

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

// The most that one copy moves between a chunked allocation and its cpu_backup slot. The
// copies run beside the mapping calls of pause(), which follow them piece by piece, and of
// resume(), which they follow.
constexpr size_t kBackupCopyPieceBytes = size_t(1) << 30;

// One piece of a chunked allocation's backup: host.size bytes at ptr and at host.data.
struct BackupPiece {
    void *ptr;
    CpuBackupSlot host;
    CUdevice device;
    // resume(): how many of the call's mappings have to be made before the piece is written.
    size_t mappings;
};

// Copies the pieces in order on a thread of its own, in the caller's context, while the
// caller makes its mapping calls. The thread reads its pieces only: the caller keeps
// allocator_metadata_mutex_, and ends the thread with join() or stop() before it returns.
class BackupCopy {
public:
    BackupCopy(std::vector<BackupPiece> pieces, bool to_host)
        : pieces_(std::move(pieces)), to_host_(to_host) {
        if (pieces_.empty()) {
            return;
        }
        CUcontext context = nullptr;
        CURESULT_CHECK(cuCtxGetCurrent(&context));
        thread_ = std::thread([this, context] { run(context); });
    }

    // What the side that leads has reached: the copy of pause() in bytes that are on the
    // host, the caller of resume() in mappings that are made.
    void advance(size_t reached) {
        const std::lock_guard<std::mutex> lock(mutex_);
        reached_ = reached;
        if (reached_ >= awaited_) {
            changed_.notify_one();
        }
    }

    // The side that follows waits for it. False once stop() was called.
    bool wait_for(size_t awaited) {
        std::unique_lock<std::mutex> lock(mutex_);
        awaited_ = awaited;
        changed_.wait(lock, [this] { return reached_ >= awaited_ || stopped_; });
        return !stopped_;
    }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    // The mappings that are left will not be made: the copy ends after the piece it is at.
    void stop() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        changed_.notify_one();
        join();
    }

private:
    void run(CUcontext context) {
        CURESULT_CHECK(cuCtxSetCurrent(context));
        size_t copied = 0;
        for (BackupPiece &piece : pieces_) {
            if (to_host_) {
                cpu_backuper::offload(piece.ptr, piece.host.size, piece.host);
                copied += piece.host.size;
                advance(copied);
            } else if (wait_for(piece.mappings)) {
                cpu_backuper::onload(piece.ptr, piece.host.size, piece.device, piece.host);
            } else {
                return;
            }
        }
    }

    std::vector<BackupPiece> pieces_;
    const bool to_host_;
    std::mutex mutex_;
    std::condition_variable changed_;
    size_t reached_ = 0;
    size_t awaited_ = 0;
    bool stopped_ = false;
    std::thread thread_;
};

// One address range that a call has to map: through a handle of its own, which create_all()
// makes, or, when it is added after create_all(), through the handle of another chunk.
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

// All or nothing: on out of memory the mappings made by this call are removed again, once
// the copy that follows them, if there is one, has stopped.
cudaError_t map_all(std::vector<PendingMapping> &pending, BackupCopy *copy = nullptr) {
    for (size_t i = 0; i < pending.size(); ++i) {
        PendingMapping &mapping = pending[i];
        const cudaError_t ret = try_map(mapping.address, mapping.size, *mapping.handle, mapping.device);
        if (ret != cudaSuccess) {
            if (copy != nullptr) {
                copy->stop();
            }
            for (PendingMapping &done : pending) {
                if (done.mapped) {
                    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) done.address, done.size));
                    done.mapped = false;
                }
            }
            return ret;
        }
        mapping.mapped = true;
        if (copy != nullptr) {
            copy->advance(i + 1);
        }
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

// A chunk that map_chunks mapped shares the memory of its source: unmapping it frees none.
void unmap_borrowed_chunks(void *ptr, const AllocationMetadata &metadata) {
    for (size_t i = 0; i < metadata.chunks.size(); ++i) {
        if (metadata.chunks[i].source != nullptr) {
            CURESULT_CHECK(cuMemUnmap(
                (CUdeviceptr) chunk_ptr(ptr, metadata, i), chunk_bytes(metadata, i)));
        }
    }
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

// The copy of a chunked allocation's backup, cut in pieces: a run of neighbouring chunks that
// are in the cpu_backup slot, where they are neighbours too, is copied kBackupCopyPieceBytes
// at a time. resume() passes, for every address it maps, the number of its mappings after
// which the address can be written: a piece waits for the last of its chunks.
void add_backup_pieces(
    void *ptr, const AllocationMetadata &metadata, std::vector<BackupPiece> &pieces,
    const std::unordered_map<void *, size_t> *writable_after = nullptr) {
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
        uint8_t *run_ptr = static_cast<uint8_t *>(chunk_ptr(ptr, metadata, first));
        uint8_t *run_host =
            static_cast<uint8_t *>(metadata.cpu_backup.data) + metadata.chunks[first].backup_offset;
        for (size_t offset = 0; offset < bytes; offset += kBackupCopyPieceBytes) {
            const size_t piece_bytes = std::min(kBackupCopyPieceBytes, bytes - offset);
            size_t mappings = 0;
            if (writable_after != nullptr) {
                const size_t last = first + (offset + piece_bytes - 1) / metadata.chunk_size;
                for (size_t i = first + offset / metadata.chunk_size; i <= last; ++i) {
                    const auto mapping = writable_after->find(chunk_ptr(ptr, metadata, i));
                    SIMPLE_CHECK(mapping != writable_after->end(),
                                 "resume() does not map a chunk that is in the cpu_backup slot");
                    mappings = std::max(mappings, mapping->second);
                }
            }
            pieces.push_back(BackupPiece{
                run_ptr + offset,
                CpuBackupSlot{metadata.cpu_backup.kind, run_host + offset, piece_bytes},
                metadata.device, mappings});
        }
        first = end;
    }
}

// Gives the chunks that hold memory and are to be backed up their places in the cpu_backup
// slot, packed in index order: an allocation costs host memory, and a copy, for those chunks
// only.
void pack_chunk_backup(AllocationMetadata &metadata) {
    size_t total = 0;
    for (size_t i = 0; i < metadata.chunks.size(); ++i) {
        Chunk &chunk = metadata.chunks[i];
        const bool backed_up = chunk.backup && chunk.holds_memory();
        chunk.backup_offset = backed_up ? total : kNoChunkBackup;
        if (backed_up) {
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
}

void release_chunk(void *ptr, const AllocationMetadata &metadata, size_t index) {
    CURESULT_CHECK(cuMemUnmap(
        (CUdeviceptr) chunk_ptr(ptr, metadata, index), chunk_bytes(metadata, index)));
    if (metadata.chunks[index].own) {
        CURESULT_CHECK(cuMemRelease(metadata.chunks[index].handle));
    }
}

// pause() of the chunked allocations of a call, their backups packed. A mapped chunk is
// backed up through its own allocation's address, on a thread beside these calls: that
// mapping goes once the copy of the chunk has returned, every other mapping at once. The
// memory of a chunk lives until its last mapping is gone, whichever of the two allocations
// goes first.
void pause_chunked(const std::vector<std::pair<void *, AllocationMetadata *>> &allocations) {
    std::vector<BackupPiece> pieces;
    for (const auto &allocation : allocations) {
        add_backup_pieces(allocation.first, *allocation.second, pieces);
    }
    BackupCopy copy(std::move(pieces), true);
    for (const auto &allocation : allocations) {
        const AllocationMetadata &metadata = *allocation.second;
        for (size_t i = 0; i < metadata.chunks.size(); ++i) {
            const Chunk &chunk = metadata.chunks[i];
            if (chunk.holds_memory() && chunk.backup_offset == kNoChunkBackup) {
                release_chunk(allocation.first, metadata, i);
            }
        }
    }
    // The copy takes the slots in this order, and each of them from its start to its end.
    size_t copied_before = 0;
    for (const auto &allocation : allocations) {
        AllocationMetadata &metadata = *allocation.second;
        for (size_t i = 0; i < metadata.chunks.size(); ++i) {
            const size_t offset = metadata.chunks[i].backup_offset;
            if (offset != kNoChunkBackup) {
                SIMPLE_CHECK(copy.wait_for(copied_before + offset + chunk_bytes(metadata, i)),
                             "the copy of a pause() was stopped");
                release_chunk(allocation.first, metadata, i);
            }
        }
        copied_before += metadata.cpu_backup.size;
        metadata.state = AllocationState::PAUSED;
    }
    copy.join();
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

#if defined(USE_CUDA)
        // Chunks that other allocations mapped from this one lose their memory with it.
        if (metadata.chunk_size != 0) {
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
            for (auto &kv : allocation_metadata_) {
                AllocationMetadata &borrower = kv.second;
                for (size_t i = 0; i < borrower.chunks.size(); ++i) {
                    Chunk &chunk = borrower.chunks[i];
                    if (chunk.source != ptr) {
                        continue;
                    }
                    if (borrower.state == AllocationState::ACTIVE) {
                        CURESULT_CHECK(cuMemUnmap(
                            (CUdeviceptr) chunk_ptr(kv.first, borrower, i), chunk_bytes(borrower, i)));
                    }
                    chunk.source = nullptr;
                    chunk.backup_offset = kNoChunkBackup;
                }
            }
        }
#endif
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

#if defined(USE_CUDA)
    if (metadata.chunk_size != 0) {
        if (AllocationState::ACTIVE == metadata.state) {
            unmap_borrowed_chunks(ptr, metadata);
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

#if defined(USE_CUDA)
    // Memory that an active allocation of another tag maps would not return to the device,
    // and resume() would give the tag other memory than that allocation maps.
    for (const auto &kv : allocation_metadata_) {
        const AllocationMetadata &borrower = kv.second;
        if (tag.empty() || borrower.tag == tag || borrower.state != AllocationState::ACTIVE) {
            continue;
        }
        for (const Chunk &chunk : borrower.chunks) {
            if (chunk.source != nullptr && allocation_metadata_.at(chunk.source).tag == tag) {
                return kMappedChunkOrder;
            }
        }
    }
    std::vector<std::pair<void *, AllocationMetadata *>> chunked;
#endif

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
                pack_chunk_backup(metadata);
            }
            chunked.emplace_back(ptr, &metadata);
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
#if defined(USE_CUDA)
    pause_chunked(chunked);
#endif
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
    // Every handle is created before a backup is read, and every mapping is made before a
    // backup is released or a state is changed: out of memory leaves every matching allocation
    // paused with its backup, and the call can be repeated.
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
                const Chunk &chunk = metadata.chunks[i];
                if (chunk.own) {
                    pending.push_back(PendingMapping{
                        chunk_ptr(kv.first, metadata, i), chunk_bytes(metadata, i),
                        metadata.device, &metadata.chunks[i].handle});
                } else if (chunk.source != nullptr) {
                    // Mapped below from its source, which has to hold memory by then: it is
                    // active, or this call resumes it.
                    const AllocationMetadata &source = allocation_metadata_.at(chunk.source);
                    if (source.state != AllocationState::ACTIVE && !tag.empty() && source.tag != tag) {
                        return kMappedChunkOrder;
                    }
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
    // The handles of the sources are current now, those that this call made included.
    for (auto &kv : allocation_metadata_) {
        AllocationMetadata &metadata = kv.second;
        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }
        for (size_t i = 0; i < metadata.chunks.size(); ++i) {
            const Chunk &chunk = metadata.chunks[i];
            if (chunk.source != nullptr) {
                pending.push_back(PendingMapping{
                    chunk_ptr(kv.first, metadata, i), chunk_bytes(metadata, i), metadata.device,
                    &allocation_metadata_.at(chunk.source).chunks[chunk.source_index].handle});
            }
        }
    }
    // The backups of the chunked allocations are copied back on a thread beside the mapping
    // calls. A chunk of new memory is written once it is mapped: the memory goes with a call
    // that fails. A mapping of a source's memory shows memory that stays, and is written
    // once every mapping is made.
    std::unordered_map<void *, size_t> writable_after;
    writable_after.reserve(pending.size());
    for (size_t i = 0; i < pending.size(); ++i) {
        writable_after[pending[i].address] = pending[i].created ? i + 1 : pending.size();
    }
    std::vector<BackupPiece> pieces;
    for (const auto &kv : allocation_metadata_) {
        const AllocationMetadata &metadata = kv.second;
        if ((tag.empty() || metadata.tag == tag) && metadata.cpu_backup.data != nullptr
                && metadata.chunk_size != 0) {
            add_backup_pieces(kv.first, metadata, pieces, &writable_after);
        }
    }
    BackupCopy copy(std::move(pieces), false);
    pending_ret = map_all(pending, &copy);
    if (pending_ret != cudaSuccess) {
        release_created(pending);
        return pending_ret;
    }
    copy.join();
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
            if (!retain_cpu_backup) {
                cpu_backuper::release(metadata.cpu_backup);
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
        out_states[i] = metadata.state == AllocationState::ACTIVE
            && metadata.chunks[refs[i].index].holds_memory();
    }
    return refs.size();
}

int TorchMemorySaver::resume_chunks(const std::string& tag, const size_t* indices, size_t count) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const std::vector<ChunkRef> refs = chunks_of_tag(tag);
    const std::set<size_t> wanted(indices, indices + count);

    std::vector<PendingMapping> pending;
    std::vector<Chunk*> given;
    for (const size_t index : wanted) {
        if (index >= refs.size()) {
            return TMS_CHUNK_INDEX_OUT_OF_RANGE;
        }
        AllocationMetadata &metadata = *refs[index].metadata;
        if (metadata.state != AllocationState::ACTIVE) {
            return TMS_CHUNK_ALLOCATION_PAUSED;
        }
        Chunk &chunk = metadata.chunks[refs[index].index];
        if (!chunk.holds_memory()) {
            pending.push_back(PendingMapping{
                chunk_ptr(refs[index].base, metadata, refs[index].index),
                chunk_bytes(metadata, refs[index].index), metadata.device, &chunk.handle});
            given.push_back(&chunk);
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
    for (Chunk *chunk : given) {
        chunk->own = true;
    }
    return TMS_CHUNK_OK;
}

int TorchMemorySaver::map_chunks(
    const std::string& dst_tag, const size_t* dst_indices,
    const std::string& src_tag, const size_t* src_indices, size_t count) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const std::vector<ChunkRef> dst_refs = chunks_of_tag(dst_tag);
    const std::vector<ChunkRef> src_refs = chunks_of_tag(src_tag);

    std::set<size_t> destinations;
    std::vector<PendingMapping> pending;
    for (size_t i = 0; i < count; ++i) {
        if (dst_indices[i] >= dst_refs.size() || src_indices[i] >= src_refs.size()) {
            return TMS_CHUNK_INDEX_OUT_OF_RANGE;
        }
        const ChunkRef &dst = dst_refs[dst_indices[i]];
        const ChunkRef &src = src_refs[src_indices[i]];
        if (dst.metadata->state != AllocationState::ACTIVE
                || src.metadata->state != AllocationState::ACTIVE) {
            return TMS_CHUNK_ALLOCATION_PAUSED;
        }
        if (dst.metadata->chunks[dst.index].holds_memory()
                || !destinations.insert(dst_indices[i]).second) {
            return TMS_CHUNK_DESTINATION_HOLDS_MEMORY;
        }
        Chunk &src_chunk = src.metadata->chunks[src.index];
        if (!src_chunk.own) {
            return TMS_CHUNK_SOURCE_WITHOUT_MEMORY;
        }
        const size_t bytes = chunk_bytes(*dst.metadata, dst.index);
        if (bytes != chunk_bytes(*src.metadata, src.index)
                || dst.metadata->device != src.metadata->device) {
            return TMS_CHUNK_PAIR_MISMATCH;
        }
        pending.push_back(PendingMapping{
            chunk_ptr(dst.base, *dst.metadata, dst.index), bytes, dst.metadata->device,
            &src_chunk.handle});
    }
    if (map_all(pending) != cudaSuccess) {
        return TMS_CHUNK_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < count; ++i) {
        const ChunkRef &dst = dst_refs[dst_indices[i]];
        const ChunkRef &src = src_refs[src_indices[i]];
        dst.metadata->chunks[dst.index].source = src.base;
        dst.metadata->chunks[dst.index].source_index = src.index;
    }
    return TMS_CHUNK_OK;
}

int TorchMemorySaver::set_chunk_backup(
    const std::string& tag, const size_t* indices, size_t count, const bool enabled) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const std::vector<ChunkRef> refs = chunks_of_tag(tag);
    for (size_t i = 0; i < count; ++i) {
        if (indices[i] >= refs.size()) {
            return TMS_CHUNK_INDEX_OUT_OF_RANGE;
        }
        const AllocationMetadata &metadata = *refs[indices[i]].metadata;
        if (!metadata.enable_cpu_backup) {
            return TMS_CHUNK_WITHOUT_CPU_BACKUP;
        }
        // A paused allocation has its backup already: the choice is for the next pause().
        if (metadata.state != AllocationState::ACTIVE) {
            return TMS_CHUNK_ALLOCATION_PAUSED;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const ChunkRef &ref = refs[indices[i]];
        ref.metadata->chunks[ref.index].backup = enabled;
    }
    return TMS_CHUNK_OK;
}
#endif
