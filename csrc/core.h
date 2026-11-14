#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "utils.h"
#include "macro.h"
#include "cpu_backup.h"
#include "disk_backend.h"

#if TMS_ROCM_LEGACY_CHUNKED
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

#if defined(USE_XPU)
#include "hardware_xpu_support.h"
#endif

#if defined(USE_CUDA)
constexpr size_t kNoChunkBackup = static_cast<size_t>(-1);

// One chunk of an allocation that a region with chunk_size made.
struct Chunk {
    // The chunk has device memory of its own whenever its allocation is active. pause() and
    // resume() pass a chunk without memory by; handle is valid while the allocation is active.
    bool own = false;
    CUmemGenericAllocationHandle handle = 0;
    // Where the chunk's bytes start in the allocation's cpu_backup slot, if they are in it.
    size_t backup_offset = kNoChunkBackup;
};

// What the chunk calls return. Anything but TMS_CHUNK_OK means that nothing changed; the
// negative ones are errors of the caller.
enum TmsChunkStatus : int {
    TMS_CHUNK_OK = 0,
    TMS_CHUNK_OUT_OF_MEMORY = 2,  // cudaErrorMemoryAllocation
    TMS_CHUNK_INDEX_OUT_OF_RANGE = -1,
    TMS_CHUNK_ALLOCATION_PAUSED = -2,
};
#endif

struct AllocationMetadata {
    size_t raw_size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    CpuBackupSlot cpu_backup;
    bool enable_disk_backup;
    DiskBackupSlot disk;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#elif defined(USE_XPU)
    size_t aligned_size;
    XPUAllocExtra xpu;
#else
    // CUDA, ROCm 7.0+ and MUSA: Single allocation handle
    size_t allocation_size;
    CUmemGenericAllocationHandle allocHandle;
    // Born paused by an initially_released region and not resumed yet: there is no backup.
    bool born_released = false;
#if defined(USE_CUDA)
    // An allocation of a region with chunk_size: chunk i covers
    // [i * chunk_size, min((i + 1) * chunk_size, allocation_size)), allocHandle is unused and
    // cpu_backup packs the chunks that held memory at pause() in index order. chunk_size is 0
    // and chunks is empty for every other allocation.
    size_t chunk_size = 0;
    std::vector<Chunk> chunks = {};
    // Order of allocation: numbers the chunks of a tag across its allocations.
    uint64_t sequence = 0;
#endif
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(
        void** ptr,
        CUdevice device,
        size_t raw_size,
        const std::string& tag,
        bool enable_cpu_backup,
        CpuBackupKind cpu_backup_kind,
        bool enable_disk_backup,
        bool initially_released = false,
        size_t chunk_size = 0);
    cudaError_t free(void *ptr);

#if defined(USE_CUDA)
    // The chunk calls address the chunked allocations of one tag. Its chunks are numbered
    // across those allocations in the order they were made.
    // Fills up to `capacity` entries and returns the number of chunks. A state is 1 for a
    // chunk that holds memory now.
    size_t get_chunk_layout(
        const std::string& tag, uint64_t* out_addresses, size_t* out_sizes, uint8_t* out_states,
        size_t capacity);
    // Gives memory to the listed chunks that have none, all or nothing. Returns a TmsChunkStatus.
    int resume_chunks(const std::string& tag, const size_t* indices, size_t count);
#endif

    cudaError_t pause(const std::string& tag);
    cudaError_t resume(const std::string& tag);
    uint32_t affected_devices(const char* tag, int* out_device_ids, uint32_t capacity);
    void set_memory_margin_bytes(uint64_t value) {
#if defined(USE_XPU)
        if (value != 0) {
            std::cerr << "[torch_memory_saver.cpp] set_memory_margin_bytes("
                      << value << ") ignored: NOT supported on Intel XPU "
                         "(OOM-margin guard needs device free-bytes the driver "
                         "reports frozen). Manage headroom outside torch_memory_saver."
                      << std::endl;
        }
        (void)value;
#else
        memory_margin_bytes_.store(value);
#endif
    }
    void set_retain_cpu_backup(bool value) {
        retain_cpu_backup_.store(value);
    }
    bool get_retain_cpu_backup() const {
        return retain_cpu_backup_.load();
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);
    void set_disk_backup_dir(const std::string& dir) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        disk_backend_.set_dir(dir);
    }

#if defined(USE_XPU)
    uint64_t xpu_committed_bytes(int device_id);
    uint64_t xpu_leaked_bytes(int device_id);
    uint64_t xpu_tracked_bytes(int device_id);
#endif

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

#if defined(USE_CUDA)
    struct ChunkRef {
        void* base;
        AllocationMetadata* metadata;
        size_t index;
    };
    cudaError_t malloc_chunked(void** ptr, AllocationMetadata metadata, bool initially_released);
    // Every chunk of the tag, in the order of the chunk calls. The caller holds
    // allocator_metadata_mutex_.
    std::vector<ChunkRef> chunks_of_tag(const std::string& tag);
    // Guarded by allocator_metadata_mutex_.
    uint64_t next_sequence_ = 0;
#endif

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;
    std::atomic<bool> retain_cpu_backup_ = false;

    // Guarded by allocator_metadata_mutex_.
    DiskBackend disk_backend_;
};
