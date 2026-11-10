#pragma once

#include <cstddef>
#include <cstdint>

#include "device/cuda_ipc_buffer_handle.h"

namespace mooncake {
namespace device {

tl::expected<CudaIpcBufferHandle, ErrorCode> ExportCudaIpcBuffer(
    const void *ptr, size_t size);

tl::expected<void, ErrorCode> SynchronizeCudaStream(int32_t device_id,
                                                    uintptr_t stream_handle);

class CudaIpcBufferMapping {
   public:
    CudaIpcBufferMapping() = default;
    ~CudaIpcBufferMapping();

    CudaIpcBufferMapping(const CudaIpcBufferMapping &) = delete;
    CudaIpcBufferMapping &operator=(const CudaIpcBufferMapping &) = delete;

    CudaIpcBufferMapping(CudaIpcBufferMapping &&other) noexcept;
    CudaIpcBufferMapping &operator=(CudaIpcBufferMapping &&other) noexcept;

    static tl::expected<CudaIpcBufferMapping, ErrorCode> Open(
        const CudaIpcBufferHandle &handle);
    static tl::expected<CudaIpcBufferMapping, ErrorCode> OpenAllocation(
        const CudaIpcBufferHandle &handle);

    void *ptr() const { return ptr_; }
    void *base() const { return base_; }
    size_t allocation_size() const { return allocation_size_; }
    int32_t device_id() const { return device_id_; }

   private:
    CudaIpcBufferMapping(void *base, void *ptr, int32_t device_id,
                         size_t allocation_size)
        : base_(base),
          ptr_(ptr),
          device_id_(device_id),
          allocation_size_(allocation_size) {}

    void Close();

    void *base_ = nullptr;
    void *ptr_ = nullptr;
    int32_t device_id_ = -1;
    size_t allocation_size_ = 0;
};

}  // namespace device
}  // namespace mooncake
