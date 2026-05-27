#include "nvme_kv_connector.h"

#include <cassert>
#include <utility>

namespace mooncake {

NvmeKvConnector::NvmeKvConnector(
    std::string device_id, std::unique_ptr<NvmeKvCommandExecutor> executor)
    : device_id_(std::move(device_id)), executor_(std::move(executor)) {
    assert(executor_ != nullptr &&
           "NvmeKvConnector requires a non-null executor");
}

tl::expected<void, ErrorCode> NvmeKvConnector::Flush() {
    return executor_->Flush();
}

tl::expected<void, ErrorCode> NvmeKvConnector::Store(const PhysicalKey& key,
                                                     std::string value) {
    return executor_->Store(key, std::move(value));
}

tl::expected<std::string, ErrorCode> NvmeKvConnector::Retrieve(
    const PhysicalKey& key) const {
    return executor_->Retrieve(key);
}

tl::expected<uint32_t, ErrorCode> NvmeKvConnector::RetrieveInto(
    const PhysicalKey& key, void* buffer, uint32_t buffer_size) const {
    return executor_->RetrieveInto(key, buffer, buffer_size);
}

tl::expected<bool, ErrorCode> NvmeKvConnector::Exists(
    const PhysicalKey& key) const {
    return executor_->Exists(key);
}

tl::expected<void, ErrorCode> NvmeKvConnector::Delete(const PhysicalKey& key) {
    return executor_->Delete(key);
}

tl::expected<std::vector<NvmeKvConnector::PhysicalKey>, ErrorCode>
NvmeKvConnector::List(const PhysicalKey& prefix, uint8_t prefix_len,
                      uint32_t max_keys) const {
    return executor_->List(prefix, prefix_len, max_keys);
}

const NvmeKvConnector::Capabilities& NvmeKvConnector::GetCapabilities() const {
    return executor_->GetCapabilities();
}

}  // namespace mooncake
