#include "nvme_kv_connector.h"

#include <utility>

namespace mooncake {

NvmeKvConnector::NvmeKvConnector(
    std::string device_id, std::unique_ptr<NvmeKvCommandExecutor> executor)
    : device_id_(std::move(device_id)), executor_(std::move(executor)) {}

tl::expected<void, ErrorCode> NvmeKvConnector::Flush() {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->Flush();
}

tl::expected<void, ErrorCode> NvmeKvConnector::Store(const PhysicalKey& key,
                                                     std::string value) {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->Store(key, std::move(value));
}

tl::expected<std::string, ErrorCode> NvmeKvConnector::Retrieve(
    const PhysicalKey& key) const {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->Retrieve(key);
}

tl::expected<uint32_t, ErrorCode> NvmeKvConnector::RetrieveInto(
    const PhysicalKey& key, void* buffer, uint32_t buffer_size) const {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->RetrieveInto(key, buffer, buffer_size);
}

tl::expected<bool, ErrorCode> NvmeKvConnector::Exists(
    const PhysicalKey& key) const {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->Exists(key);
}

tl::expected<void, ErrorCode> NvmeKvConnector::Delete(const PhysicalKey& key) {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->Delete(key);
}

tl::expected<std::vector<NvmeKvConnector::PhysicalKey>, ErrorCode>
NvmeKvConnector::List(const PhysicalKey& prefix, uint8_t prefix_len,
                      uint32_t max_keys) const {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->List(prefix, prefix_len, max_keys);
}

const NvmeKvConnector::Capabilities& NvmeKvConnector::GetCapabilities() const {
    static const Capabilities kDefaultCapabilities{};
    if (executor_ == nullptr) {
        return kDefaultCapabilities;
    }
    return executor_->GetCapabilities();
}

}  // namespace mooncake
