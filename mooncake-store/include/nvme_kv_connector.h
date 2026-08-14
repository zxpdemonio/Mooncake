#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "nvme_kv_executor.h"
#include "types.h"

namespace mooncake {

struct FileStorageConfig;

class NvmeKvConnector {
   public:
    using PhysicalKey = NvmeKvCommandExecutor::PhysicalKey;
    using Capabilities = NvmeKvCommandExecutor::Capabilities;

    explicit NvmeKvConnector(const FileStorageConfig &file_storage_config);

    tl::expected<void, ErrorCode> Init();
    tl::expected<void, ErrorCode> Store(const PhysicalKey &key,
                                        std::string value);
    void StoreBatch(std::vector<NvmeKvCommandExecutor::StoreRequest> &requests);
    tl::expected<std::string, ErrorCode> Retrieve(const PhysicalKey &key,
                                                  uint32_t size_hint = 0) const;
    void RetrieveBufferBatch(
        std::vector<NvmeKvCommandExecutor::RetrieveBufferRequest> &requests)
        const;
    void RetrieveIntoBatch(
        std::vector<NvmeKvCommandExecutor::RetrieveIntoRequest> &requests)
        const;
    tl::expected<void, ErrorCode> Delete(const PhysicalKey &key);
    tl::expected<void, ErrorCode> Iterate(
        const std::function<tl::expected<void, ErrorCode>(
            const PhysicalKey &key)> &visitor) const;
    const Capabilities &GetCapabilities() const;

   private:
    tl::expected<void, ErrorCode> InitRealExecutor();
    std::string storage_path_;
    std::unique_ptr<NvmeKvCommandExecutor> executor_;
};

}  // namespace mooncake
