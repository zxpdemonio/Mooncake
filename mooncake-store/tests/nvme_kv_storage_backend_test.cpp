#include "nvme_kv_backend.h"
#include "nvme_kv_executor_util.h"
#include "nvme_kv_object_layout.h"
#include "storage_backend.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

namespace mooncake::test {

namespace {

class EnvVarGuard {
   public:
    EnvVarGuard(const char *name, const char *value) : name_(name) {
        if (const char *old_value = getenv(name)) {
            old_value_ = old_value;
        }
        setenv(name, value, 1);
    }

    ~EnvVarGuard() {
        if (old_value_.has_value()) {
            setenv(name_.c_str(), old_value_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

   private:
    std::string name_;
    std::optional<std::string> old_value_;
};

class NvmeKvStorageBackendTest : public ::testing::Test {
   protected:
    void SetUp() override {
        data_path_ = (fs::current_path() / "nvme_kv_test_data").string();
        std::error_code ec;
        fs::remove_all(data_path_, ec);
        ASSERT_FALSE(ec) << ec.message();
        fs::create_directories(data_path_, ec);
        ASSERT_FALSE(ec) << ec.message();
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(data_path_, ec);
    }

    std::string data_path_;
};

}  // namespace

TEST_F(NvmeKvStorageBackendTest, PhysicalKeyPackingEncodesCommandSet) {
    NvmeKvCommandExecutor::PhysicalKey key{};
    for (size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<uint8_t>(0x10 + i);
    }

    const auto fields = PackNvmeKvPhysicalKey(key);
    uint32_t expected_cdw14 = 0;
    std::memcpy(&expected_cdw14, key.data() + 8, sizeof(expected_cdw14));
    expected_cdw14 &= 0x00FFFFFFu;
    expected_cdw14 |= static_cast<uint32_t>(kNvmeKvCommandSetIdentifier) << 24;

    EXPECT_EQ(fields.cdw14, expected_cdw14);
    EXPECT_EQ(static_cast<uint8_t>(fields.cdw14 >> 24),
              kNvmeKvCommandSetIdentifier);
}

TEST_F(NvmeKvStorageBackendTest, StatusMappingHandlesKvSpecificStatusCodes) {
    EXPECT_EQ(MapNvmeKvStatus(0x81, true), ErrorCode::KEYS_ULTRA_LIMIT);
    EXPECT_EQ(MapNvmeKvStatus(0x85, true), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(MapNvmeKvStatus(0x86, false), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(MapNvmeKvStatus(0x87, false), ErrorCode::OBJECT_NOT_FOUND);
    EXPECT_EQ(MapNvmeKvStatus(0x89, true), ErrorCode::OBJECT_ALREADY_EXISTS);

    EXPECT_EQ(MapNvmeKvTransportError(ENOENT, false),
              ErrorCode::OBJECT_NOT_FOUND);
    EXPECT_EQ(MapNvmeKvTransportError(ENOSPC, true),
              ErrorCode::KEYS_ULTRA_LIMIT);
    EXPECT_EQ(MapNvmeKvTransportError(ENOMEM, false),
              ErrorCode::BUFFER_OVERFLOW);
    EXPECT_EQ(MapNvmeKvTransportError(0x4087, false),
              ErrorCode::OBJECT_NOT_FOUND);

    constexpr uint32_t kRawValueSize = 1234;
    std::array<char, 4096> raw_value{};
    EXPECT_EQ(
        ResolveNvmeKvInitialRetrieveBytes(kRawValueSize, raw_value.size()),
        RoundUpToNvmeKvTransferBytes(kRawValueSize));
    EXPECT_EQ(ResolveNvmeKvRetrievedValueSize(raw_value.data(), 0,
                                              raw_value.size(), kRawValueSize),
              kRawValueSize);
    EXPECT_EQ(ResolveNvmeKvRetrievedValueSize(raw_value.data(), 0, 1024,
                                              kRawValueSize),
              0);

    EXPECT_TRUE(ShouldRetryNvmeKvRetrieveWithMaxBuffer(
        ErrorCode::INVALID_PARAMS, 0, kDefaultNvmeKvTransferAlignmentBytes,
        kDefaultNvmeKvRuntimeTransferLimit));
    EXPECT_FALSE(ShouldRetryNvmeKvRetrieveWithMaxBuffer(
        ErrorCode::INVALID_PARAMS, kDefaultNvmeKvTransferAlignmentBytes,
        kDefaultNvmeKvTransferAlignmentBytes,
        kDefaultNvmeKvRuntimeTransferLimit));
    EXPECT_FALSE(ShouldRetryNvmeKvRetrieveWithMaxBuffer(
        ErrorCode::OBJECT_NOT_FOUND, 0, kDefaultNvmeKvTransferAlignmentBytes,
        kDefaultNvmeKvRuntimeTransferLimit));
    EXPECT_FALSE(ShouldRetryNvmeKvRetrieveWithMaxBuffer(
        ErrorCode::INTERNAL_ERROR, 0, kDefaultNvmeKvTransferAlignmentBytes,
        kDefaultNvmeKvRuntimeTransferLimit));
    EXPECT_FALSE(ShouldRetryNvmeKvRetrieveWithMaxBuffer(
        ErrorCode::INVALID_PARAMS, 0, kDefaultNvmeKvRuntimeTransferLimit,
        kDefaultNvmeKvRuntimeTransferLimit));
}

TEST_F(NvmeKvStorageBackendTest, ListParserRequiresPhysicalKeyLength) {
    std::array<char, 64> buffer{};
    auto *bytes = reinterpret_cast<uint8_t *>(buffer.data());
    bytes[0] = 1;
    bytes[4] = 15;
    bytes[5] = 0;

    auto short_key = ParseNvmeKvListResponse(buffer.data(), buffer.size());
    ASSERT_FALSE(short_key.has_value());
    EXPECT_EQ(short_key.error(), ErrorCode::INVALID_PARAMS);

    std::memset(buffer.data(), 0, buffer.size());
    bytes[0] = 1;
    bytes[4] = 16;
    bytes[5] = 0;
    for (uint8_t i = 0; i < 16; ++i) {
        bytes[6 + i] = i;
    }

    auto valid_key = ParseNvmeKvListResponse(buffer.data(), buffer.size());
    ASSERT_TRUE(valid_key.has_value());
    ASSERT_EQ(valid_key->size(), 1);
    for (uint8_t i = 0; i < 16; ++i) {
        EXPECT_EQ(valid_key->front()[i], i);
    }
}

TEST_F(NvmeKvStorageBackendTest, BackendLoadsKnownObjectsFromDevice) {
    EnvVarGuard driver_guard("MOONCAKE_NVME_KV_DRIVER", "stub");

    FileStorageConfig config;
    config.storage_filepath = data_path_ + "/known_objects";
    config.storage_backend_type = StorageBackendType::kNvmeKv;

    constexpr int kObjectCount = 9;
    std::vector<std::string> keys;
    std::vector<std::string> values;
    keys.reserve(kObjectCount);
    values.reserve(kObjectCount);
    std::unordered_map<std::string, std::vector<Slice>> batch;
    for (int i = 0; i < kObjectCount; ++i) {
        keys.emplace_back("nvme_kv_known_key_" + std::to_string(i));
        const size_t value_size = i == 0 ? 4096 : 128 * 1024 + 4096 + i * 17;
        values.emplace_back(value_size, static_cast<char>('a' + (i % 26)));
        batch.emplace(keys.back(),
                      std::vector<Slice>{
                          Slice{values.back().data(), values.back().size()}});
    }

    {
        NvmeKvStorageBackend backend(config);
        ASSERT_TRUE(backend.Init().has_value());
        auto offload_result = backend.BatchOffload(
            batch,
            [](const std::vector<std::string> &,
               std::vector<StorageObjectMetadata> &) { return ErrorCode::OK; });
        ASSERT_TRUE(offload_result.has_value());
        ASSERT_EQ(offload_result.value(), kObjectCount);
    }

    NvmeKvStorageBackend reader(config);
    ASSERT_TRUE(reader.Init().has_value());
    for (const auto &key : keys) {
        EXPECT_TRUE(reader.IsExist(key).value_or(false));
    }
    EXPECT_FALSE(reader.IsExist("nvme_kv_missing_key").value_or(true));

    auto duplicate_result = reader.BatchOffload(
        batch,
        [](const std::vector<std::string> &,
           std::vector<StorageObjectMetadata> &) { return ErrorCode::OK; });
    ASSERT_TRUE(duplicate_result.has_value());
    EXPECT_EQ(duplicate_result.value(), kObjectCount);

    std::vector<std::string> loaded_values;
    loaded_values.reserve(kObjectCount);
    std::unordered_map<std::string, Slice> load_batch;
    for (int i = 0; i < kObjectCount; ++i) {
        loaded_values.emplace_back(values[i].size(), '\0');
        load_batch.emplace(keys[i], Slice{loaded_values.back().data(),
                                          loaded_values.back().size()});
    }
    ASSERT_TRUE(reader.BatchLoad(load_batch).has_value());
    EXPECT_EQ(loaded_values, values);
}

TEST_F(NvmeKvStorageBackendTest, ScanMetaRebuildsChunkedObjectsFromDevice) {
    EnvVarGuard driver_guard("MOONCAKE_NVME_KV_DRIVER", "stub");
    EnvVarGuard runtime_limit_guard("MOONCAKE_NVME_KV_RUNTIME_TRANSFER_LIMIT",
                                    "65536");
    EnvVarGuard protocol_limit_guard("MOONCAKE_NVME_KV_PROTOCOL_MAX_VALUE_SIZE",
                                     "65536");

    FileStorageConfig config;
    config.storage_filepath = data_path_ + "/scanmeta_rebuild";
    config.storage_backend_type = StorageBackendType::kNvmeKv;
    config.scanmeta_iterator_keys_limit = 1;

    const std::string inline_key = "nvme_kv_scanmeta_inline";
    const std::string chunked_key = "nvme_kv_scanmeta_chunked";
    std::string inline_value(4096, 'i');
    std::string chunk_like_value(4096, 'x');
    NvmeKvObjectHeader fake_header{};
    fake_header.magic = NvmeKvObjectHeader::kMagic;
    fake_header.object_type = static_cast<uint32_t>(NvmeKvObjectType::kInline);
    fake_header.identity_metadata_size = 64;
    fake_header.payload_size = 128 * 1024;
    fake_header.header_checksum = ComputeNvmeKvHeaderChecksum(fake_header);
    std::memcpy(chunk_like_value.data(), &fake_header, sizeof(fake_header));
    std::string chunked_value(180 * 1024 + 123, 'c');
    std::memcpy(chunked_value.data(), chunk_like_value.data(),
                chunk_like_value.size());
    ASSERT_EQ(ResolveNvmeKvObjectBlobSizeFromPrefix(chunked_value.data(),
                                                    sizeof(NvmeKvObjectHeader)),
              sizeof(NvmeKvObjectHeader) + fake_header.identity_metadata_size +
                  fake_header.payload_size);

    {
        NvmeKvStorageBackend backend(config);
        ASSERT_TRUE(backend.Init().has_value());

        std::unordered_map<std::string, std::vector<Slice>> batch;
        batch.emplace(inline_key,
                      std::vector<Slice>{
                          Slice{inline_value.data(), inline_value.size()}});
        batch.emplace(chunked_key,
                      std::vector<Slice>{
                          Slice{chunked_value.data(), chunked_value.size()}});

        auto offload_result = backend.BatchOffload(
            batch,
            [](const std::vector<std::string> &,
               std::vector<StorageObjectMetadata> &) { return ErrorCode::OK; });
        ASSERT_TRUE(offload_result.has_value());
        EXPECT_EQ(offload_result.value(), 2);
    }

    NvmeKvStorageBackend rebuilt(config);
    ASSERT_TRUE(rebuilt.Init().has_value());

    std::unordered_map<std::string, StorageObjectMetadata> scanned;
    size_t scanned_callback_count = 0;
    bool saw_duplicate_key = false;
    auto scan_result =
        rebuilt.ScanMeta([&](const std::vector<std::string> &keys,
                             std::vector<StorageObjectMetadata> &metadatas) {
            EXPECT_EQ(keys.size(), metadatas.size());
            for (size_t i = 0; i < keys.size(); ++i) {
                ++scanned_callback_count;
                saw_duplicate_key |=
                    !scanned.emplace(keys[i], metadatas[i]).second;
            }
            return ErrorCode::OK;
        });
    ASSERT_TRUE(scan_result.has_value());
    EXPECT_FALSE(saw_duplicate_key);
    EXPECT_EQ(scanned_callback_count, 2);
    ASSERT_EQ(scanned.size(), 2);
    auto inline_it = scanned.find(inline_key);
    auto chunked_it = scanned.find(chunked_key);
    ASSERT_NE(inline_it, scanned.end());
    ASSERT_NE(chunked_it, scanned.end());
    EXPECT_EQ(inline_it->second.data_size,
              static_cast<int64_t>(inline_value.size()));
    EXPECT_EQ(chunked_it->second.data_size,
              static_cast<int64_t>(chunked_value.size()));

    std::string loaded_inline(inline_value.size(), '\0');
    std::string loaded_chunked(chunked_value.size(), '\0');
    std::unordered_map<std::string, Slice> load_batch{
        {inline_key, Slice{loaded_inline.data(), loaded_inline.size()}},
        {chunked_key, Slice{loaded_chunked.data(), loaded_chunked.size()}}};
    ASSERT_TRUE(rebuilt.BatchLoad(load_batch).has_value());
    EXPECT_EQ(loaded_inline, inline_value);
    EXPECT_EQ(loaded_chunked, chunked_value);
}

TEST_F(NvmeKvStorageBackendTest, RealDeviceScanMetaRoundTrip) {
    const char *run_real = std::getenv("MOONCAKE_NVME_KV_RUN_REAL_DEVICE_TEST");
    if (run_real == nullptr || std::string(run_real) != "1") {
        GTEST_SKIP() << "set MOONCAKE_NVME_KV_RUN_REAL_DEVICE_TEST=1";
    }
    const char *device_path = std::getenv("MOONCAKE_NVME_KV_REAL_DEVICE_PATH");
    if (device_path == nullptr || std::string(device_path).empty()) {
        GTEST_SKIP() << "set MOONCAKE_NVME_KV_REAL_DEVICE_PATH";
    }
    const char *clean_device =
        std::getenv("MOONCAKE_NVME_KV_REAL_DEVICE_CLEAN");
    if (clean_device == nullptr || std::string(clean_device) != "1") {
        GTEST_SKIP() << "set MOONCAKE_NVME_KV_REAL_DEVICE_CLEAN=1";
    }

    EnvVarGuard driver_guard("MOONCAKE_NVME_KV_DRIVER", "");
    EnvVarGuard device_guard("MOONCAKE_NVME_KV_DEVICE_PATH", device_path);
    EnvVarGuard transport_guard(
        "MOONCAKE_NVME_KV_TRANSPORT",
        std::getenv("MOONCAKE_NVME_KV_REAL_TRANSPORT") == nullptr
            ? "io_uring"
            : std::getenv("MOONCAKE_NVME_KV_REAL_TRANSPORT"));

    FileStorageConfig config;
    config.storage_filepath = data_path_ + "/real_device_scanmeta";
    config.storage_backend_type = StorageBackendType::kNvmeKv;
    config.scanmeta_iterator_keys_limit = 1;

    auto clean_namespace = [&]() {
        NvmeKvConnector connector(config);
        ASSERT_TRUE(connector.Init().has_value());
        std::vector<NvmeKvCommandExecutor::PhysicalKey> keys;
        auto iterate_result =
            connector.Iterate([&](const NvmeKvCommandExecutor::PhysicalKey &key)
                                  -> tl::expected<void, ErrorCode> {
                keys.push_back(key);
                return {};
            });
        ASSERT_TRUE(iterate_result.has_value());
        for (const auto &key : keys) {
            auto delete_result = connector.Delete(key);
            ASSERT_TRUE(delete_result.has_value());
        }
    };
    clean_namespace();

    const auto unique_suffix =
        std::to_string(static_cast<long long>(
            std::chrono::steady_clock::now().time_since_epoch().count())) +
        "_" + std::to_string(static_cast<long long>(getpid()));
    size_t extra_inline_count = 0;
    if (const char *count_env =
            std::getenv("MOONCAKE_NVME_KV_REAL_EXTRA_INLINE_COUNT")) {
        extra_inline_count = static_cast<size_t>(std::stoull(count_env));
    }
    const std::string inline_key =
        "nvme_kv_real_scanmeta_inline_" + unique_suffix;
    const std::string chunked_key =
        "nvme_kv_real_scanmeta_chunked_" + unique_suffix;
    std::string inline_value(4096, 'r');
    std::string chunked_value(600 * 1024 + 123, 'R');
    std::unordered_map<std::string, std::string> expected_values;
    expected_values.emplace(inline_key, inline_value);
    expected_values.emplace(chunked_key, chunked_value);
    for (size_t i = 0; i < extra_inline_count; ++i) {
        expected_values.emplace(
            "nvme_kv_real_scanmeta_extra_" + unique_suffix + "_" +
                std::to_string(i),
            std::string(4096 + (i % 7) * 512, static_cast<char>('a' + i % 26)));
    }

    {
        NvmeKvStorageBackend backend(config);
        ASSERT_TRUE(backend.Init().has_value());
        std::unordered_map<std::string, std::vector<Slice>> batch;
        for (auto &[key, value] : expected_values) {
            batch.emplace(
                key, std::vector<Slice>{Slice{value.data(), value.size()}});
        }
        auto offload_result = backend.BatchOffload(
            batch,
            [](const std::vector<std::string> &,
               std::vector<StorageObjectMetadata> &) { return ErrorCode::OK; });
        ASSERT_TRUE(offload_result.has_value());
        ASSERT_EQ(offload_result.value(),
                  static_cast<int64_t>(expected_values.size()));
    }

    NvmeKvStorageBackend rebuilt(config);
    ASSERT_TRUE(rebuilt.Init().has_value());
    std::unordered_map<std::string, StorageObjectMetadata> scanned;
    size_t scanned_callback_count = 0;
    bool saw_duplicate_key = false;
    auto scan_result =
        rebuilt.ScanMeta([&](const std::vector<std::string> &keys,
                             std::vector<StorageObjectMetadata> &metadatas) {
            EXPECT_LE(keys.size(), 1);
            EXPECT_EQ(keys.size(), metadatas.size());
            for (size_t i = 0; i < keys.size(); ++i) {
                ++scanned_callback_count;
                saw_duplicate_key |=
                    !scanned.emplace(keys[i], metadatas[i]).second;
            }
            return ErrorCode::OK;
        });
    ASSERT_TRUE(scan_result.has_value());
    EXPECT_FALSE(saw_duplicate_key);
    EXPECT_EQ(scanned_callback_count, expected_values.size());
    ASSERT_EQ(scanned.size(), expected_values.size());
    for (const auto &[key, value] : expected_values) {
        ASSERT_EQ(scanned.at(key).data_size,
                  static_cast<int64_t>(value.size()));
    }

    std::vector<std::string> loaded_values;
    loaded_values.reserve(expected_values.size());
    std::unordered_map<std::string, Slice> load_batch;
    for (const auto &[key, value] : expected_values) {
        loaded_values.emplace_back(value.size(), '\0');
        load_batch.emplace(key, Slice{loaded_values.back().data(),
                                      loaded_values.back().size()});
    }
    ASSERT_TRUE(rebuilt.BatchLoad(load_batch).has_value());
    size_t value_index = 0;
    for (const auto &[_, value] : expected_values) {
        EXPECT_EQ(loaded_values[value_index++], value);
    }

    clean_namespace();
}

TEST_F(NvmeKvStorageBackendTest, PipelinedStubRoundTripChunkedObjects) {
    EnvVarGuard driver_guard("MOONCAKE_NVME_KV_DRIVER", "stub");
    EnvVarGuard queue_depth_guard("MOONCAKE_NVME_KV_QUEUE_DEPTH", "8");
    EnvVarGuard runtime_limit_guard("MOONCAKE_NVME_KV_RUNTIME_TRANSFER_LIMIT",
                                    "65536");
    EnvVarGuard io_concurrency_guard("MOONCAKE_NVME_KV_IO_CONCURRENCY", "8");
    EnvVarGuard batch_submit_guard("MOONCAKE_NVME_KV_BATCH_SUBMIT_CONCURRENCY",
                                   "3");
    EnvVarGuard root_submit_guard("MOONCAKE_NVME_KV_ROOT_SUBMIT_CONCURRENCY",
                                  "2");

    FileStorageConfig config;
    config.storage_filepath = data_path_ + "/pipelined";
    config.storage_backend_type = StorageBackendType::kNvmeKv;

    NvmeKvStorageBackend backend(config);
    ASSERT_TRUE(backend.Init().has_value());

    constexpr int kObjectCount = 24;
    std::vector<std::string> keys;
    std::vector<std::string> values;
    keys.reserve(kObjectCount);
    values.reserve(kObjectCount);
    std::unordered_map<std::string, std::vector<Slice>> batch;
    for (int i = 0; i < kObjectCount; ++i) {
        keys.emplace_back("nvme_kv_pipelined_key_" + std::to_string(i));
        values.emplace_back(180 * 1024 + i * 4096,
                            static_cast<char>('A' + (i % 26)));
        batch.emplace(keys.back(),
                      std::vector<Slice>{
                          Slice{values.back().data(), values.back().size()}});
    }

    auto offload_result = backend.BatchOffload(
        batch,
        [](const std::vector<std::string> &,
           std::vector<StorageObjectMetadata> &) { return ErrorCode::OK; });
    ASSERT_TRUE(offload_result.has_value());
    EXPECT_EQ(offload_result.value(), kObjectCount);

    std::vector<std::string> loaded_values;
    loaded_values.reserve(kObjectCount);
    std::unordered_map<std::string, Slice> load_batch;
    for (int i = 0; i < kObjectCount; ++i) {
        loaded_values.emplace_back(values[i].size(), '\0');
        load_batch.emplace(keys[i], Slice{loaded_values.back().data(),
                                          loaded_values.back().size()});
    }
    ASSERT_TRUE(backend.BatchLoad(load_batch).has_value());
    EXPECT_EQ(loaded_values, values);
}

TEST_F(NvmeKvStorageBackendTest, ManifestCacheUpdatesAfterSameKeyRewrite) {
    EnvVarGuard driver_guard("MOONCAKE_NVME_KV_DRIVER", "stub");
    EnvVarGuard runtime_limit_guard("MOONCAKE_NVME_KV_RUNTIME_TRANSFER_LIMIT",
                                    "65536");

    FileStorageConfig config;
    config.storage_filepath = data_path_ + "/manifest_cache";
    config.storage_backend_type = StorageBackendType::kNvmeKv;

    NvmeKvStorageBackend backend(config);
    ASSERT_TRUE(backend.Init().has_value());

    const std::string key = "nvme_kv_manifest_cache_rewrite";
    std::string first_value(192 * 1024, 'x');
    std::string second_value(first_value.size(), 'y');

    auto store_value = [&](std::string &value) {
        std::unordered_map<std::string, std::vector<Slice>> batch;
        batch.emplace(key,
                      std::vector<Slice>{Slice{value.data(), value.size()}});
        auto offload_result = backend.BatchOffload(
            batch,
            [](const std::vector<std::string> &,
               std::vector<StorageObjectMetadata> &) { return ErrorCode::OK; });
        ASSERT_TRUE(offload_result.has_value());
        EXPECT_EQ(offload_result.value(), 1);
    };

    store_value(first_value);
    std::string loaded(first_value.size(), '\0');
    std::unordered_map<std::string, Slice> load_batch{
        {key, Slice{loaded.data(), loaded.size()}}};
    ASSERT_TRUE(backend.BatchLoad(load_batch).has_value());
    EXPECT_EQ(loaded, first_value);

    store_value(second_value);
    std::fill(loaded.begin(), loaded.end(), '\0');
    ASSERT_TRUE(backend.BatchLoad(load_batch).has_value());
    EXPECT_EQ(loaded, second_value);
}

}  // namespace mooncake::test
