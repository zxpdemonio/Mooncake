#include "nvme_kv_executor.h"

#include "nvme_kv_key_codec.h"
#include "nvme_kv_object_layout.h"

#include <fcntl.h>
#include <linux/nvme_ioctl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

namespace mooncake {
namespace {

// Used only when Identify capability probing is unavailable. Normal operation
// uses the value-size limit reported by the namespace's NVMe KV format.
constexpr uint32_t kConservativeValueSizeFallback = 128 * 1024;
constexpr uint32_t kNvmeIoctlTimeoutMs = 30000;
constexpr uint32_t kMooncakePhysicalKeySize = 16;
constexpr uint32_t kNvmeIdentifyDataSize = 4096;
// Default KVCG fallback (dword = 4 bytes per NVMe KV spec).
// Actual value should come from Capabilities.kvcg probed at init time.
constexpr uint32_t kDefaultKvcgBytes = 4;
// NVMe KV command-specific status codes (SC field, SCT=0).
// The ioctl passthrough returns the full 15-bit status word from the CQE:
//   [14]=DNR [13]=More [12:11]=CRD [10:8]=SCT [7:0]=SC
// We mask to (SCT << 8 | SC) for matching, ignoring DNR/More/CRD bits.
constexpr uint32_t kNvmeStatusMask = 0x7FFu;          // SCT[2:0] + SC[7:0]
constexpr uint32_t kNvmeScKvKeyNotExists = 0x087u;    // SCT=0, SC=0x87
constexpr uint32_t kNvmeScKvKeyExists = 0x089u;       // SCT=0, SC=0x89
constexpr uint32_t kNvmeScInvalidValueSize = 0x085u;  // SCT=0, SC=0x85
constexpr uint32_t kNvmeScInvalidKeySize = 0x086u;    // SCT=0, SC=0x86
constexpr uint32_t kNvmeScCapExceeded = 0x081u;       // SCT=0, SC=0x81
constexpr size_t kNvmeDmaAlignment = 4096;

// NVMe command-set constants used by the Linux passthrough ioctl path.
constexpr uint8_t kNvmeAdminIdentifyOpcode = 0x06;
constexpr uint8_t kNvmeIdentifyCsiNamespace = 0x05;
constexpr uint8_t kNvmeKvFlushOpcode = 0x00;
constexpr uint8_t kNvmeKvStoreOpcode = 0x01;
constexpr uint8_t kNvmeKvRetrieveOpcode = 0x02;
constexpr uint8_t kNvmeKvListOpcode = 0x06;
constexpr uint8_t kNvmeKvDeleteOpcode = 0x10;
constexpr uint8_t kNvmeKvExistOpcode = 0x14;
constexpr uint8_t kNvmeKvCommandSetIndicator = 0x01;
constexpr uint32_t kCdw11KeyLengthMask = 0xFFu;

uint32_t BuildKeyLengthField(size_t key_length) {
    return static_cast<uint32_t>(key_length) & kCdw11KeyLengthMask;
}

uint16_t ReadLe16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) |
           (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t ReadLe32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

struct FreeDeleter {
    void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
using AlignedUniquePtr = std::unique_ptr<T, FreeDeleter>;

AlignedUniquePtr<char> AllocateAlignedBuffer(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, kNvmeDmaAlignment, size) != 0) {
        return AlignedUniquePtr<char>(nullptr);
    }
    return AlignedUniquePtr<char>(static_cast<char*>(ptr));
}

uint32_t RoundUpToKvTransferBytes(uint32_t bytes, uint32_t kvcg) {
    if (bytes == 0) {
        return 0;
    }
    const uint64_t rounded =
        ((static_cast<uint64_t>(bytes) + kvcg - 1u) / kvcg) * kvcg;
    return static_cast<uint32_t>(
        std::min(rounded, static_cast<uint64_t>(UINT32_MAX)));
}

tl::expected<uint32_t, ErrorCode> ComputeKvBlockCountMinusOne(uint32_t bytes,
                                                              uint32_t kvcg) {
    if (bytes == 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(bytes) + kvcg - 1u) / kvcg - 1u);
}

bool ShouldTraceCommands() {
    static const bool enabled = [] {
        const char* value = std::getenv("MOONCAKE_NVME_KV_TRACE_COMMANDS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void EncodeKeyIntoCommand(const NvmeKvCommandExecutor::PhysicalKey& key,
                          nvme_passthru_cmd64& cmd) {
    uint64_t key_buf[2] = {0, 0};
    std::memcpy(key_buf, key.data(), key.size());
    cmd.cdw2 = static_cast<uint32_t>(key_buf[0]);
    cmd.cdw3 = static_cast<uint32_t>(key_buf[0] >> 32);
    cmd.cdw14 = static_cast<uint32_t>(key_buf[1]);
    cmd.cdw15 = static_cast<uint32_t>(key_buf[1] >> 32);
}

void TraceCommandBuild(const char* op_name,
                       const NvmeKvCommandExecutor::PhysicalKey& key,
                       const nvme_passthru_cmd64& cmd) {
    if (!ShouldTraceCommands()) {
        return;
    }
    LOG(INFO) << "[NVME-KV-TRACE] submit"
              << " transport=ioctl"
              << " op=" << op_name << " key=" << NvmeKvPhysicalKeyToHex(key)
              << " opcode=0x" << std::hex << static_cast<uint32_t>(cmd.opcode)
              << " nsid=" << std::dec << cmd.nsid
              << " data_len=" << cmd.data_len
              << " timeout_ms=" << cmd.timeout_ms << " cdw2=0x" << std::hex
              << cmd.cdw2 << " cdw3=0x" << cmd.cdw3 << " cdw10=0x" << cmd.cdw10
              << " cdw11=0x" << cmd.cdw11 << " cdw12=0x" << cmd.cdw12
              << " cdw13=0x" << cmd.cdw13 << " cdw14=0x" << cmd.cdw14
              << " cdw15=0x" << cmd.cdw15;
}

void TraceCommandResult(const char* op_name,
                        const NvmeKvCommandExecutor::PhysicalKey& key,
                        const nvme_passthru_cmd64& cmd, int raw_res,
                        ErrorCode mapped_error, bool success,
                        uint64_t result_word0) {
    if (!ShouldTraceCommands()) {
        return;
    }
    LOG(INFO) << "[NVME-KV-TRACE] complete"
              << " transport=ioctl"
              << " op=" << op_name << " key=" << NvmeKvPhysicalKeyToHex(key)
              << " success=" << (success ? "true" : "false")
              << " raw_res=" << raw_res
              << " mapped_error=" << static_cast<int>(mapped_error)
              << " result=0x" << std::hex << result_word0 << " cdw2=0x"
              << cmd.cdw2 << " cdw3=0x" << cmd.cdw3 << " cdw10=0x" << cmd.cdw10
              << " cdw11=0x" << cmd.cdw11 << " cdw12=0x" << cmd.cdw12
              << " cdw13=0x" << cmd.cdw13 << " cdw14=0x" << cmd.cdw14
              << " cdw15=0x" << cmd.cdw15;
}

std::optional<NvmeKvCommandExecutor::Capabilities> ParseKvIdentifyNamespace(
    const std::vector<uint8_t>& data) {
    // Offsets are from the NVMe KV command set's I/O Command Set specific
    // Identify Namespace data structure: KVFC selects the active KV format,
    // and KVF[] stores fixed-size KV format descriptors.
    constexpr size_t kNamespaceKvfcOffset = 28;
    constexpr size_t kFormatTableOffset = 72;
    constexpr size_t kFormatSize = 16;
    constexpr size_t kMaxFormatCount = 16;

    if (data.size() < kNvmeIdentifyDataSize) {
        return std::nullopt;
    }

    const uint8_t active_format = data[kNamespaceKvfcOffset] & 0x0F;
    if (active_format >= kMaxFormatCount) {
        return std::nullopt;
    }

    const size_t format_offset =
        kFormatTableOffset + active_format * kFormatSize;
    const uint32_t max_key_size = ReadLe16(data.data() + format_offset);
    const uint32_t max_value_size = ReadLe32(data.data() + format_offset + 4);
    if (max_key_size == 0 || max_value_size == 0) {
        return std::nullopt;
    }

    // KV Format Descriptor offset +8: KVCG (Key Value Command Granularity)
    // in bytes. 0 means device uses dword (4-byte) granularity.
    const uint32_t kvcg_raw = ReadLe32(data.data() + format_offset + 8);

    NvmeKvCommandExecutor::Capabilities caps;
    caps.max_key_size = max_key_size;
    caps.max_value_size = max_value_size;
    caps.effective_max_value_size = max_value_size;
    caps.kvcg = (kvcg_raw > 0) ? kvcg_raw : kDefaultKvcgBytes;
    caps.probed = true;
    return caps;
}

ErrorCode MapErrno(int err_no, bool is_write) {
    switch (err_no) {
        case ENOENT:
            return ErrorCode::OBJECT_NOT_FOUND;
        case ENOSPC:
            return ErrorCode::KEYS_ULTRA_LIMIT;
        case EINVAL:
            return ErrorCode::INVALID_PARAMS;
        case ENOMEM:
            return ErrorCode::BUFFER_OVERFLOW;
        default:
            return is_write ? ErrorCode::FILE_WRITE_FAIL
                            : ErrorCode::FILE_READ_FAIL;
    }
}

ErrorCode MapNvmeStatus(int status, bool is_write) {
    // Match on SCT+SC (lower 11 bits), ignoring DNR/More/CRD.
    const uint32_t nvme_sc = static_cast<uint32_t>(status) & kNvmeStatusMask;
    switch (nvme_sc) {
        case kNvmeScKvKeyNotExists:
            return ErrorCode::OBJECT_NOT_FOUND;
        case kNvmeScKvKeyExists:
            return ErrorCode::OBJECT_ALREADY_EXISTS;
        case kNvmeScInvalidValueSize:
        case kNvmeScInvalidKeySize:
            return ErrorCode::INVALID_PARAMS;
        case kNvmeScCapExceeded:
            return ErrorCode::KEYS_ULTRA_LIMIT;
        default:
            return is_write ? ErrorCode::FILE_WRITE_FAIL
                            : ErrorCode::FILE_READ_FAIL;
    }
}

// Parse the device-returned List buffer into a vector of PhysicalKeys.
// Buffer layout per NVMe KV spec:
//   Bytes 0-3: uint32_t num_keys (LE)
//   Bytes 4-7: reserved
//   For each key:
//     Bytes 0-1: uint16_t key_length (LE)
//     Bytes 2-(2+key_length-1): key_data
std::vector<NvmeKvCommandExecutor::PhysicalKey> ParseListBuffer(
    const uint8_t* buf, uint32_t buf_size) {
    std::vector<NvmeKvCommandExecutor::PhysicalKey> keys;
    if (buf_size < 8) {
        return keys;
    }
    const uint32_t num_keys = ReadLe32(buf);
    keys.reserve(std::min(num_keys, static_cast<uint32_t>(4096)));
    size_t offset = 8;
    for (uint32_t i = 0; i < num_keys && offset + 2 <= buf_size; ++i) {
        const uint16_t key_len = ReadLe16(buf + offset);
        offset += 2;
        if (key_len != sizeof(NvmeKvCommandExecutor::PhysicalKey) ||
            offset + key_len > buf_size) {
            break;
        }
        NvmeKvCommandExecutor::PhysicalKey key{};
        std::memcpy(key.data(), buf + offset, key_len);
        offset += key_len;
        keys.push_back(key);
    }
    return keys;
}

// Open and validate an NVMe device path, returning the fd or an error.
tl::expected<int, ErrorCode> OpenNvmeDevice(const std::string& device_path) {
    struct stat st{};
    if (::stat(device_path.c_str(), &st) != 0) {
        LOG(ERROR) << "[NvmeKvIoctlExecutor] stat failed for " << device_path
                   << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }
    if (!S_ISCHR(st.st_mode) && !S_ISBLK(st.st_mode)) {
        LOG(ERROR) << "[NvmeKvIoctlExecutor] " << device_path
                   << " is not a character or block device";
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    const int fd = ::open(device_path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        LOG(ERROR) << "[NvmeKvIoctlExecutor] open failed for " << device_path
                   << ": " << strerror(errno);
        return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
    }
    return fd;
}

// Probe KV namespace capabilities via Identify command.
std::optional<NvmeKvCommandExecutor::Capabilities> ProbeKvCapabilities(
    int fd, uint32_t nsid, const std::string& device_path) {
    std::vector<uint8_t> data(kNvmeIdentifyDataSize);
    nvme_passthru_cmd64 cmd{};
    cmd.opcode = kNvmeAdminIdentifyOpcode;
    cmd.nsid = nsid;
    cmd.addr = reinterpret_cast<uint64_t>(data.data());
    cmd.data_len = static_cast<uint32_t>(data.size());
    cmd.cdw10 = kNvmeIdentifyCsiNamespace;
    cmd.cdw11 = static_cast<uint32_t>(kNvmeKvCommandSetIndicator) << 24;
    cmd.timeout_ms = kNvmeIoctlTimeoutMs;

    const int ret = ::ioctl(fd, NVME_IOCTL_ADMIN64_CMD, &cmd);
    if (ret != 0) {
        LOG(WARNING) << "[NvmeKvIoctlExecutor] capability probe failed for "
                     << device_path << ", using conservative fallback: "
                     << (ret < 0 ? strerror(errno) : "command failed");
        return std::nullopt;
    }
    auto caps = ParseKvIdentifyNamespace(data);
    if (!caps) {
        LOG(WARNING) << "[NvmeKvIoctlExecutor] capability probe returned "
                        "invalid data for "
                     << device_path << ", using conservative fallback";
    }
    return caps;
}

class NvmeKvIoctlExecutor : public NvmeKvCommandExecutor {
   public:
    // Takes ownership of an already-opened file descriptor.
    NvmeKvIoctlExecutor(std::string device_path, uint32_t nsid,
                        Capabilities capabilities, int fd)
        : device_path_(std::move(device_path)),
          nsid_(nsid),
          capabilities_(capabilities),
          fd_(fd) {}

    NvmeKvIoctlExecutor(const NvmeKvIoctlExecutor&) = delete;
    NvmeKvIoctlExecutor& operator=(const NvmeKvIoctlExecutor&) = delete;

    ~NvmeKvIoctlExecutor() override {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    tl::expected<void, ErrorCode> Flush() override {
        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kNvmeKvFlushOpcode;
        cmd.nsid = nsid_;
        cmd.timeout_ms = kNvmeIoctlTimeoutMs;

        const PhysicalKey dummy_key{};
        return SubmitVoid(cmd, true, "flush", dummy_key);
    }

    tl::expected<void, ErrorCode> Store(const PhysicalKey& key,
                                        std::string value) override {
        if (value.empty() ||
            value.size() > capabilities_.effective_max_value_size) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }

        const uint32_t value_size = static_cast<uint32_t>(value.size());
        const uint32_t kvcg = capabilities_.kvcg;
        const uint32_t transfer_bytes =
            RoundUpToKvTransferBytes(value_size, kvcg);
        auto dma_buffer = AllocateAlignedBuffer(transfer_bytes);
        if (dma_buffer == nullptr) {
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        std::memset(dma_buffer.get(), 0, transfer_bytes);
        std::memcpy(dma_buffer.get(), value.data(), value.size());

        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kNvmeKvStoreOpcode;
        cmd.nsid = nsid_;
        cmd.addr = reinterpret_cast<uint64_t>(dma_buffer.get());
        cmd.data_len = transfer_bytes;
        cmd.cdw10 = value_size;
        cmd.cdw11 = BuildKeyLengthField(key.size());
        auto block_count = ComputeKvBlockCountMinusOne(value_size, kvcg);
        if (!block_count) {
            return tl::make_unexpected(block_count.error());
        }
        cmd.cdw12 = block_count.value();
        cmd.cdw13 = 0;
        cmd.timeout_ms = kNvmeIoctlTimeoutMs;
        EncodeKeyIntoCommand(key, cmd);

        return SubmitVoid(cmd, true, "store", key);
    }

    tl::expected<std::string, ErrorCode> Retrieve(
        const PhysicalKey& key) const override {
        auto raw = RawRetrieve(key, "retrieve");
        if (!raw) {
            return tl::make_unexpected(raw.error());
        }
        auto& [buf, size] = raw.value();
        return std::string(buf.get(), buf.get() + size);
    }

    tl::expected<uint32_t, ErrorCode> RetrieveInto(
        const PhysicalKey& key, void* buffer,
        uint32_t buffer_size) const override {
        auto raw = RawRetrieve(key, "retrieve_into");
        if (!raw) {
            return tl::make_unexpected(raw.error());
        }
        auto& [buf, size] = raw.value();
        if (size > buffer_size) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        std::memcpy(buffer, buf.get(), size);
        return size;
    }

    tl::expected<bool, ErrorCode> Exists(
        const PhysicalKey& key) const override {
        auto cmd = BuildKeyOnlyCommand(kNvmeKvExistOpcode, key);
        auto result = Submit(cmd, false, "exists", key);
        if (!result) {
            if (result.error() == ErrorCode::OBJECT_NOT_FOUND) {
                return false;
            }
            return tl::make_unexpected(result.error());
        }
        return true;
    }

    tl::expected<void, ErrorCode> Delete(const PhysicalKey& key) override {
        auto cmd = BuildKeyOnlyCommand(kNvmeKvDeleteOpcode, key);
        return SubmitVoid(cmd, true, "delete", key);
    }

    tl::expected<std::vector<PhysicalKey>, ErrorCode> List(
        const PhysicalKey& prefix, uint8_t prefix_len,
        uint32_t max_keys) const override {
        if (prefix_len > sizeof(PhysicalKey)) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        if (max_keys == 0) {
            return std::vector<PhysicalKey>{};
        }
        // Buffer layout: 4-byte num_keys + 4-byte reserved + entries
        // Each entry: 2-byte key_length + key_data
        const uint32_t kvcg = capabilities_.kvcg;
        const uint32_t entry_size =
            static_cast<uint32_t>(sizeof(uint16_t) + sizeof(PhysicalKey));
        const uint64_t raw_buf_size =
            8ull + static_cast<uint64_t>(max_keys) * entry_size;
        if (raw_buf_size > UINT32_MAX) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        const uint32_t buf_size =
            RoundUpToKvTransferBytes(static_cast<uint32_t>(raw_buf_size), kvcg);
        auto dma_buffer = AllocateAlignedBuffer(buf_size);
        if (dma_buffer == nullptr) {
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        std::memset(dma_buffer.get(), 0, buf_size);

        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kNvmeKvListOpcode;
        cmd.nsid = nsid_;
        cmd.addr = reinterpret_cast<uint64_t>(dma_buffer.get());
        cmd.data_len = buf_size;
        cmd.cdw10 = buf_size;
        cmd.cdw11 = BuildKeyLengthField(prefix_len);
        auto block_count = ComputeKvBlockCountMinusOne(buf_size, kvcg);
        if (!block_count) {
            return tl::make_unexpected(block_count.error());
        }
        cmd.cdw12 = block_count.value();
        cmd.cdw13 = 0;
        cmd.timeout_ms = kNvmeIoctlTimeoutMs;
        EncodeKeyIntoCommand(prefix, cmd);

        auto result = Submit(cmd, false, "list", prefix);
        if (!result) {
            return tl::make_unexpected(result.error());
        }

        return ParseListBuffer(
            reinterpret_cast<const uint8_t*>(dma_buffer.get()), buf_size);
    }

    const Capabilities& GetCapabilities() const override {
        return capabilities_;
    }

    std::string GetBackendType() const override { return "ioctl"; }

   private:
    using RawResult = std::pair<AlignedUniquePtr<char>, uint32_t>;

    tl::expected<RawResult, ErrorCode> RawRetrieve(const PhysicalKey& key,
                                                   const char* op_name) const {
        const uint32_t max_size = capabilities_.effective_max_value_size;
        const uint32_t kvcg = capabilities_.kvcg;
        auto dma_buffer = AllocateAlignedBuffer(max_size);
        if (dma_buffer == nullptr) {
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        std::memset(dma_buffer.get(), 0, max_size);
        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kNvmeKvRetrieveOpcode;
        cmd.nsid = nsid_;
        cmd.addr = reinterpret_cast<uint64_t>(dma_buffer.get());
        cmd.data_len = max_size;
        cmd.cdw10 = max_size;
        cmd.cdw11 = BuildKeyLengthField(key.size());
        auto block_count = ComputeKvBlockCountMinusOne(max_size, kvcg);
        if (!block_count) {
            return tl::make_unexpected(block_count.error());
        }
        cmd.cdw12 = block_count.value();
        cmd.cdw13 = 0;
        cmd.timeout_ms = kNvmeIoctlTimeoutMs;
        EncodeKeyIntoCommand(key, cmd);

        auto result = Submit(cmd, false, op_name, key);
        if (!result) {
            return tl::make_unexpected(result.error());
        }

        const uint32_t actual_size = ResolveNvmeKvObjectValueSize(
            dma_buffer.get(), result.value(),
            capabilities_.effective_max_value_size);
        if (actual_size == 0 ||
            actual_size > capabilities_.effective_max_value_size) {
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
        }
        return RawResult{std::move(dma_buffer), actual_size};
    }

    nvme_passthru_cmd64 BuildKeyOnlyCommand(uint8_t opcode,
                                            const PhysicalKey& key) const {
        nvme_passthru_cmd64 cmd{};
        cmd.opcode = opcode;
        cmd.nsid = nsid_;
        cmd.cdw10 = 0;
        cmd.cdw11 = BuildKeyLengthField(key.size());
        cmd.cdw12 = 0;
        cmd.cdw13 = 0;
        cmd.timeout_ms = kNvmeIoctlTimeoutMs;
        EncodeKeyIntoCommand(key, cmd);
        return cmd;
    }

    tl::expected<void, ErrorCode> SubmitVoid(nvme_passthru_cmd64& cmd,
                                             bool is_write, const char* op_name,
                                             const PhysicalKey& key) const {
        auto result = Submit(cmd, is_write, op_name, key);
        if (!result) {
            return tl::make_unexpected(result.error());
        }
        return {};
    }

    tl::expected<uint32_t, ErrorCode> Submit(nvme_passthru_cmd64& cmd,
                                             bool is_write, const char* op_name,
                                             const PhysicalKey& key) const {
        TraceCommandBuild(op_name, key, cmd);
        const int ret = ::ioctl(fd_, NVME_IOCTL_IO64_CMD, &cmd);
        if (ret != 0) {
            const int err = (ret < 0) ? errno : ret;
            const ErrorCode mapped = (ret < 0) ? MapErrno(err, is_write)
                                               : MapNvmeStatus(err, is_write);
            TraceCommandResult(op_name, key, cmd, -err, mapped, false,
                               cmd.result);
            return tl::make_unexpected(mapped);
        }
        TraceCommandResult(op_name, key, cmd, 0, ErrorCode::OK, true,
                           cmd.result);
        return static_cast<uint32_t>(cmd.result);
    }

    std::string device_path_;
    uint32_t nsid_ = 1;
    Capabilities capabilities_;
    int fd_ = -1;
};

}  // namespace

std::unique_ptr<NvmeKvCommandExecutor> CreateNvmeKvIoctlExecutor(
    std::string device_path, uint32_t nsid,
    tl::expected<NvmeKvCommandExecutor::Capabilities, ErrorCode>&
        capabilities) {
    constexpr uint32_t fallback_limit = kConservativeValueSizeFallback;

    NvmeKvCommandExecutor::Capabilities fallback_caps;
    fallback_caps.max_key_size = kMooncakePhysicalKeySize;
    fallback_caps.max_value_size = fallback_limit;
    fallback_caps.effective_max_value_size = fallback_limit;
    fallback_caps.kvcg = kDefaultKvcgBytes;

    auto fd_result = OpenNvmeDevice(device_path);
    if (!fd_result) {
        capabilities = tl::make_unexpected(fd_result.error());
        return nullptr;
    }
    const int fd = fd_result.value();

    auto probed_caps = ProbeKvCapabilities(fd, nsid, device_path);
    NvmeKvCommandExecutor::Capabilities caps =
        probed_caps.value_or(fallback_caps);
    if (caps.max_key_size < kMooncakePhysicalKeySize) {
        LOG(ERROR) << "[NvmeKvIoctlExecutor] device max key size "
                   << caps.max_key_size << " is smaller than Mooncake key size "
                   << kMooncakePhysicalKeySize;
        ::close(fd);
        capabilities = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        return nullptr;
    }

    capabilities = caps;
    return std::make_unique<NvmeKvIoctlExecutor>(std::move(device_path), nsid,
                                                 caps, fd);
}

std::unique_ptr<NvmeKvCommandExecutor> CreateNvmeKvExecutor(
    const std::string& type, const std::string& device_path, uint32_t nsid,
    tl::expected<NvmeKvCommandExecutor::Capabilities, ErrorCode>&
        capabilities) {
    if (type.empty() || type == "ioctl") {
        return CreateNvmeKvIoctlExecutor(device_path, nsid, capabilities);
    }
    LOG(ERROR) << "[NvmeKvExecutor] unsupported executor type: " << type;
    capabilities = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    return nullptr;
}

}  // namespace mooncake
