#include "nvme_kv_key_codec.h"

#include <cstdio>
#include <cstring>

#include <xxhash.h>

namespace mooncake {

namespace {

constexpr char kHexDigits[] = "0123456789abcdef";

int HexCharValue(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

}  // namespace

uint32_t ComputeNvmeKvChecksum(std::span<const uint8_t> data) {
    return static_cast<uint32_t>(XXH32(data.data(), data.size(), 0));
}

std::array<uint8_t, 32> ComputeNvmeKvVerifyHash(const std::string& key) {
    std::array<uint8_t, 32> hash{};
    static constexpr std::string_view kPrefixes[] = {
        "verify:0:", "verify:1:", "verify:2:", "verify:3:"};
    std::string buf;
    buf.reserve(kPrefixes[0].size() + key.size());
    uint64_t hashes[4];
    for (int i = 0; i < 4; ++i) {
        buf.assign(kPrefixes[i]);
        buf.append(key);
        hashes[i] = XXH64(buf.data(), buf.size(), 0);
    }
    std::memcpy(hash.data(), hashes, sizeof(hashes));
    return hash;
}

std::string HexEncode(std::string_view data) {
    std::string result;
    result.resize(data.size() * 2);
    for (size_t i = 0; i < data.size(); ++i) {
        const auto byte = static_cast<uint8_t>(data[i]);
        result[i * 2] = kHexDigits[byte >> 4];
        result[i * 2 + 1] = kHexDigits[byte & 0x0F];
    }
    return result;
}

bool HexDecode(std::string_view encoded, std::string& value) {
    if (encoded.size() % 2 != 0) {
        return false;
    }
    value.clear();
    value.reserve(encoded.size() / 2);
    for (size_t i = 0; i < encoded.size(); i += 2) {
        const int high = HexCharValue(encoded[i]);
        const int low = HexCharValue(encoded[i + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        value.push_back(static_cast<char>((high << 4) | low));
    }
    return true;
}

std::string NvmeKvPhysicalKeyToHex(const NvmeKvPhysicalKey& physical_key) {
    char buf[32];
    for (size_t i = 0; i < physical_key.size(); ++i) {
        buf[i * 2] = kHexDigits[physical_key[i] >> 4];
        buf[i * 2 + 1] = kHexDigits[physical_key[i] & 0x0F];
    }
    return std::string(buf, 32);
}

bool ParseNvmeKvPhysicalKeyHex(std::string_view physical_key_hex,
                               NvmeKvPhysicalKey& physical_key) {
    if (physical_key_hex.size() != physical_key.size() * 2) {
        return false;
    }
    for (size_t i = 0; i < physical_key.size(); ++i) {
        const int high = HexCharValue(physical_key_hex[i * 2]);
        const int low = HexCharValue(physical_key_hex[i * 2 + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        physical_key[i] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

NvmeKvPhysicalKey EncodeNvmeKvPhysicalKey(const std::string& key) {
    NvmeKvPhysicalKey physical_key{};
    std::string buf;
    buf.reserve(5 + key.size());
    buf.assign("pk:0:");
    buf.append(key);
    const uint64_t h0 = XXH64(buf.data(), buf.size(), 0);
    buf.assign("pk:1:");
    buf.append(key);
    const uint64_t h1 = XXH64(buf.data(), buf.size(), 0);
    std::memcpy(physical_key.data(), &h0, sizeof(h0));
    std::memcpy(physical_key.data() + 8, &h1, sizeof(h1));
    return physical_key;
}

NvmeKvPhysicalKey EncodeNvmeKvChunkPhysicalKey(const std::string& key,
                                               uint32_t chunk_index) {
    NvmeKvPhysicalKey physical_key{};
    char index_buf[16];
    const int index_len =
        std::snprintf(index_buf, sizeof(index_buf), "%u", chunk_index);
    std::string buf;
    buf.reserve(8 + key.size() + 1 + static_cast<size_t>(index_len));
    buf.assign("chunk:0:");
    buf.append(key);
    buf.push_back(':');
    buf.append(index_buf, static_cast<size_t>(index_len));
    const uint64_t h0 = XXH64(buf.data(), buf.size(), 0);
    buf.assign("chunk:1:");
    buf.append(key);
    buf.push_back(':');
    buf.append(index_buf, static_cast<size_t>(index_len));
    const uint64_t h1 = XXH64(buf.data(), buf.size(), 0);
    std::memcpy(physical_key.data(), &h0, sizeof(h0));
    std::memcpy(physical_key.data() + 8, &h1, sizeof(h1));
    return physical_key;
}

}  // namespace mooncake
