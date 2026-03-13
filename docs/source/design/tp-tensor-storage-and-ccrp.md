# TP Tensor Storage and CCRP Design

This document describes the TP-based tensor storage interfaces, metadata structures, and CCRP (Coalesced Chunk Retrieval Protocol) implementation in Mooncake Store.

## 1. Overview

In LLM inference, different stages may use different TP sizes (e.g., 8-way TP for prefill, 4-way TP for decode). Mooncake Store supports **heterogeneous TP put/get** via **metadata + chunk separation**: each writer rank writes its own chunk independently; reader ranks can reconstruct slices for any target TP size on demand, without All Gather over the full tensor.

- **Put side**: `put_tensor_chunk_with_tp` — each rank writes only its chunk, plus chunk metadata and global metadata.
- **Get side**: `get_tensor_with_tp_into` — computes the slice range from tp_rank/tp_size/split_dim and reconstructs via CCRP with a single batch read into the user buffer.

## 2. Interface Description

### 2.1 put_tensor_chunk_with_tp

Each TP writer rank writes its chunk independently.

```python
def put_tensor_chunk_with_tp(
    self,
    key: str,
    tensor_chunk: torch.Tensor,
    put_tp_rank: int,
    put_tp_size: int = 1,
    split_dim: int = 0,
    full_shape: Optional[List[int]] = None,
    config: ReplicateConfig = None
) -> int
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|--------------|
| key | str | Logical key; used to derive chunk keys and metadata keys |
| tensor_chunk | torch.Tensor | Chunk owned by this rank; must be contiguous |
| put_tp_rank | int | Writer TP rank (0..put_tp_size-1) |
| put_tp_size | int | Writer-side TP size |
| split_dim | int | Dimension along which to split; must match get side |
| full_shape | Optional | Full tensor shape; inferred from chunk if omitted |
| config | ReplicateConfig | Optional replication config |

**Stored content:**

- Chunk data: `key_tp_{put_tp_rank}`, format: `TensorMetadata(40B) + raw data`
- Chunk metadata: `key_tp_{put_tp_rank}_meta`, `ChunkMetadata`
- Global metadata: `key_global_meta`, written by rank 0 only, `GlobalMetadata`

### 2.2 get_tensor_with_tp_into

Writes the target slice directly into the user-provided buffer based on tp_rank/tp_size. Supports put_tp ≠ get_tp.

```python
def get_tensor_with_tp_into(
    self,
    key: str,
    buffer_ptr: int,
    size: int,
    tp_rank: int = 0,
    tp_size: int = 1,
    split_dim: int = 0
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|--------------|
| key | str | Logical key |
| buffer_ptr | int | Pre-allocated buffer address, must be registered |
| size | int | Buffer size; must be ≥ `TENSOR_METADATA_SIZE + output_numel * element_size` |
| tp_rank | int | Reader TP rank |
| tp_size | int | Reader-side TP size |
| split_dim | int | Split dimension |

**Returns:** `torch.Tensor` wrapping the buffer data; `None` on failure.

**Output format:** `TensorMetadata(40B) + slice data`.

### 2.3 Related Batch APIs

- `batch_put_tensor_chunk_with_tp`: Batch put chunks
- `batch_get_tensor_with_tp_into`: Batch get slices

## 3. Metadata Structures

### 3.1 TensorMetadata

First 40 bytes of each chunk object; describes dtype, ndim, and shape.

```cpp
struct TensorMetadata {
    int32_t dtype;   // TensorDtype enum
    int32_t ndim;    // Number of dimensions, max 4
    int64_t shape[4];// Shape of this chunk (not full shape)
};
// sizeof = 4 + 4 + 8*4 = 40 bytes
```

### 3.2 GlobalMetadata

One per logical key, written by put rank 0; describes the full tensor.

```cpp
#pragma pack(push, 1)
struct GlobalMetadata {
    int32_t dtype;
    int32_t ndim;
    int32_t split_dim;   // Split dimension
    int64_t shape[4];    // Full tensor shape
};
#pragma pack(pop)
```

**Key:** `{logical_key}_global_meta`

### 3.3 ChunkMetadata

One per chunk; describes this chunk's range on split_dim.

```cpp
#pragma pack(push, 1)
struct ChunkMetadata {
    int64_t start_idx;  // Start index on split_dim
    int64_t size;       // Length on split_dim
};
#pragma pack(pop)
```

**Key:** `{logical_key}_tp_{rank}_meta`

### 3.4 Key Naming Convention

| Purpose | Key Format | Description |
|---------|------------|-------------|
| Chunk data | `{key}_tp_{rank}` | Rank's chunk, contains TensorMetadata + data |
| Chunk metadata | `{key}_tp_{rank}_meta` | ChunkMetadata |
| Global metadata | `{key}_global_meta` | GlobalMetadata |

## 4. CCRP Implementation

When `put_tp_size != get_tp_size` or slices must be reconstructed, data is extracted from multiple writer chunks for the reader's `[r_start, r_end)` range. CCRP performs this with a single `batch_get_buffer_ranges` call across multiple keys and ranges, avoiding N separate single-range gets.

### 4.1 Flow

1. **Read GlobalMetadata**: Obtain full shape, split_dim, element_size
2. **Compute reader slice range**: `[r_start, r_end) = calculate_chunk_range(dim_size, tp_rank, tp_size)`
3. **Batch fetch ChunkMetadata**: `key_tp_0_meta` .. `key_tp_{put_tp-1}_meta`
4. **Build CCRP range list**: For each overlapping writer chunk, compute `[inter_start, inter_end) = [r_start,r_end) ∩ [w_start,w_end)`, expand by slice into (chunk_key, dest_offset, src_offset, size)
5. **Single batch_get_buffer_ranges**: Write all ranges into the output buffer

### 4.2 reconstruct_tensor_from_chunks Pseudocode

```cpp
void reconstruct_tensor_from_chunks(key, global_meta, output_data,
                                    r_start, r_end, r_size, element_size) {
    // Phase 1: Batch get all ChunkMetadata
    chunk_meta_buffers = batch_get_buffer([key_tp_0_meta, ..., key_tp_N_meta]);

    // Phase 2: Build (keys, dest_offsets, src_offsets, sizes) for each slice
    for each writer_rank with overlapping [w_start, w_end):
        inter = overlap([r_start,r_end), [w_start,w_end))
        chunk_key = key_tp_{writer_rank}
        if split_dim == 0:
            append one range (chunk_key, dst_off, src_off, size)
        else:
            for each slice_idx in elements_before:
                append range for this slice
    // Phase 3: Single batch_get_buffer_ranges (RealClient internal API)
    batch_get_buffer_ranges(keys, output_data, dest_offsets, src_offsets, sizes);
}
```

### 4.3 batch_get_buffer_ranges (Internal API)

`batch_get_buffer_ranges` is an internal RealClient method and is not exposed publicly. It supports multiple ranges for the same key in one batch read; ranges are aggregated by key and transferred via a single Transfer-layer `submitTransfer`.

**Parameters:**

- `keys`: May repeat; same key means multiple ranges from the same chunk
- `dest_buffer`: Base address of destination buffer
- `dest_offsets`: Offset of each range in dest_buffer
- `src_offsets`: Offset of each range within the corresponding key object
- `sizes`: Size in bytes of each range

### 4.4 split_dim and Data Layout

- **split_dim == 0**: Each range maps to contiguous memory; one (key, dest_off, src_off, size) per range
- **split_dim > 0**: Must expand over `elements_before` slices; each slice is one range, `src_offset = chunk_meta + slice_idx * chunk_stride + src_start * slice_size`

## 5. Buffer Requirements

The buffer used by `get_tensor_with_tp_into` must:

1. Be registered via `register_buffer(ptr, size)`
2. Be at least `TENSOR_METADATA_SIZE + (full_numel / get_tp_size) * element_size` bytes
3. Be unregistered via `unregister_buffer(ptr)` when done

## 6. References

- Performance test script: `scripts/test_tensor_tp_reconstruction.py`
- Test results summary: `scripts/TP_RECONSTRUCTION_RESULTS_512MB.md`
