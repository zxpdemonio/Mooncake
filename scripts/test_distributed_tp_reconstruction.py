#!/usr/bin/env python3
"""
Distributed TP reconstruction performance test across multiple nodes.

Coordinates nodes via torch.distributed (gloo backend).
Each node runs a local process pool for mooncake store workers.
All put/get use zero-copy interfaces over RDMA (put_tensor_chunk_with_tp_from / put_tensor_from / put_tensor_with_tp_from / get_tensor_into / get_tensor_with_tp_into).

Each iteration measures overlapped end-to-end makespan: local put workers and
local get workers are launched together, and readers retry until data becomes
visible.

Test groups (each has its own independent put + get):
  1. CCRP (experimental): distributed put_tensor_chunk_with_tp_from + get_tensor_with_tp_into
  2. All Gather (control 1): node0 gathers + put_tensor_from (full tensor) + each reader get_tensor_into (full) + local slice
  3. Gather+TP (control 2): node0 gathers + single put_tensor_with_tp_from(full) + get_tensor_with_tp_into (fast path)

Example (2 nodes, put_tp=16, get_tp=8 and get_tp=32):
  Node 0:
    MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    python scripts/test_distributed_tp_reconstruction.py \
      --node_id 0 --num_nodes 2 --put_ranks_per_node 8 \
      --get_ranks_per_node 4,16 --size_mb 512 --split_dims 0 --iters 3

  Node 1:
    MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    python scripts/test_distributed_tp_reconstruction.py \
      --node_id 1 --num_nodes 2 --put_ranks_per_node 8 \
      --get_ranks_per_node 4,16 --size_mb 512 --split_dims 0 --iters 3
"""

import argparse
import ctypes
import multiprocessing
import os
import struct
import time

import numpy as np
import torch
import torch.distributed as dist

TENSOR_METADATA_SIZE = 4 + 4 + 8 * 4  # 40 bytes
DTYPE_MAP = {
    torch.float32: 0, torch.float64: 1, torch.int8: 2, torch.uint8: 3,
    torch.int16: 4, torch.uint16: 5, torch.int32: 6, torch.uint32: 7,
    torch.int64: 8, torch.uint64: 9, torch.bool: 10, torch.float16: 11,
    torch.bfloat16: 12,
}
GET_RETRY_TIMEOUT_SEC = 10.0
GET_RETRY_INTERVAL_SEC = 0.01


def _tp_data_key(base_key, rank):
    return f"{base_key}_tp_{rank}"


def _tp_meta_key(base_key, rank):
    return f"{base_key}_tp_{rank}_meta"


def _global_meta_key(base_key):
    return f"{base_key}_global_meta"


def _ready_key(base_key):
    return f"{base_key}_ready"


def _tp_ready_key(base_key, rank):
    return f"{base_key}_ready_tp_{rank}"


def _mark_ready(store, key):
    return store.put(key, b"1") == 0


# ── Helpers ──

def chunk_serialized_size(chunk):
    return TENSOR_METADATA_SIZE + chunk.numel() * chunk.element_size()


def serialize_chunk_to_buffer(chunk, buf):
    dtype_id = DTYPE_MAP.get(chunk.dtype, 0)
    ndim = chunk.ndim
    shape = list(chunk.shape) + [-1] * (4 - chunk.ndim)
    struct.pack_into("<iiqqqq", buf, 0, dtype_id, ndim, *shape[:4])
    data = chunk.cpu().numpy().tobytes()
    ctypes.memmove(ctypes.addressof(buf) + TENSOR_METADATA_SIZE, data, len(data))


def get_output_buffer_size(tensor, get_tp):
    output_numel = (tensor.numel() * tensor.element_size()) // get_tp
    return TENSOR_METADATA_SIZE + output_numel


def create_store():
    from mooncake.store import MooncakeDistributedStore
    from mooncake.mooncake_config import MooncakeConfig
    store = MooncakeDistributedStore()
    config = MooncakeConfig.load_from_env()
    rc = store.setup(
        config.local_hostname, config.metadata_server,
        config.global_segment_size, config.local_buffer_size,
        "rdma", config.device_name, config.master_server_address,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to setup mooncake store, error code: {rc}")
    return store


def _init_worker_store():
    global _worker_store
    _worker_store = create_store()


def make_tensor_for_split_dim(split_dim, size_mb, max_tp=32):
    size_bytes = int(size_mb * 1024 * 1024)
    target_numel = size_bytes // 4  # float32
    other_dims = [32, 32, 32]
    prod_other = 32 * 32 * 32
    split_dim_size = target_numel // prod_other
    split_dim_size = max(max_tp, (split_dim_size // max_tp) * max_tp)
    shape = other_dims[:split_dim] + [split_dim_size] + other_dims[split_dim:]
    return torch.randn(*shape, dtype=torch.float32).contiguous()


# ── Worker functions (all zero-copy, run in pool processes) ──

def _put_worker(args):
    """CCRP put: each rank puts its chunk via put_tensor_chunk_with_tp_from (zero-copy)."""
    put_rank, key, chunk, put_tp, split_dim = args
    chunk = chunk.contiguous()
    sz = chunk_serialized_size(chunk)
    buf = (ctypes.c_ubyte * sz)()
    serialize_chunk_to_buffer(chunk, buf)
    ptr = ctypes.addressof(buf)
    if _worker_store.register_buffer(ptr, sz) != 0:
        return False
    try:
        rc = _worker_store.put_tensor_chunk_with_tp_from(
            key, ptr, sz, tp_rank=put_rank, tp_size=put_tp, split_dim=split_dim
        )
        if rc != 0:
            return False
        return _mark_ready(_worker_store, _tp_ready_key(key, put_rank))
    finally:
        _worker_store.unregister_buffer(ptr)


def _put_worker_from_tensor(args):
    """CCRP put using a shared full tensor to avoid parent->worker chunk copies."""
    put_rank, key, tensor, put_tp, split_dim = args
    chunk = tensor.chunk(put_tp, split_dim)[put_rank]
    return _put_worker((put_rank, key, chunk, put_tp, split_dim))


def _get_worker(args):
    """CCRP get: get_tensor_with_tp_into (zero-copy)."""
    get_rank, key, tensor, get_tp, split_dim = args
    store = _worker_store
    expected = tensor.chunk(get_tp, split_dim)[get_rank]
    sz = get_output_buffer_size(tensor, get_tp)
    buf = (ctypes.c_ubyte * sz)()
    ptr = ctypes.addressof(buf)
    if store.register_buffer(ptr, sz) != 0:
        return False
    try:
        t = store.get_tensor_with_tp_into(
            key, ptr, sz, tp_rank=get_rank, tp_size=get_tp, split_dim=split_dim
        )
        if t is None:
            return False
        t_cpu = t.cpu().float() if t.is_cuda else t.float()
        exp_cpu = expected.cpu().float() if expected.is_cuda else expected.float()
        return bool(torch.allclose(t_cpu, exp_cpu, rtol=1e-4, atol=1e-5))
    finally:
        store.unregister_buffer(ptr)


def _put_full_tensor_worker(args):
    """All Gather put: rank0 puts full tensor via put_tensor_from (zero-copy)."""
    key, tensor = args
    tensor = tensor.contiguous()
    sz = chunk_serialized_size(tensor)
    buf = (ctypes.c_ubyte * sz)()
    serialize_chunk_to_buffer(tensor, buf)
    ptr = ctypes.addressof(buf)
    if _worker_store.register_buffer(ptr, sz) != 0:
        return False
    try:
        rc = _worker_store.put_tensor_from(key, ptr, sz)
        if rc != 0:
            return False
        return _mark_ready(_worker_store, _ready_key(key))
    finally:
        _worker_store.unregister_buffer(ptr)


def _get_full_tensor_worker(args):
    """All Gather get: read full tensor via get_tensor_into, slice locally (zero-copy)."""
    get_rank, key, tensor, get_tp, split_dim = args
    store = _worker_store
    expected = tensor.chunk(get_tp, split_dim)[get_rank]
    full_sz = chunk_serialized_size(tensor)
    buf = (ctypes.c_ubyte * full_sz)()
    ptr = ctypes.addressof(buf)
    if store.register_buffer(ptr, full_sz) != 0:
        return False
    try:
        t = store.get_tensor_into(key, ptr, full_sz)
        if t is None:
            return False
        my_chunk = t.chunk(get_tp, split_dim)[get_rank]
        t_cpu = my_chunk.cpu().float() if my_chunk.is_cuda else my_chunk.float()
        exp_cpu = expected.cpu().float() if expected.is_cuda else expected.float()
        return bool(torch.allclose(t_cpu, exp_cpu, rtol=1e-4, atol=1e-5))
    finally:
        store.unregister_buffer(ptr)


def _put_gather_tp_worker(args):
    """Gather+TP put: rank0 writes one full tensor buffer via put_tensor_with_tp_from."""
    key, tensor, get_tp, split_dim = args
    store = _worker_store
    tensor = tensor.contiguous()
    sz = chunk_serialized_size(tensor)
    buf = (ctypes.c_ubyte * sz)()
    serialize_chunk_to_buffer(tensor, buf)
    ptr = ctypes.addressof(buf)
    if store.register_buffer(ptr, sz) != 0:
        return False
    try:
        rc = store.put_tensor_with_tp_from(
            key, ptr, sz, tp_rank=0, tp_size=get_tp, split_dim=split_dim
        )
        if rc != 0:
            return False
        return _mark_ready(store, _ready_key(key))
    finally:
        store.unregister_buffer(ptr)


def _get_tp_chunk_worker(args):
    """Gather+TP get: get_tensor_with_tp_into (zero-copy, put_tp==get_tp fast path)."""
    get_rank, key, tensor, get_tp, split_dim = args
    store = _worker_store
    expected = tensor.chunk(get_tp, split_dim)[get_rank]
    sz = get_output_buffer_size(tensor, get_tp)
    buf = (ctypes.c_ubyte * sz)()
    ptr = ctypes.addressof(buf)
    if store.register_buffer(ptr, sz) != 0:
        return False
    try:
        t = store.get_tensor_with_tp_into(
            key, ptr, sz,
            tp_rank=get_rank, tp_size=get_tp, split_dim=split_dim
        )
        if t is None:
            return False
        t_cpu = t.cpu().float() if t.is_cuda else t.float()
        exp_cpu = expected.cpu().float() if expected.is_cuda else expected.float()
        return bool(torch.allclose(t_cpu, exp_cpu, rtol=1e-4, atol=1e-5))
    finally:
        store.unregister_buffer(ptr)


# ── Distributed timing helper ──

def dist_max_time(local_time):
    """All-reduce local elapsed time, return global max across all nodes."""
    t = torch.tensor([local_time], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _simulate_gathered_tensor(tensor, put_tp, split_dim):
    """Simulate writer-side gather by reassembling the full tensor from put_tp chunks."""
    if put_tp <= 1:
        return tensor.contiguous()
    return torch.cat(list(tensor.chunk(put_tp, split_dim)), dim=split_dim).contiguous()


def _ccrp_ready(store, key, put_tp):
    for rank in range(put_tp):
        if store.is_exist(_tp_ready_key(key, rank)) != 1:
            return False
    return True


def _all_gather_ready(store, key):
    return store.is_exist(_ready_key(key)) == 1


def _gather_tp_ready(store, key, _get_tp):
    return store.is_exist(_ready_key(key)) == 1


def _get_with_retry(get_fn, ready_fn, args, timeout_s=GET_RETRY_TIMEOUT_SEC,
                    interval_s=GET_RETRY_INTERVAL_SEC):
    deadline = time.perf_counter() + timeout_s
    while True:
        if ready_fn(*args):
            return get_fn(args)
        if time.perf_counter() >= deadline:
            return False
        time.sleep(interval_s)


def _get_worker_retry(args):
    get_rank, key, tensor, get_tp, split_dim, put_tp = args
    return _get_with_retry(
        lambda inner_args: _get_worker(inner_args[:5]),
        lambda _get_rank, key_name, _tensor, _get_tp, _split_dim, put_tp_size: _ccrp_ready(_worker_store, key_name, put_tp_size),
        (get_rank, key, tensor, get_tp, split_dim, put_tp),
    )


def _get_full_tensor_worker_retry(args):
    get_rank, key, tensor, get_tp, split_dim = args
    return _get_with_retry(
        _get_full_tensor_worker,
        lambda _get_rank, key_name, _tensor, _get_tp, _split_dim: _all_gather_ready(_worker_store, key_name),
        (get_rank, key, tensor, get_tp, split_dim),
    )


def _get_tp_chunk_worker_retry(args):
    get_rank, key, tensor, get_tp, split_dim = args
    return _get_with_retry(
        _get_tp_chunk_worker,
        lambda _get_rank, key_name, _tensor, get_tp_size, _split_dim: _gather_tp_ready(_worker_store, key_name, get_tp_size),
        (get_rank, key, tensor, get_tp, split_dim),
    )


def _run_overlapped_iteration(pool, put_fn, put_args, get_fn, get_args):
    put_async = pool.map_async(put_fn, put_args) if put_args else None
    get_async = pool.map_async(get_fn, get_args) if get_args else None
    put_results = put_async.get() if put_async is not None else []
    get_results = get_async.get() if get_async is not None else []
    return all(put_results), all(get_results)


# ── Distributed test runner ──

def run_distributed_case(pool, node_id, num_nodes, key, tensor,
                         put_tp, put_ranks_per_node,
                         get_tp, get_ranks_per_node,
                         split_dim, iters):
    """Run one case across all nodes with overlapped put/get timing."""
    put_rank_start = node_id * put_ranks_per_node
    get_rank_start = node_id * get_ranks_per_node

    full_size = tensor.numel() * tensor.element_size()
    output_size = full_size // get_tp

    if node_id == 0:
        tmp = create_store()
        tmp.remove_all()
        tmp.close()
    dist.barrier()

    ccrp_times = []
    for it in range(iters):
        ccrp_key = f"{key}_ccrp_{it}"
        put_args = [
            (put_rank_start + r, ccrp_key, tensor, put_tp, split_dim)
            for r in range(put_ranks_per_node)
        ]
        get_args = [
            (get_rank_start + r, ccrp_key, tensor, get_tp, split_dim, put_tp)
            for r in range(get_ranks_per_node)
        ]
        dist.barrier()
        t0 = time.perf_counter()
        put_ok, get_ok = _run_overlapped_iteration(
            pool, _put_worker_from_tensor, put_args, _get_worker_retry, get_args
        )
        local_elapsed = time.perf_counter() - t0
        global_elapsed = dist_max_time(local_elapsed)
        if put_ok and get_ok:
            ccrp_times.append(global_elapsed)
        elif node_id == 0:
            print(f"  CCRP iter {it} FAILED")

    all_gather_times = []
    for it in range(iters):
        ag_key = f"{key}_ag_{it}"
        gathered_tensor = _simulate_gathered_tensor(tensor, put_tp, split_dim)
        put_args = [(ag_key, gathered_tensor)] if node_id == 0 else []
        get_args = [
            (get_rank_start + r, ag_key, tensor, get_tp, split_dim)
            for r in range(get_ranks_per_node)
        ]
        dist.barrier()
        t0 = time.perf_counter()
        put_ok, get_ok = _run_overlapped_iteration(
            pool, _put_full_tensor_worker, put_args,
            _get_full_tensor_worker_retry, get_args
        )
        local_elapsed = time.perf_counter() - t0
        global_elapsed = dist_max_time(local_elapsed)
        if put_ok and get_ok:
            all_gather_times.append(global_elapsed)
        elif node_id == 0:
            print(f"  AllGather iter {it} FAILED")

    gather_tp_times = []
    for it in range(iters):
        gtp_key = f"{key}_gtp_{it}"
        gathered_tensor = _simulate_gathered_tensor(tensor, put_tp, split_dim)
        put_args = [(gtp_key, gathered_tensor, get_tp, split_dim)] if node_id == 0 else []
        get_args = [
            (get_rank_start + r, gtp_key, tensor, get_tp, split_dim)
            for r in range(get_ranks_per_node)
        ]
        dist.barrier()
        t0 = time.perf_counter()
        put_ok, get_ok = _run_overlapped_iteration(
            pool, _put_gather_tp_worker, put_args,
            _get_tp_chunk_worker_retry, get_args
        )
        local_elapsed = time.perf_counter() - t0
        global_elapsed = dist_max_time(local_elapsed)
        if put_ok and get_ok:
            gather_tp_times.append(global_elapsed)
        elif node_id == 0:
            print(f"  Gather+TP iter {it} FAILED")

    ag_mean = np.mean(all_gather_times) if all_gather_times else None
    ccrp_mean = np.mean(ccrp_times) if ccrp_times else None
    gtp_mean = np.mean(gather_tp_times) if gather_tp_times else None
    ccrp_total_mb = (2 * full_size) / 1e6
    ag_total_mb = ((get_tp + 1) * full_size) / 1e6
    gtp_total_mb = (2 * full_size) / 1e6
    return {
        "put_tp": put_tp,
        "get_tp": get_tp,
        "split_dim": split_dim,
        "full_mb": full_size / 1e6,
        "output_mb": output_size / 1e6,
        "all_gather_ms": ag_mean * 1000 if ag_mean else None,
        "all_gather_mbps": ag_total_mb / ag_mean if ag_mean else None,
        "gather_tp_ms": gtp_mean * 1000 if gtp_mean else None,
        "gather_tp_mbps": gtp_total_mb / gtp_mean if gtp_mean else None,
        "ccrp_ms": ccrp_mean * 1000 if ccrp_mean else None,
        "ccrp_mbps": ccrp_total_mb / ccrp_mean if ccrp_mean else None,
    }


# ── Output ──

def print_table(results_table, fmt="text"):
    if fmt == "csv":
        print(
            "put_tp,get_tp,split_dim,full_mb,output_mb,"
            "all_gather_ms,all_gather_mbps,gather_tp_ms,gather_tp_mbps,"
            "ccrp_ms,ccrp_mbps,ag_speedup,gtp_speedup"
        )
        for r in results_table:
            ag_ms = r["all_gather_ms"] if r["all_gather_ms"] is not None else ""
            ag_mb = r["all_gather_mbps"] if r["all_gather_mbps"] is not None else ""
            gtp_ms = r["gather_tp_ms"] if r["gather_tp_ms"] is not None else ""
            gtp_mb = r["gather_tp_mbps"] if r["gather_tp_mbps"] is not None else ""
            ccrp_ms = r["ccrp_ms"] if r["ccrp_ms"] is not None else ""
            ccrp_mb = r["ccrp_mbps"] if r["ccrp_mbps"] is not None else ""
            ag_sp = ""
            if r["all_gather_ms"] and r["ccrp_ms"] and r["ccrp_ms"] > 0:
                ag_sp = f"{r['all_gather_ms'] / r['ccrp_ms']:.2f}x"
            gtp_sp = ""
            if r["gather_tp_ms"] and r["ccrp_ms"] and r["ccrp_ms"] > 0:
                gtp_sp = f"{r['gather_tp_ms'] / r['ccrp_ms']:.2f}x"
            print(
                f"{r['put_tp']},{r['get_tp']},{r['split_dim']},"
                f"{r['full_mb']:.2f},{r['output_mb']:.2f},"
                f"{ag_ms},{ag_mb},{gtp_ms},{gtp_mb},"
                f"{ccrp_ms},{ccrp_mb},{ag_sp},{gtp_sp}"
            )
        return

    print("\n" + "=" * 120)
    print(
        f"{'put_tp':>6} {'get_tp':>6} {'split':>5} | "
        f"{'ag_ms':>8} {'ag_MB/s':>10} | "
        f"{'gtp_ms':>8} {'gtp_MB/s':>10} | "
        f"{'ccrp_ms':>8} {'ccrp_MB/s':>10} | "
        f"{'ag/ccrp':>8} {'gtp/ccrp':>8}"
    )
    print("-" * 120)
    for r in results_table:
        ag_ms = f"{r['all_gather_ms']:.1f}" if r["all_gather_ms"] is not None else "-"
        ag_mb = f"{r['all_gather_mbps']:.0f}" if r["all_gather_mbps"] is not None else "-"
        gtp_ms = f"{r['gather_tp_ms']:.1f}" if r["gather_tp_ms"] is not None else "-"
        gtp_mb = f"{r['gather_tp_mbps']:.0f}" if r["gather_tp_mbps"] is not None else "-"
        ccrp_ms = f"{r['ccrp_ms']:.1f}" if r["ccrp_ms"] is not None else "-"
        ccrp_mb = f"{r['ccrp_mbps']:.0f}" if r["ccrp_mbps"] is not None else "-"
        ag_sp = "-"
        if r["all_gather_ms"] and r["ccrp_ms"] and r["ccrp_ms"] > 0:
            ag_sp = f"{r['all_gather_ms'] / r['ccrp_ms']:.2f}x"
        gtp_sp = "-"
        if r["gather_tp_ms"] and r["ccrp_ms"] and r["ccrp_ms"] > 0:
            gtp_sp = f"{r['gather_tp_ms'] / r['ccrp_ms']:.2f}x"
        print(
            f"{r['put_tp']:>6} {r['get_tp']:>6} {r['split_dim']:>5} | "
            f"{ag_ms:>8} {ag_mb:>10} | "
            f"{gtp_ms:>8} {gtp_mb:>10} | "
            f"{ccrp_ms:>8} {ccrp_mb:>10} | "
            f"{ag_sp:>8} {gtp_sp:>8}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Distributed TP reconstruction test"
    )
    parser.add_argument("--node_id", type=int, required=True,
                        help="This node's ID (0-indexed)")
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--master_addr", type=str, default=None,
                        help="torch.distributed master addr (or set MASTER_ADDR env)")
    parser.add_argument("--master_port", type=int, default=29500,
                        help="torch.distributed master port (or set MASTER_PORT env)")
    parser.add_argument("--put_ranks_per_node", type=int, default=8)
    parser.add_argument("--get_ranks_per_node", type=str, default="4,16",
                        help="Comma-separated get ranks per node, e.g. '4,16'")
    parser.add_argument("--size_mb", type=float, default=512)
    parser.add_argument("--split_dims", type=str, default="0",
                        help="Comma-separated split dimensions to test")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--csv", action="store_true", help="Output CSV format")
    args = parser.parse_args()

    # ── torch.distributed setup ──
    if args.master_addr:
        os.environ["MASTER_ADDR"] = args.master_addr
    if "MASTER_ADDR" not in os.environ:
        raise RuntimeError("Set --master_addr or MASTER_ADDR env var")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    dist.init_process_group(
        backend="gloo",
        world_size=args.num_nodes,
        rank=args.node_id,
    )

    get_ranks_list = [int(x) for x in args.get_ranks_per_node.split(",")]
    split_dims = [int(x) for x in args.split_dims.split(",")]
    put_tp = args.num_nodes * args.put_ranks_per_node

    # Pool size must accommodate overlapped local put + get workers.
    max_local = max(args.put_ranks_per_node + grpn for grpn in get_ranks_list)
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(max_local, initializer=_init_worker_store)

    # Build test cases
    cases = []
    for grpn in get_ranks_list:
        get_tp = args.num_nodes * grpn
        for sd in split_dims:
            if sd >= 4:
                continue
            cases.append((put_tp, args.put_ranks_per_node, get_tp, grpn, sd))

    max_split_tp = max(put_tp, max(args.num_nodes * g for g in get_ranks_list))

    if args.node_id == 0:
        print(f"\n=== Distributed TP Reconstruction Test (all zero-copy) ===")
        print(f"Nodes: {args.num_nodes}, put_ranks/node: {args.put_ranks_per_node}")
        print(f"  -> put_tp = {put_tp}")
        print(f"get_ranks/node configs: {get_ranks_list}")
        print(f"  -> get_tp values: {[args.num_nodes * g for g in get_ranks_list]}")
        print(f"size_mb={args.size_mb}, split_dims={split_dims}, iters={args.iters}")
        print(f"Total cases: {len(cases)}")
        print()
        print(f"Test groups per case:")
        print(f"  1. CCRP: distributed put_tensor_chunk_with_tp_from + get_tensor_with_tp_into")
        print(f"  2. All Gather: node0 put_tensor_from(full) + get_tensor_into(full) + local slice")
        print(f"  3. Gather+TP: node0 gathers full tensor + single put_tensor_with_tp_from + get_tensor_with_tp_into(fast path)")

    results_table = []
    try:
        for i, (pt, prpn, gt, grpn, sd) in enumerate(cases):
            # Same seed on all nodes -> identical tensor
            torch.manual_seed(42 + i)
            tensor = make_tensor_for_split_dim(sd, args.size_mb, max_split_tp)
            key = f"dist_tp_{pt}_{gt}_{sd}_{i}"

            if args.node_id == 0:
                print(f"\n--- Case {i+1}/{len(cases)}: "
                      f"put_tp={pt} get_tp={gt} split_dim={sd} ---")

            row = run_distributed_case(
                pool, args.node_id, args.num_nodes, key, tensor,
                pt, prpn, gt, grpn, sd, args.iters,
            )
            if row:
                results_table.append(row)
                if args.node_id == 0:
                    ccrp = f"{row['ccrp_ms']:.1f}ms" if row["ccrp_ms"] else "FAIL"
                    ag = f"{row['all_gather_ms']:.1f}ms" if row["all_gather_ms"] else "FAIL"
                    gtp = f"{row['gather_tp_ms']:.1f}ms" if row["gather_tp_ms"] else "FAIL"
                    print(f"  CCRP={ccrp}  AllGather={ag}  Gather+TP={gtp}")
            else:
                if args.node_id == 0:
                    print(f"  [FAIL]")
    finally:
        pool.close()
        pool.join()
        dist.destroy_process_group()

    if args.node_id == 0 and results_table:
        print_table(results_table, fmt="csv" if args.csv else "text")

    print(f"\n[Node {args.node_id}] Done.")


if __name__ == "__main__":
    main()
