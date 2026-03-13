#!/usr/bin/env python3
"""
TP reconstruction performance test.

Requirements:
1. Real multi-process for put and get
2. CCRP (get_tensor_with_tp_into) - direct is covered as CCRP fast path when put_tp==get_tp
3. All Gather as control group - each rank gets full tensor, slices locally
4. Each process puts/gets only its own chunk - single key, no batch API

Put: put_tp processes, each does put_tensor_chunk_with_tp(key, my_chunk, ...)
Get (CCRP): get_tp processes, each does get_tensor_with_tp_into(key, buffer, size, ...)
Get (All Gather control): get_tp processes, each fetches all chunks via get_tensor, concats, slices locally
"""

import argparse
import ctypes
import multiprocessing
import time
import uuid

import numpy as np
import torch

TENSOR_METADATA_SIZE = 4 + 4 + 8 * 4  # 40 bytes


def get_output_buffer_size(tensor, get_tp):
    """Size needed for get_tensor_with_tp_into buffer: meta + data."""
    output_numel = (tensor.numel() * tensor.element_size()) // get_tp
    return TENSOR_METADATA_SIZE + output_numel


def create_store():
    from mooncake.store import MooncakeDistributedStore
    from mooncake.mooncake_config import MooncakeConfig
    store = MooncakeDistributedStore()
    config = MooncakeConfig.load_from_env()
    rc = store.setup(
        config.local_hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to setup mooncake store, error code: {rc}")
    return store


def _init_worker_store():
    """Initialize worker's mooncake client (called once when worker process starts)."""
    global _worker_store
    _worker_store = create_store()


def make_tensor_for_split_dim(split_dim, size_mb, max_tp=8):
    """Create 4D tensor where split_dim dimension is divisible by max_tp."""
    size_bytes = int(size_mb * 1024 * 1024)
    target_numel = size_bytes // 4
    other_dims = [32, 32, 32]
    prod_other = 32 * 32 * 32
    split_dim_size = target_numel // prod_other
    split_dim_size = max(max_tp, (split_dim_size // max_tp) * max_tp)
    shape = other_dims[:split_dim] + [split_dim_size] + other_dims[split_dim:]
    return torch.randn(*shape, dtype=torch.float32).contiguous()


def _put_worker(args):
    """Put rank uses pre-created client, puts its single chunk (put_tensor_chunk_with_tp)."""
    put_rank, key, chunk, put_tp, split_dim = args
    rc = _worker_store.put_tensor_chunk_with_tp(
        key, chunk,
        put_tp_rank=put_rank, put_tp_size=put_tp, split_dim=split_dim
    )
    return rc == 0


def _get_worker(args):
    """Get rank uses pre-created client, gets its slice (get_tensor_with_tp_into, CCRP).
    Verifies data correctness against expected slice."""
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
        # Verify data correctness
        t_cpu = t.cpu().float() if t.is_cuda else t.float()
        exp_cpu = expected.cpu().float() if expected.is_cuda else expected.float()
        return bool(torch.allclose(t_cpu, exp_cpu, rtol=1e-4, atol=1e-5))
    finally:
        store.unregister_buffer(ptr)


def _get_chunk_key(base_key, rank):
    return f"{base_key}_tp_{rank}"


def _get_all_gather_worker(args):
    """Get rank uses pre-created client, gets full tensor (All Gather control).
    Verifies data correctness: concat chunks should match original, slice is correct."""
    get_rank, key, put_tp, get_tp, split_dim, expected_full = args
    store = _worker_store
    # Fetch all put_tp chunks one by one (All Gather: each rank gets all data)
    chunks = []
    for r in range(put_tp):
        c = store.get_tensor(_get_chunk_key(key, r))
        if c is None:
            return False
        chunks.append(c)
    # Concat along split_dim to get full tensor
    full = torch.cat(chunks, dim=split_dim)
    # Verify full tensor matches original
    full_cpu = full.cpu().float() if full.is_cuda else full.float()
    exp_cpu = expected_full.cpu().float() if expected_full.is_cuda else expected_full.float()
    if not torch.allclose(full_cpu, exp_cpu, rtol=1e-4, atol=1e-5):
        return False
    # Slice locally (rank's share)
    _ = full.chunk(get_tp, split_dim)[get_rank]
    return True


def run_one_case(pool, key, tensor, put_tp, get_tp, split_dim, iters, results_table):
    """Run one case: multi-process put + multi-process get (CCRP + All Gather control).
    Uses pre-created pool - workers have their own mooncake clients, no fork/spawn during test."""
    # Clear store before put (use temp client, close immediately)
    tmp_store = create_store()
    tmp_store.remove_all()
    tmp_store.close()

    # Chunk for put
    chunks = list(tensor.chunk(put_tp, split_dim))

    # Multi-process put: use pre-created pool
    put_results = pool.map(
        _put_worker,
        [(r, key, chunks[r], put_tp, split_dim) for r in range(put_tp)]
    )
    if not all(put_results):
        return False

    full_size = tensor.numel() * tensor.element_size()
    output_size = full_size // get_tp

    # Multi-process get (CCRP): use pre-created pool
    ccrp_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        get_results = pool.map(
            _get_worker,
            [(r, key, tensor, get_tp, split_dim) for r in range(get_tp)]
        )
        if all(get_results):
            ccrp_times.append(time.perf_counter() - t0)

    # Multi-process get (All Gather control): use pre-created pool
    all_gather_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        get_results = pool.map(
            _get_all_gather_worker,
            [(r, key, put_tp, get_tp, split_dim, tensor) for r in range(get_tp)]
        )
        if all(get_results):
            all_gather_times.append(time.perf_counter() - t0)

    row = {
        "put_tp": put_tp,
        "get_tp": get_tp,
        "split_dim": split_dim,
        "full_mb": full_size / 1e6,
        "output_mb": output_size / 1e6,
        "all_gather_ms": np.mean(all_gather_times) * 1000 if all_gather_times else None,
        "all_gather_mbps": (full_size / 1e6) / np.mean(all_gather_times) if all_gather_times else None,
        "ccrp_ms": np.mean(ccrp_times) * 1000 if ccrp_times else None,
        "ccrp_mbps": (full_size / 1e6) / np.mean(ccrp_times) if ccrp_times else None,
    }
    results_table.append(row)
    return bool(ccrp_times) and bool(all_gather_times)


def print_table(results_table, fmt="text"):
    """Print results as table."""
    if fmt == "csv":
        print(
            "put_tp,get_tp,split_dim,rel,full_mb,output_mb,"
            "all_gather_ms,all_gather_mbps,ccrp_ms,ccrp_mbps,speedup"
        )
        for r in results_table:
            rel = (
                "eq" if r["put_tp"] == r["get_tp"]
                else ("gt" if r["put_tp"] > r["get_tp"] else "lt")
            )
            ag_ms = r["all_gather_ms"] if r["all_gather_ms"] is not None else ""
            ag_mb = r["all_gather_mbps"] if r["all_gather_mbps"] is not None else ""
            ccrp_ms = r["ccrp_ms"] if r["ccrp_ms"] is not None else ""
            ccrp_mb = r["ccrp_mbps"] if r["ccrp_mbps"] is not None else ""
            sp = ""
            if r["all_gather_ms"] and r["ccrp_ms"] and r["all_gather_ms"] > 0:
                sp = f"{r['all_gather_ms'] / r['ccrp_ms']:.2f}x"
            print(
                f"{r['put_tp']},{r['get_tp']},{r['split_dim']},{rel},"
                f"{r['full_mb']:.2f},{r['output_mb']:.2f},"
                f"{ag_ms},{ag_mb},{ccrp_ms},{ccrp_mb},{sp}"
            )
        return

    print("\n" + "=" * 95)
    print(
        f"{'put_tp':>6} {'get_tp':>6} {'split':>5} {'rel':>3} | "
        f"{'all_gather_ms':>12} {'all_gather_MB/s':>14} | "
        f"{'ccrp_ms':>8} {'ccrp_MB/s':>10} | {'speedup':>8}"
    )
    print("-" * 95)
    for r in results_table:
        rel = (
            "==" if r["put_tp"] == r["get_tp"]
            else (">" if r["put_tp"] > r["get_tp"] else "<")
        )
        ag_ms = f"{r['all_gather_ms']:.1f}" if r["all_gather_ms"] is not None else "-"
        ag_mb = f"{r['all_gather_mbps']:.0f}" if r["all_gather_mbps"] is not None else "-"
        ccrp_ms = f"{r['ccrp_ms']:.1f}" if r["ccrp_ms"] is not None else "-"
        ccrp_mb = f"{r['ccrp_mbps']:.0f}" if r["ccrp_mbps"] is not None else "-"
        sp = "-"
        if r["all_gather_ms"] and r["ccrp_ms"] and r["all_gather_ms"] > 0:
            sp = f"{r['all_gather_ms'] / r['ccrp_ms']:.2f}x"
        print(
            f"{r['put_tp']:>6} {r['get_tp']:>6} {r['split_dim']:>5} {rel:>3} | "
            f"{ag_ms:>12} {ag_mb:>14} | "
            f"{ccrp_ms:>8} {ccrp_mb:>10} | {sp:>8}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="TP reconstruction test: multi-process put/get, CCRP + All Gather control"
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--size_mb", type=float, default=512)
    parser.add_argument("--tp_sizes", type=str, default="4,8")
    parser.add_argument("--split_dims", type=str, default="0,1,2")
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--skip_eq", action="store_true", help="Skip put_tp==get_tp")
    parser.add_argument("--skip_gt", action="store_true", help="Skip put_tp>get_tp")
    parser.add_argument("--skip_lt", action="store_true", help="Skip put_tp<get_tp")
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Override: run only these cases, format 'put_tp,get_tp,split_dim' space-separated, e.g. '8,8,2 8,8,0 8,4,2'",
    )
    args = parser.parse_args()

    tp_sizes = [int(x) for x in args.tp_sizes.split(",")]
    split_dims = [int(x) for x in args.split_dims.split(",")]

    cases = []
    if args.cases:
        for s in args.cases.split():
            parts = [int(x) for x in s.split(",")]
            if len(parts) == 3:
                cases.append(tuple(parts))
    else:
        for put_tp in tp_sizes:
            for get_tp in tp_sizes:
                if put_tp == get_tp and args.skip_eq:
                    continue
                if put_tp > get_tp and args.skip_gt:
                    continue
                if put_tp < get_tp and args.skip_lt:
                    continue
                for split_dim in split_dims:
                    if split_dim >= 4:
                        continue
                    cases.append((put_tp, get_tp, split_dim))

    print(f"\n=== TP Reconstruction Test (multi-process, CCRP + All Gather control) ===")
    print(f"Put: put_tp processes, each put_tensor_chunk_with_tp(key, chunk)")
    print(f"Get (CCRP): get_tp processes, each get_tensor_with_tp_into(key, buffer)")
    print(f"Get (All Gather control): get_tp processes, each fetches all chunks, concats, slices")
    print(f"Single key, {args.size_mb} MB, iters={args.iters}")
    print(f"TP sizes: {tp_sizes}, split_dims: {split_dims}")
    print(f"Total cases: {len(cases)}")

    results_table = []
    max_tp = max(max(put_tp, get_tp) for put_tp, get_tp, _ in cases) if cases else max(tp_sizes)

    # Pre-create pool: workers start with their own mooncake client, avoid fork/spawn during test
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(max_tp, initializer=_init_worker_store)

    try:
        for i, (put_tp, get_tp, split_dim) in enumerate(cases):
            tensor = make_tensor_for_split_dim(split_dim, args.size_mb, max_tp)
            key = f"tp_{put_tp}_{get_tp}_{split_dim}_{i}_{uuid.uuid4()}"
            ok = run_one_case(
                pool, key, tensor, put_tp, get_tp, split_dim, args.iters, results_table
            )
            if not ok:
                print(f"  [FAIL] put_tp={put_tp} get_tp={get_tp} split_dim={split_dim}")
            elif (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(cases)}")
    finally:
        pool.close()
        pool.join()

    if args.json:
        import json
        out = []
        for r in results_table:
            o = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
            out.append(o)
        print(json.dumps(out, indent=2))
    else:
        print_table(results_table, fmt="csv" if args.csv else "text")
    print("\nDone.")


if __name__ == "__main__":
    main()
