#!/usr/bin/env python3
"""
Comprehensive TP reconstruction performance test.

Covers:
- split_dim: 0, 1, 2, 3
- put_tp / get_tp: 2, 4, 8
- Relations: put_tp > get_tp, put_tp == get_tp, put_tp < get_tp
- Methods: Direct get (when put_tp==get_tp), All gather, CCRP
"""

import argparse
import ctypes
import sys
import time

TENSOR_METADATA_SIZE = 4 + 4 + 8 * 4  # 40 bytes


def create_store():
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


def get_tp_key(base_key, rank):
    return f"{base_key}_tp_{rank}"


def make_tensor_for_split_dim(split_dim, size_mb, max_tp=8):
    """Create 4D tensor where split_dim dimension is divisible by max_tp."""
    size_bytes = int(size_mb * 1024 * 1024)
    target_numel = size_bytes // 4
    # 4D [d0,d1,d2,d3], split_dim-th dim divisible by max_tp
    other_dims = [32, 32, 32]
    prod_other = np.prod(other_dims)
    split_dim_size = target_numel // prod_other
    split_dim_size = max(max_tp, (split_dim_size // max_tp) * max_tp)
    shape = other_dims[:split_dim] + [split_dim_size] + other_dims[split_dim:]
    return torch.randn(*shape, dtype=torch.float32).contiguous()


def reconstruct_direct(store, base_keys, tp_size, split_dim, get_rank):
    """Direct get when put_tp == get_tp: each rank gets its chunk."""
    keys = [get_tp_key(k, get_rank) for k in base_keys]
    bufs = store.batch_get_buffer(keys)
    if not bufs or any(b is None for b in bufs):
        return None
    return [bytes(b) for b in bufs]


def reconstruct_all_gather(store, base_keys, put_tp, get_tp, split_dim, get_rank):
    """All gather: fetch all put chunks, concat, extract slice."""
    chunk_keys = []
    for k in base_keys:
        for r in range(put_tp):
            chunk_keys.append(get_tp_key(k, r))
    buffers = store.batch_get_buffer(chunk_keys)
    if not buffers or any(b is None for b in buffers):
        return None
    import struct
    results = []
    for key_idx in range(len(base_keys)):
        chunks = [
            bytes(buffers[key_idx * put_tp + r])[TENSOR_METADATA_SIZE:]
            for r in range(put_tp)
        ]
        full_data = b"".join(chunks)
        full_numel = len(full_data) // 4
        step = full_numel // get_tp
        start, end = get_rank * step, (get_rank + 1) * step
        slice_data = full_data[start * 4 : end * 4]
        meta = bytearray(bytes(buffers[key_idx * put_tp])[:TENSOR_METADATA_SIZE])
        shape = list(struct.unpack_from("<" + "q" * 4, meta, 8))
        shape[split_dim] = shape[split_dim] // get_tp
        struct.pack_into("<" + "q" * 4, meta, 8, *shape)
        results.append(bytes(meta) + slice_data)
    return results


def reconstruct_ccrp(store, base_keys, put_tp, get_tp, split_dim, get_rank):
    """CCRP: batch_get_tensor_with_tp - uses store metadata (put_tp/size/offset)."""
    tensors = store.batch_get_tensor_with_tp(
        base_keys, tp_rank=get_rank, tp_size=get_tp, split_dim=split_dim
    )
    if not tensors or any(t is None for t in tensors):
        return None
    return tensors


def run_one_case(
    store, base_keys, tensors, put_tp, get_tp, split_dim, iters, results_table
):
    """Run one (put_tp, get_tp, split_dim) case and record results."""
    store.remove_all()
    rc = store.batch_put_tensor_with_tp(
        base_keys, tensors, tp_size=put_tp, split_dim=split_dim
    )
    if not all(r == 0 for r in rc):
        return False

    full_size = sum(t.numel() * t.element_size() for t in tensors)
    output_size = full_size // get_tp

    row = {
        "put_tp": put_tp,
        "get_tp": get_tp,
        "split_dim": split_dim,
        "full_mb": full_size / 1e6,
        "output_mb": output_size / 1e6,
        "direct_ms": None,
        "all_gather_ms": None,
        "ccrp_ms": None,
        "direct_mbps": None,
        "all_gather_mbps": None,
        "ccrp_mbps": None,
    }

    # Direct (only when put_tp == get_tp)
    if put_tp == get_tp:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for rank in range(get_tp):
                r = reconstruct_direct(store, base_keys, get_tp, split_dim, rank)
                if r is None:
                    break
            else:
                times.append(time.perf_counter() - t0)
        if times:
            row["direct_ms"] = np.mean(times) * 1000
            # Direct: each rank reads output_size, total = full_size
            row["direct_mbps"] = (full_size / 1e6) / np.mean(times)

    # All gather: ALWAYS run as baseline for comparison (both put_tp==get_tp and put_tp!=get_tp)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for rank in range(get_tp):
            r = reconstruct_all_gather(
                store, base_keys, put_tp, get_tp, split_dim, rank
            )
            if r is None:
                break
        else:
            times.append(time.perf_counter() - t0)
    if times:
        row["all_gather_ms"] = np.mean(times) * 1000
        # All gather: each rank reads full_size, total = get_tp * full_size
        row["all_gather_mbps"] = (get_tp * full_size / 1e6) / np.mean(times)

    # CCRP: run for all cases (when put_tp==get_tp, CCRP equiv to Direct; when put_tp!=get_tp, CCRP avoids read amplification)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for rank in range(get_tp):
            r = reconstruct_ccrp(
                store, base_keys, put_tp, get_tp, split_dim, rank
            )
            if r is None:
                break
        else:
            times.append(time.perf_counter() - t0)
    if times:
        row["ccrp_ms"] = np.mean(times) * 1000
        # CCRP: each rank reads output_size, total = full_size
        row["ccrp_mbps"] = (full_size / 1e6) / np.mean(times)

    results_table.append(row)
    return True


def print_table(results_table, fmt="text"):
    """Print results as table."""
    if fmt == "csv":
        print(
            "put_tp,get_tp,split_dim,rel,full_mb,output_mb,"
            "direct_ms,direct_mbps,all_gather_ms,all_gather_mbps,ccrp_ms,ccrp_mbps"
        )
        for r in results_table:
            rel = (
                "eq"
                if r["put_tp"] == r["get_tp"]
                else ("gt" if r["put_tp"] > r["get_tp"] else "lt")
            )
            print(
                f"{r['put_tp']},{r['get_tp']},{r['split_dim']},{rel},"
                f"{r['full_mb']:.2f},{r['output_mb']:.2f},"
                f"{r['direct_ms'] or ''},{r['direct_mbps'] or ''},"
                f"{r['all_gather_ms'] or ''},{r['all_gather_mbps'] or ''},"
                f"{r['ccrp_ms'] or ''},{r['ccrp_mbps'] or ''}"
            )
        return

    # Text table
    print("\n" + "=" * 100)
    print(
        f"{'put_tp':>6} {'get_tp':>6} {'split':>5} {'rel':>3} | "
        f"{'direct_ms':>9} {'direct_MB/s':>10} | "
        f"{'all_gather_ms':>12} {'all_gather_MB/s':>14} | "
        f"{'ccrp_ms':>8} {'ccrp_MB/s':>10} | speedup"
    )
    print("-" * 100)

    for r in results_table:
        rel = (
            "=="
            if r["put_tp"] == r["get_tp"]
            else (">" if r["put_tp"] > r["get_tp"] else "<")
        )
        d_ms = f"{r['direct_ms']:.1f}" if r["direct_ms"] is not None else "-"
        d_mb = f"{r['direct_mbps']:.0f}" if r["direct_mbps"] is not None else "-"
        ag_ms = f"{r['all_gather_ms']:.1f}" if r["all_gather_ms"] is not None else "-"
        ag_mb = f"{r['all_gather_mbps']:.0f}" if r["all_gather_mbps"] is not None else "-"
        c_ms = f"{r['ccrp_ms']:.1f}" if r["ccrp_ms"] is not None else "-"
        c_mb = f"{r['ccrp_mbps']:.0f}" if r["ccrp_mbps"] is not None else "-"

        speedup = ""
        if r["ccrp_ms"] and r["all_gather_ms"] and r["ccrp_ms"] > 0:
            sp = r["all_gather_ms"] / r["ccrp_ms"]
            speedup = f"{sp:.2f}x" if sp > 1 else f"1/{1/sp:.2f}x"

        print(
            f"{r['put_tp']:>6} {r['get_tp']:>6} {r['split_dim']:>5} {rel:>3} | "
            f"{d_ms:>9} {d_mb:>10} | "
            f"{ag_ms:>12} {ag_mb:>14} | "
            f"{c_ms:>8} {c_mb:>10} | {speedup}"
        )


def _run_demo(args):
    """Output synthetic demo data without connecting to store."""
    tp_sizes = [int(x) for x in args.tp_sizes.split(",")]
    split_dims = [int(x) for x in args.split_dims.split(",")]
    full_mb = args.num_tensors * args.size_mb
    results_table = []
    for put_tp in tp_sizes:
        for get_tp in tp_sizes:
            for split_dim in split_dims:
                if split_dim >= 4:
                    continue
                output_mb = full_mb / get_tp
                row = {
                    "put_tp": put_tp,
                    "get_tp": get_tp,
                    "split_dim": split_dim,
                    "full_mb": full_mb,
                    "output_mb": output_mb,
                    "direct_ms": None,
                    "all_gather_ms": None,
                    "ccrp_ms": None,
                    "direct_mbps": None,
                    "all_gather_mbps": None,
                    "ccrp_mbps": None,
                }
                if put_tp == get_tp:
                    row["direct_ms"] = 15 + split_dim * 2 + put_tp
                    row["direct_mbps"] = (full_mb * 1e6 / 1e6) / (
                        row["direct_ms"] / 1000
                    )
                # All gather: always present as baseline for comparison
                if put_tp == get_tp:
                    ag_ms = 50 + split_dim * 3 + put_tp * 2  # all-gather reads full per rank
                else:
                    ag_ms = 45 + split_dim * 5 + abs(put_tp - get_tp) * 3
                row["all_gather_ms"] = ag_ms
                row["all_gather_mbps"] = (get_tp * full_mb) / (ag_ms / 1000)
                # CCRP: always present
                ccrp_ms = 22 + split_dim * 3 + (abs(put_tp - get_tp) if put_tp != get_tp else 0) * 2
                row["ccrp_ms"] = ccrp_ms
                row["ccrp_mbps"] = full_mb / (ccrp_ms / 1000)
                results_table.append(row)
    print(f"\n=== TP Reconstruction Comprehensive Test (DEMO - synthetic data) ===")
    print(f"Tensors: {args.num_tensors} x {args.size_mb} MB")
    print(f"TP sizes: {tp_sizes}, split_dims: {split_dims}")
    print(f"Total cases: {len(results_table)}\n")
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


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive TP reconstruction test"
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--size_mb", type=float, default=4)
    parser.add_argument("--num_tensors", type=int, default=2)
    parser.add_argument("--tp_sizes", type=str, default="2,4,8")
    parser.add_argument("--split_dims", type=str, default="0,1,2,3")
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--skip_eq", action="store_true", help="Skip put_tp==get_tp")
    parser.add_argument("--skip_gt", action="store_true", help="Skip put_tp>get_tp")
    parser.add_argument("--skip_lt", action="store_true", help="Skip put_tp<get_tp")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Output synthetic demo data (no store connection)",
    )
    args = parser.parse_args()

    if args.demo:
        _run_demo(args)
        return

    import numpy as np
    import torch
    from mooncake.store import MooncakeDistributedStore
    from mooncake.mooncake_config import MooncakeConfig
    globals().update(np=np, torch=torch, MooncakeDistributedStore=MooncakeDistributedStore, MooncakeConfig=MooncakeConfig)
    tp_sizes = [int(x) for x in args.tp_sizes.split(",")]
    split_dims = [int(x) for x in args.split_dims.split(",")]

    store = create_store()

    # Build test matrix
    cases = []
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

    print(f"\n=== TP Reconstruction Comprehensive Test ===")
    print(f"Tensors: {args.num_tensors} x {args.size_mb} MB, iters={args.iters}")
    print(f"TP sizes: {tp_sizes}, split_dims: {split_dims}")
    print(f"Total cases: {len(cases)}")

    results_table = []
    max_tp = max(tp_sizes)

    for i, (put_tp, get_tp, split_dim) in enumerate(cases):
        tensors = [
            make_tensor_for_split_dim(split_dim, args.size_mb, max_tp)
            for _ in range(args.num_tensors)
        ]
        base_keys = [
            f"tp_{put_tp}_{get_tp}_{split_dim}_{i}_{int(time.time())}"
            for i in range(args.num_tensors)
        ]
        ok = run_one_case(
            store, base_keys, tensors, put_tp, get_tp, split_dim, args.iters, results_table
        )
        if not ok:
            print(f"  [FAIL] put_tp={put_tp} get_tp={get_tp} split_dim={split_dim}")
        elif (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(cases)}")

    if args.json:
        import json
        out = []
        for r in results_table:
            o = {}
            for k, v in r.items():
                if v is not None and isinstance(v, float):
                    o[k] = round(v, 2)
                else:
                    o[k] = v
            out.append(o)
        print(json.dumps(out, indent=2))
    else:
        print_table(results_table, fmt="csv" if args.csv else "text")
    store.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
