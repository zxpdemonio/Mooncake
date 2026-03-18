#!/usr/bin/env python3
"""
TP reconstruction performance test. All put/get use zero-copy interfaces over RDMA.

Each iteration measures overlapped end-to-end makespan: put_tp writers and
get_tp readers are launched together, and readers retry until data becomes
visible.

Test groups (each has its own independent put + get):
1. CCRP: distributed put_tensor_chunk_with_tp_from + get_tensor_with_tp_into
2. All Gather: rank0 put_tensor_from(full tensor) + get_tensor_into(full) + local slice
3. Gather+TP: rank0 gather full tensor + single put_tensor_with_tp_from(full) + get_tensor_with_tp_into(fast path)
"""

import argparse
import ctypes
import multiprocessing
import struct
import time
import uuid

import numpy as np
import torch

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


def chunk_serialized_size(chunk):
    """Size in bytes of [TensorMetadata][data]."""
    return TENSOR_METADATA_SIZE + chunk.numel() * chunk.element_size()


def serialize_chunk_to_buffer(chunk, buf):
    """Serialize tensor to buffer [TensorMetadata][data]. Must be contiguous."""
    dtype_id = DTYPE_MAP.get(chunk.dtype, 0)
    ndim = chunk.ndim
    shape = list(chunk.shape) + [-1] * (4 - chunk.ndim)
    struct.pack_into("<iiqqqq", buf, 0, dtype_id, ndim, *shape[:4])
    data = chunk.cpu().numpy().tobytes()
    ctypes.memmove(ctypes.addressof(buf) + TENSOR_METADATA_SIZE, data, len(data))


def get_output_buffer_size(tensor, get_tp):
    """Buffer size for get_tensor_with_tp_into: meta + chunk data."""
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
    """Called once per pool worker process to create its own mooncake client."""
    global _worker_store
    _worker_store = create_store()


def make_tensor_for_split_dim(split_dim, size_mb, max_tp=8):
    """Create 4D tensor where split_dim is divisible by max_tp."""
    size_bytes = int(size_mb * 1024 * 1024)
    target_numel = size_bytes // 4
    other_dims = [32, 32, 32]
    prod_other = 32 * 32 * 32
    split_dim_size = target_numel // prod_other
    split_dim_size = max(max_tp, (split_dim_size // max_tp) * max_tp)
    shape = other_dims[:split_dim] + [split_dim_size] + other_dims[split_dim:]
    return torch.randn(*shape, dtype=torch.float32).contiguous()


# ── Worker functions (each runs in a separate process, all zero-copy) ──

def _put_worker(args):
    """CCRP put: put_tensor_chunk_with_tp_from (zero-copy, one chunk per rank)."""
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
            key, ptr, sz,
            tp_rank=put_rank, tp_size=put_tp, split_dim=split_dim
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
    """CCRP get: get_tensor_with_tp_into (zero-copy, cross-chunk reconstruction)."""
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


def _put_full_tensor_worker(args):
    """All Gather put: put_tensor_from (zero-copy, full tensor as single key)."""
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
    """All Gather get: get_tensor_into (zero-copy, read full tensor + local slice)."""
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
    """Gather+TP put: put_tensor_with_tp_from once with a full tensor buffer."""
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


def _timed_ok_result(ok, start_t, **extra):
    return {
        "ok": bool(ok),
        "total_s": time.perf_counter() - start_t,
        **extra,
    }


def _put_worker_timed(args):
    start_t = time.perf_counter()
    return _timed_ok_result(_put_worker(args), start_t)


def _put_worker_from_tensor_timed(args):
    start_t = time.perf_counter()
    return _timed_ok_result(_put_worker_from_tensor(args), start_t)


def _put_full_tensor_worker_timed(args):
    start_t = time.perf_counter()
    return _timed_ok_result(_put_full_tensor_worker(args), start_t)


def _put_gather_tp_worker_timed(args):
    start_t = time.perf_counter()
    return _timed_ok_result(_put_gather_tp_worker(args), start_t)


def _get_with_retry(get_fn, ready_fn, args, timeout_s=GET_RETRY_TIMEOUT_SEC,
                    interval_s=GET_RETRY_INTERVAL_SEC, return_stats=False):
    start_t = time.perf_counter()
    deadline = start_t + timeout_s
    ready_checks = 0
    while True:
        ready_checks += 1
        if ready_fn(*args):
            get_start_t = time.perf_counter()
            ok = get_fn(args)
            get_elapsed = time.perf_counter() - get_start_t
            if return_stats:
                return _timed_ok_result(
                    ok,
                    start_t,
                    wait_s=get_start_t - start_t,
                    get_s=get_elapsed,
                    ready_checks=ready_checks,
                )
            return ok
        if time.perf_counter() >= deadline:
            if return_stats:
                return _timed_ok_result(
                    False,
                    start_t,
                    wait_s=time.perf_counter() - start_t,
                    get_s=0.0,
                    ready_checks=ready_checks,
                )
            return False
        time.sleep(interval_s)


def _get_worker_retry(args):
    get_rank, key, tensor, get_tp, split_dim, put_tp = args
    return _get_with_retry(
        lambda inner_args: _get_worker(inner_args[:5]),
        lambda _get_rank, key_name, _tensor, _get_tp, _split_dim, put_tp_size: _ccrp_ready(_worker_store, key_name, put_tp_size),
        (get_rank, key, tensor, get_tp, split_dim, put_tp),
    )


def _get_worker_retry_timed(args):
    get_rank, key, tensor, get_tp, split_dim, put_tp = args
    return _get_with_retry(
        lambda inner_args: _get_worker(inner_args[:5]),
        lambda _get_rank, key_name, _tensor, _get_tp, _split_dim, put_tp_size: _ccrp_ready(_worker_store, key_name, put_tp_size),
        (get_rank, key, tensor, get_tp, split_dim, put_tp),
        return_stats=True,
    )


def _get_full_tensor_worker_retry(args):
    get_rank, key, tensor, get_tp, split_dim = args
    return _get_with_retry(
        _get_full_tensor_worker,
        lambda _get_rank, key_name, _tensor, _get_tp, _split_dim: _all_gather_ready(_worker_store, key_name),
        (get_rank, key, tensor, get_tp, split_dim),
    )


def _get_full_tensor_worker_retry_timed(args):
    get_rank, key, tensor, get_tp, split_dim = args
    return _get_with_retry(
        _get_full_tensor_worker,
        lambda _get_rank, key_name, _tensor, _get_tp, _split_dim: _all_gather_ready(_worker_store, key_name),
        (get_rank, key, tensor, get_tp, split_dim),
        return_stats=True,
    )


def _get_tp_chunk_worker_retry(args):
    get_rank, key, tensor, get_tp, split_dim = args
    return _get_with_retry(
        _get_tp_chunk_worker,
        lambda _get_rank, key_name, _tensor, get_tp_size, _split_dim: _gather_tp_ready(_worker_store, key_name, get_tp_size),
        (get_rank, key, tensor, get_tp, split_dim),
    )


def _get_tp_chunk_worker_retry_timed(args):
    get_rank, key, tensor, get_tp, split_dim = args
    return _get_with_retry(
        _get_tp_chunk_worker,
        lambda _get_rank, key_name, _tensor, get_tp_size, _split_dim: _gather_tp_ready(_worker_store, key_name, get_tp_size),
        (get_rank, key, tensor, get_tp, split_dim),
        return_stats=True,
    )


def _result_ok(result):
    if isinstance(result, dict):
        return bool(result.get("ok"))
    return bool(result)



def _run_overlapped_iteration(pool, put_fn, put_args, get_fn, get_args, return_results=False):
    put_async = pool.map_async(put_fn, put_args) if put_args else None
    get_async = pool.map_async(get_fn, get_args) if get_args else None
    put_results = put_async.get() if put_async is not None else []
    get_results = get_async.get() if get_async is not None else []
    if return_results:
        return put_results, get_results
    return all(_result_ok(r) for r in put_results), all(_result_ok(r) for r in get_results)


# ── Test runner ──

def _summarize_timed_results(results):
    if not results:
        return None
    totals = [r["total_s"] for r in results]
    waits = [r.get("wait_s", 0.0) for r in results]
    gets = [r.get("get_s", 0.0) for r in results]
    checks = [r.get("ready_checks", 0) for r in results]
    return {
        "max_total_ms": max(totals) * 1000,
        "mean_total_ms": float(np.mean(totals)) * 1000,
        "max_wait_ms": max(waits) * 1000,
        "mean_wait_ms": float(np.mean(waits)) * 1000,
        "max_get_ms": max(gets) * 1000,
        "mean_get_ms": float(np.mean(gets)) * 1000,
        "max_ready_checks": max(checks),
        "mean_ready_checks": float(np.mean(checks)),
    }



def _print_timing_breakdown(label, put_results, get_results, makespan_s, prep_ms=0.0):
    put_summary = _summarize_timed_results(put_results)
    get_summary = _summarize_timed_results(get_results)
    print(
        f"[TIMING] scheme={label} makespan_ms={makespan_s * 1000:.2f} "
        f"prep_ms={prep_ms:.2f} "
        f"put_max_ms={put_summary['max_total_ms']:.2f} put_mean_ms={put_summary['mean_total_ms']:.2f} "
        f"put_with_prep_max_ms={put_summary['max_total_ms'] + prep_ms:.2f} "
        f"put_with_prep_mean_ms={put_summary['mean_total_ms'] + prep_ms:.2f} "
        f"get_max_ms={get_summary['max_total_ms']:.2f} get_mean_ms={get_summary['mean_total_ms']:.2f} "
        f"wait_max_ms={get_summary['max_wait_ms']:.2f} wait_mean_ms={get_summary['mean_wait_ms']:.2f} "
        f"call_max_ms={get_summary['max_get_ms']:.2f} call_mean_ms={get_summary['mean_get_ms']:.2f} "
        f"ready_checks_max={get_summary['max_ready_checks']} ready_checks_mean={get_summary['mean_ready_checks']:.1f}"
    )



def run_one_case(pool, key, tensor, put_tp, get_tp, split_dim, iters, results_table,
                 run_all_gather=True, run_gather_tp=True, print_timing_breakdown=False):
    """Run one case with overlapped put/get timing per scheme."""
    tmp_store = create_store()
    tmp_store.remove_all()
    tmp_store.close()

    full_size = tensor.numel() * tensor.element_size()
    output_size = full_size // get_tp

    ccrp_times = []
    for it in range(iters):
        ccrp_key = f"{key}_ccrp_{it}"
        put_args = [(r, ccrp_key, tensor, put_tp, split_dim) for r in range(put_tp)]
        get_args = [(r, ccrp_key, tensor, get_tp, split_dim, put_tp) for r in range(get_tp)]
        prep_ms = 0.0
        t0 = time.perf_counter()
        if print_timing_breakdown:
            put_results, get_results = _run_overlapped_iteration(
                pool, _put_worker_from_tensor_timed, put_args, _get_worker_retry_timed, get_args, return_results=True
            )
            put_ok = all(_result_ok(r) for r in put_results)
            get_ok = all(_result_ok(r) for r in get_results)
        else:
            put_ok, get_ok = _run_overlapped_iteration(
                pool, _put_worker_from_tensor, put_args, _get_worker_retry, get_args
            )
        elapsed = time.perf_counter() - t0
        if put_ok and get_ok:
            ccrp_times.append(elapsed)
        if print_timing_breakdown:
            _print_timing_breakdown("ccrp", put_results, get_results, elapsed, prep_ms=prep_ms)

    all_gather_times = []
    if run_all_gather:
        for it in range(iters):
            ag_key = f"{key}_ag_{it}"
            prep_t0 = time.perf_counter()
            gathered_tensor = _simulate_gathered_tensor(tensor, put_tp, split_dim)
            prep_ms = (time.perf_counter() - prep_t0) * 1000
            put_args = [(ag_key, gathered_tensor)]
            get_args = [(r, ag_key, tensor, get_tp, split_dim) for r in range(get_tp)]
            t0 = time.perf_counter()
            if print_timing_breakdown:
                put_results, get_results = _run_overlapped_iteration(
                    pool, _put_full_tensor_worker_timed, put_args,
                    _get_full_tensor_worker_retry_timed, get_args, return_results=True
                )
                put_ok = all(_result_ok(r) for r in put_results)
                get_ok = all(_result_ok(r) for r in get_results)
            else:
                put_ok, get_ok = _run_overlapped_iteration(
                    pool, _put_full_tensor_worker, put_args,
                    _get_full_tensor_worker_retry, get_args
                )
            elapsed = time.perf_counter() - t0
            if put_ok and get_ok:
                all_gather_times.append(elapsed)
            if print_timing_breakdown:
                _print_timing_breakdown("all_gather", put_results, get_results, elapsed, prep_ms=prep_ms)

    gather_tp_times = []
    if run_gather_tp:
        for it in range(iters):
            gtp_key = f"{key}_gtp_{it}"
            prep_t0 = time.perf_counter()
            gathered_tensor = _simulate_gathered_tensor(tensor, put_tp, split_dim)
            prep_ms = (time.perf_counter() - prep_t0) * 1000
            put_args = [(gtp_key, gathered_tensor, get_tp, split_dim)]
            get_args = [(r, gtp_key, tensor, get_tp, split_dim) for r in range(get_tp)]
            t0 = time.perf_counter()
            if print_timing_breakdown:
                put_results, get_results = _run_overlapped_iteration(
                    pool, _put_gather_tp_worker_timed, put_args,
                    _get_tp_chunk_worker_retry_timed, get_args, return_results=True
                )
                put_ok = all(_result_ok(r) for r in put_results)
                get_ok = all(_result_ok(r) for r in get_results)
            else:
                put_ok, get_ok = _run_overlapped_iteration(
                    pool, _put_gather_tp_worker, put_args,
                    _get_tp_chunk_worker_retry, get_args
                )
            elapsed = time.perf_counter() - t0
            if put_ok and get_ok:
                gather_tp_times.append(elapsed)
            if print_timing_breakdown:
                _print_timing_breakdown("gather_tp", put_results, get_results, elapsed, prep_ms=prep_ms)

    ag_mean = np.mean(all_gather_times) if all_gather_times else None
    ccrp_mean = np.mean(ccrp_times) if ccrp_times else None
    gtp_mean = np.mean(gather_tp_times) if gather_tp_times else None
    ccrp_total_mb = (2 * full_size) / 1e6
    ag_total_mb = ((get_tp + 1) * full_size) / 1e6
    gtp_total_mb = (2 * full_size) / 1e6
    row = {
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
    results_table.append(row)
    required_ok = bool(ccrp_times)
    if run_all_gather:
        required_ok = required_ok and bool(all_gather_times)
    if run_gather_tp:
        required_ok = required_ok and bool(gather_tp_times)
    return required_ok


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

    print("\n" + "=" * 115)
    print(
        f"{'put_tp':>6} {'get_tp':>6} {'split':>5} | "
        f"{'ag_ms':>8} {'ag_MB/s':>10} | "
        f"{'gtp_ms':>8} {'gtp_MB/s':>10} | "
        f"{'ccrp_ms':>8} {'ccrp_MB/s':>10} | "
        f"{'ag/ccrp':>8} {'gtp/ccrp':>8}"
    )
    print("-" * 115)
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
    parser = argparse.ArgumentParser(description="TP reconstruction perf test (RDMA)")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--size_mb", type=float, default=512)
    parser.add_argument("--tp_sizes", type=str, default="4,8")
    parser.add_argument("--split_dims", type=str, default="0")
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    parser.add_argument("--timing-breakdown", action="store_true",
                        help="Print per-scheme timing breakdown for put/get overlap analysis")
    parser.add_argument(
        "--cases", type=str, default=None,
        help="Override: 'put_tp,get_tp,split_dim' space-separated, e.g. '8,4,0 4,8,0'",
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
                for split_dim in split_dims:
                    if split_dim >= 4:
                        continue
                    cases.append((put_tp, get_tp, split_dim))

    max_split_tp = max(max(pt, gt) for pt, gt, _ in cases) if cases else max(tp_sizes)
    pool_workers = max((pt + gt) for pt, gt, _ in cases) if cases else max(tp_sizes)

    print(f"\n=== TP Reconstruction Test (RDMA, all zero-copy) ===")
    print(f"  1. CCRP: put_tensor_chunk_with_tp_from + get_tensor_with_tp_into")
    print(f"  2. All Gather: put_tensor_from(full) + get_tensor_into(full) + local slice")
    print(f"  3. Gather+TP: put_tensor_with_tp_from + get_tensor_with_tp_into(fast path)")
    print(f"size_mb={args.size_mb}, iters={args.iters}, cases={len(cases)}")

    results_table = []

    # Each pool worker is a separate process with its own mooncake RDMA client
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(pool_workers, initializer=_init_worker_store)

    try:
        for i, (put_tp, get_tp, split_dim) in enumerate(cases):
            tensor = make_tensor_for_split_dim(split_dim, args.size_mb, max_split_tp)
            key = f"tp_{put_tp}_{get_tp}_{split_dim}_{i}_{uuid.uuid4()}"
            ok = run_one_case(
                pool,
                key,
                tensor,
                put_tp,
                get_tp,
                split_dim,
                args.iters,
                results_table,
                print_timing_breakdown=args.timing_breakdown,
            )
            if not ok:
                print(f"  [FAIL] put_tp={put_tp} get_tp={get_tp} split_dim={split_dim}")
            elif (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(cases)}")
    finally:
        pool.close()
        pool.join()

    print_table(results_table, fmt="csv" if args.csv else "text")
    print("\nDone.")


if __name__ == "__main__":
    main()
