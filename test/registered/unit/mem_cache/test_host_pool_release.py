"""On ROCm the host KV pool must be handed back at destroy, not at process exit."""

import pytest
import torch

from sglang.srt.utils import is_hip

pytestmark = pytest.mark.skipif(not is_hip(), reason="the pinned-cache path is ROCm-only")


def _mem_free_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) / 1048576
    raise RuntimeError("MemFree missing")


def test_torch_caches_pinned_memory_until_the_cache_is_emptied():
    """The premise of the fix: dropping the tensor does not hipHostFree it."""
    assert hasattr(torch._C, "_host_emptyCache"), "torch too old for the fix"
    torch._C._host_emptyCache()
    before = _mem_free_gb()
    buf = torch.empty(4 * 1024**3, dtype=torch.uint8, pin_memory=True)
    buf[::4096] = 1  # fault the pages in
    held = _mem_free_gb()
    assert before - held > 3.0, f"expected ~4 GB pinned, saw {before - held:.2f} GB"

    del buf
    after_del = _mem_free_gb()
    assert before - after_del > 3.0, (
        "torch returned the pages on del, so the caching allocator changed and "
        "HostKVCache.destroy no longer needs to empty the cache"
    )

    torch._C._host_emptyCache()
    after_empty = _mem_free_gb()
    assert before - after_empty < 1.0, (
        f"emptying the cache left {before - after_empty:.2f} GB pinned"
    )


def test_destroy_empties_the_pinned_cache():
    from sglang.srt.mem_cache.pool_host.base import HostKVCache

    calls = []
    real = torch._C._host_emptyCache
    torch._C._host_emptyCache = lambda: calls.append(1)
    try:
        stub = type("Stub", (), {})()
        stub.kv_buffer = None
        stub.pin_memory = True
        HostKVCache.destroy(stub)
        assert calls, "destroy did not empty torch's pinned cache"
        assert stub.kv_buffer is None
        HostKVCache.destroy(stub)
        assert len(calls) == 1, "destroy is not idempotent"
    finally:
        torch._C._host_emptyCache = real
