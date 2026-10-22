"""destroy() must drop every reference to the host pool, not just kv_buffer."""

import pytest
import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.utils import is_hip

pytestmark = pytest.mark.skipif(not is_hip(), reason="the pinned-cache path is ROCm-only")


def _mem_free_gb():
    """This process's own resident size, not system MemFree: anything else on
    the box moves MemFree by more than the amount under test. Returned negated
    so callers can keep reading it as "free", i.e. before - now > 0 means the
    process grew.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return -int(line.split()[1]) / 1048576
    raise RuntimeError("VmRSS missing")


def test_torch_caches_pinned_memory_until_the_cache_is_emptied():
    """The premise: dropping the tensor does not hipHostFree it."""
    assert hasattr(torch._C, "_host_emptyCache"), "torch too old for the fix"
    torch._C._host_emptyCache()
    before = _mem_free_gb()
    buf = torch.empty(4 * 1024**3, dtype=torch.uint8, pin_memory=True)
    buf[::4096] = 1
    assert before - _mem_free_gb() > 3.0
    del buf
    assert before - _mem_free_gb() > 3.0, (
        "torch returned the pages on del, so the caching allocator changed"
    )
    torch._C._host_emptyCache()
    assert before - _mem_free_gb() < 1.0


def _stub(**attrs):
    s = type("Stub", (), {})()
    s.pin_memory = True
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def test_page_first_layer_views_are_released():
    """MLATokenToKVPoolHost keeps one strided view per layer in data_refs; each
    pins the same storage, so clearing kv_buffer alone frees nothing."""
    torch._C._host_emptyCache()
    before = _mem_free_gb()
    buf = torch.empty((4 * 1024**3 // 2, 2), dtype=torch.uint8, pin_memory=True)
    buf[::4096] = 1
    stub = _stub(kv_buffer=buf, data_refs=[buf.transpose(0, 1)[i] for i in range(2)])
    del buf
    assert before - _mem_free_gb() > 3.0, "test did not actually pin 4 GB"
    HostKVCache.destroy(stub)
    assert stub.data_refs is None, "destroy left the per-layer views in place"
    assert before - _mem_free_gb() < 1.0, "the pool was not handed back"


def test_the_indexer_buffer_is_released():
    """DSAIndexerPoolHost never assigns kv_buffer; its pool is
    index_k_with_scale_buffer, which destroy() used to ignore entirely."""
    torch._C._host_emptyCache()
    before = _mem_free_gb()
    buf = torch.empty(4 * 1024**3, dtype=torch.uint8, pin_memory=True)
    buf[::4096] = 1
    stub = _stub(index_k_with_scale_buffer=buf)
    del buf
    assert before - _mem_free_gb() > 3.0
    HostKVCache.destroy(stub)
    assert before - _mem_free_gb() < 1.0, "the indexer pool was not handed back"


def test_destroy_is_idempotent():
    stub = _stub(kv_buffer=None, data_refs=None)
    HostKVCache.destroy(stub)
    HostKVCache.destroy(stub)
    assert stub._destroyed
