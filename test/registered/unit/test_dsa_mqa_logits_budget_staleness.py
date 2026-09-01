"""The MQA-logits budget must not decide from a stale free-memory reading.

The budget was one mem_get_info reading, taken on the first non-capture
prefill -- which lands while free memory is collapsing as the pools fill. Which
second that read landed in decided every later chunking verdict, so the same
configuration behaved three ways: too large a budget skipped chunking and the
allocation aborted the scheduler with HSA_STATUS_ERROR_OUT_OF_RESOURCES, too
small a budget chunked everything and ran ~2x slower for the life of the
process, and in between it was fine.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

GiB = 1 << 30


class _Probe:
    """Just enough of Indexer to exercise the budget path."""

    _MQA_LOGITS_BYTES_PER_ELEM = Indexer._MQA_LOGITS_BYTES_PER_ELEM
    _MQA_LOGITS_STATIC_SKIP_ELEMS = Indexer._MQA_LOGITS_STATIC_SKIP_ELEMS
    _MQA_LOGITS_BUDGET_TTL_S = Indexer._MQA_LOGITS_BUDGET_TTL_S
    _should_chunk_mqa_logits = Indexer._should_chunk_mqa_logits
    _get_mqa_logits_budget_bytes = Indexer._get_mqa_logits_budget_bytes

    def __init__(self, cached, read_at):
        self._mqa_logits_budget_bytes = {0: cached}
        self._mqa_logits_budget_read_at = {0: read_at}

    @staticmethod
    def _mqa_logits_free_mem_fraction():
        return 1.0


def _elems_for(nbytes):
    return nbytes // Indexer._MQA_LOGITS_BYTES_PER_ELEM


class TestMqaLogitsBudgetStaleness(unittest.TestCase):
    def test_within_the_ttl_the_cached_figure_stands_and_costs_no_sync(self):
        with mock.patch("time.monotonic", return_value=100.0):
            p = _Probe(cached=8 * GiB, read_at=99.5)
            with mock.patch.object(torch.cuda, "mem_get_info") as m:
                need, budget = p._should_chunk_mqa_logits(1, _elems_for(GiB), 0)
        m.assert_not_called()
        self.assertFalse(need)
        self.assertEqual(budget, 8 * GiB)

    def test_a_budget_read_too_high_is_corrected_down_and_chunks(self):
        """The crash direction: 39 GiB read, 1 GiB actually left."""
        with mock.patch("time.monotonic", return_value=100.0):
            p = _Probe(cached=39 * GiB, read_at=90.0)
            with mock.patch.object(
                torch.cuda, "mem_get_info", return_value=(GiB, 288 * GiB)
            ) as m:
                need, budget = p._should_chunk_mqa_logits(1, _elems_for(4 * GiB), 0)
        m.assert_called_once()
        self.assertTrue(need, "4 GiB of logits must chunk when 1 GiB is left")
        self.assertEqual(budget, GiB)

    def test_a_budget_read_too_low_is_corrected_up_and_stops_chunking(self):
        """The slowdown direction: 0.5 GiB read during the collapse, 16 GiB now."""
        with mock.patch("time.monotonic", return_value=100.0):
            p = _Probe(cached=GiB // 2, read_at=90.0)
            with mock.patch.object(
                torch.cuda, "mem_get_info", return_value=(16 * GiB, 288 * GiB)
            ):
                need, budget = p._should_chunk_mqa_logits(1, _elems_for(4 * GiB), 0)
        self.assertFalse(need, "must stop chunking once the room is really there")
        self.assertEqual(budget, 16 * GiB)

    def test_small_requests_skip_the_budget_entirely(self):
        with mock.patch("time.monotonic", return_value=100.0):
            p = _Probe(cached=GiB, read_at=0.0)
            with mock.patch.object(torch.cuda, "mem_get_info") as m:
                need, budget = p._should_chunk_mqa_logits(1, 1000, 0)
        m.assert_not_called()
        self.assertFalse(need)
        self.assertEqual(budget, 0)


if __name__ == "__main__":
    unittest.main()
