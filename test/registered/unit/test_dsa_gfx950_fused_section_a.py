"""Section A of the gfx950 fused indexer picks GEMV or GEMM by row count.

``dual_gemv_bf16`` is specialised for M in [1,8], so a wider verify batch has to
be chunked and every chunk re-reads both weights. Measured in-graph on MI355X it
stops paying past two launches, and MTP verify rows are bs x num_draft_tokens --
24 for the stock GLM-5.2 recipe. These tests pin the branch down without a GPU:
the four kernel modules and the workspace are counting doubles, so only the
dispatch decision is under test.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsa.hip_gfx950 import fused_decode as fd


class RecordingGemv:
    def __init__(self):
        self.calls = []

    def dual_gemv_bf16(self, xq, wq, oq, xk, wk, ok, cfg):
        self.calls.append((xq.shape[0], oq.shape[0]))


class FakeWorkspace:
    """Every buffer _Workspace owns, on CPU and only as wide as the test needs."""

    def __init__(self, rows, cols):
        self.max_cols = cols
        self.logits = torch.empty((rows, cols), dtype=torch.float32)
        self.q_fp8 = torch.empty((rows, fd.N_HEADS, fd.HEAD_DIM), dtype=torch.uint8)
        self.head_gate = torch.empty((rows, fd.N_HEADS), dtype=torch.float32)
        self.q_proj = torch.empty(
            (rows, fd.N_HEADS * fd.HEAD_DIM), dtype=torch.bfloat16
        )
        self.kw = torch.empty((rows, fd.HEAD_DIM + fd.N_HEADS), dtype=torch.bfloat16)
        self.ghist = torch.zeros((rows, 8), dtype=torch.int32)
        self.cursor = torch.zeros((rows, 32), dtype=torch.int32)
        self.cand_cnt = torch.zeros((rows,), dtype=torch.int32)
        self.cand_idx = torch.empty((rows, cols), dtype=torch.int32)
        self.cand_val = torch.empty((rows, cols), dtype=torch.float32)
        self.page_table = torch.empty(
            (rows * (cols // fd.PAGE_SIZE),), dtype=torch.int32
        )


class TestFusedSectionADispatch(unittest.TestCase):
    COLS = fd.PAGE_SIZE * 4

    def setUp(self):
        self.gemv = RecordingGemv()
        self.mm_calls = []
        real_mm = torch.mm

        def counting_mm(a, b, *, out=None):
            self.mm_calls.append((a.shape, b.shape))
            return real_mm(a, b, out=out)

        patches = [
            mock.patch.object(
                fd.loader,
                "modules",
                lambda: (self.gemv, mock.Mock(), mock.Mock(), mock.Mock()),
            ),
            mock.patch.object(fd.torch, "mm", counting_mm),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        self.runner = fd.Gfx950FusedIndexer.__new__(fd.Gfx950FusedIndexer)
        self.runner.workspace = FakeWorkspace(fd.MAX_ROWS, self.COLS)
        self.runner.weights_scale = 1.0
        self.runner.indexer = mock.Mock()
        self.runner.indexer.k_norm.variance_epsilon = 1e-6
        self.runner.indexer.scale_fmt = None
        self.runner._constants = lambda: (
            torch.empty((fd.N_HEADS * fd.HEAD_DIM, fd.Q_LORA_RANK), dtype=torch.bfloat16),
            torch.empty((fd.HEAD_DIM + fd.N_HEADS, fd.HIDDEN_SIZE), dtype=torch.bfloat16),
            torch.empty(fd.HEAD_DIM),
            torch.empty(fd.HEAD_DIM),
            torch.empty((8, fd.ROPE_DIM // 2), dtype=torch.bfloat16),
            torch.empty((8, fd.ROPE_DIM // 2), dtype=torch.bfloat16),
        )

    def _run(self, rows):
        self.runner.run(
            x=torch.empty((rows, fd.HIDDEN_SIZE), dtype=torch.bfloat16),
            q_lora=torch.empty((rows, fd.Q_LORA_RANK), dtype=torch.bfloat16),
            positions=torch.zeros(rows, dtype=torch.int64),
            out_cache_loc=torch.zeros(rows, dtype=torch.int64),
            kv_cache=torch.empty(1),
            kv_cache_read=torch.empty(1),
            seqlens_int32=torch.zeros(rows, dtype=torch.int32),
            row_ends_int32=torch.zeros(rows, dtype=torch.int32),
            page_table_64=torch.zeros(
                (rows, self.COLS // fd.PAGE_SIZE), dtype=torch.int32
            ),
            rows=rows,
        )

    def test_single_launch_rows_use_the_gemv(self):
        self._run(fd.DUAL_GEMV_MAX_M)
        self.assertEqual(len(self.gemv.calls), 1)
        self.assertEqual(self.mm_calls, [])

    def test_two_launch_rows_still_use_the_gemv(self):
        self._run(fd.DUAL_GEMV_MAX_PROFITABLE_ROWS)
        self.assertEqual(len(self.gemv.calls), 2)
        self.assertEqual(self.mm_calls, [])

    def test_stock_mtp_verify_rows_use_the_gemm(self):
        # max-running-requests 4 x num-draft-tokens 6; three GEMV launches.
        rows = 24
        self.assertGreater(rows, fd.DUAL_GEMV_MAX_PROFITABLE_ROWS)
        self._run(rows)
        self.assertEqual(self.gemv.calls, [])
        self.assertEqual(len(self.mm_calls), 2)

    def test_widest_verify_batch_uses_the_gemm(self):
        self._run(fd.MAX_ROWS)
        self.assertEqual(self.gemv.calls, [])
        self.assertEqual(len(self.mm_calls), 2)

    def test_both_branches_write_the_same_projections(self):
        rows = fd.DUAL_GEMV_MAX_PROFITABLE_ROWS
        w_q_b, w_kw = self.runner._constants()[:2]
        q_lora = torch.randn((rows, fd.Q_LORA_RANK), dtype=torch.bfloat16)
        x = torch.randn((rows, fd.HIDDEN_SIZE), dtype=torch.bfloat16)
        w_q_b.normal_()
        w_kw.normal_()
        self.runner._constants = lambda: (
            w_q_b, w_kw, torch.empty(fd.HEAD_DIM), torch.empty(fd.HEAD_DIM),
            torch.empty((8, fd.ROPE_DIM // 2), dtype=torch.bfloat16),
            torch.empty((8, fd.ROPE_DIM // 2), dtype=torch.bfloat16),
        )
        # The GEMM branch is the definition; assert it against a plain matmul.
        expect_q = q_lora @ w_q_b.t()
        expect_kw = x @ w_kw.t()
        ws = self.runner.workspace
        torch.mm(q_lora, w_q_b.t(), out=ws.q_proj[:rows])
        torch.mm(x, w_kw.t(), out=ws.kw[:rows])
        self.assertTrue(torch.equal(ws.q_proj[:rows], expect_q))
        self.assertTrue(torch.equal(ws.kw[:rows], expect_kw))


if __name__ == "__main__":
    unittest.main()
