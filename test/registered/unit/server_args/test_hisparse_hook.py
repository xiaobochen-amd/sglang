import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.hisparse_hook import validate_hisparse
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseHook(unittest.TestCase):
    @patch(
        "sglang.srt.configs.model_config.is_deepseek_dsa",
        return_value=False,
    )
    @patch(
        "sglang.srt.configs.model_config.is_deepseek_v4",
        return_value=True,
    )
    @patch(
        "sglang.srt.arg_groups.hisparse_hook.get_platform",
        return_value=SimpleNamespace(is_hip=True),
    )
    def test_dsv4_pd_prefill_is_rejected(
        self, _mock_get_platform, _mock_is_v4, _mock_is_dsa
    ):
        hf_config = object()
        server_args = SimpleNamespace(
            enable_hisparse=True,
            disable_radix_cache=True,
            disaggregation_mode="prefill",
            get_model_config=lambda: SimpleNamespace(hf_config=hf_config),
        )

        with (
            patch(
                "sglang.srt.arg_groups.hisparse_hook.model_config_of",
                return_value=SimpleNamespace(hf_config=hf_config),
            ),
            self.assertRaisesRegex(ValueError, "decode-only"),
        ):
            validate_hisparse(server_args)


if __name__ == "__main__":
    unittest.main()
