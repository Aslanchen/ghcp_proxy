import unittest
from types import SimpleNamespace

import excel_upstream
from proxy_client_config import ProxyClientConfigService, _model_token_pricing_description


class ReasoningLevelTests(unittest.TestCase):
    def setUp(self):
        self.service = object.__new__(ProxyClientConfigService)

    def _effort_names(self, model_name, raw_efforts):
        levels, _ = self.service._resolve_reasoning_levels(
            "gpt", raw_efforts, model_name=model_name
        )
        return [level["effort"] for level in levels]

    def test_excel_models_never_expose_max(self):
        raw_efforts = ["low", "medium", "high", "xhigh", "max"]
        for model_name in (
            "gpt-6-sol-excel",
            "gpt-6-luna-excel",
            "gpt-5.6-luna-excel",
            "gpt-5.6-terra-excel",
            "gpt-5.6-sol-excel",
        ):
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    self._effort_names(model_name, raw_efforts),
                    ["low", "medium", "high", "xhigh"],
                )

    def test_non_excel_gpt_56_still_exposes_max(self):
        self.assertEqual(
            self._effort_names(
                "gpt-5.6-sol", ["low", "medium", "high", "xhigh"]
            ),
            ["low", "medium", "high", "xhigh", "max"],
        )


class ExcelModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.service = object.__new__(ProxyClientConfigService)
        self.service._config = SimpleNamespace(
            codex_model_context_window=272_000,
            codex_model_auto_compact_token_limit=180_000,
        )
        self.service._model_capabilities = lambda: excel_upstream.merge_local_model_capabilities({})
        self.service._model_routing_settings = lambda: {}

    def test_gpt6_excel_models_are_listed_first(self):
        payload = self.service._build_codex_model_catalog_payload()
        self.assertEqual(
            [model["slug"] for model in payload["models"][:3]],
            ["gpt-6-astra-excel", "gpt-6-sol-excel", "gpt-6-luna-excel"],
        )

    def test_gpt6_sol_and_luna_catalog_capabilities(self):
        payload = self.service._build_codex_model_catalog_payload()
        models = {model["slug"]: model for model in payload["models"]}
        for model_id, display_name, context_window in (
            ("gpt-6-sol-excel", "6-Sol Excel", 272_000),
            ("gpt-6-luna-excel", "6-Luna Excel", 200_000),
        ):
            with self.subTest(model_id=model_id):
                model = models[model_id]
                self.assertEqual(model["display_name"], display_name)
                self.assertEqual(model["context_window"], context_window)
                self.assertEqual(model["max_context_window"], context_window)
                self.assertEqual(model["auto_compact_token_limit"], 180_000)
                self.assertEqual(model["input_modalities"], ["text", "image"])
                self.assertFalse(model["supports_parallel_tool_calls"])
                self.assertEqual(model["default_reasoning_level"], "medium")
                self.assertEqual(
                    [level["effort"] for level in model["supported_reasoning_levels"]],
                    ["low", "medium", "high", "xhigh"],
                )
                self.assertIn("OpenAI Excel", model["description"])
                self.assertIn("ChatGPT subscription usage", model["description"])

    def test_all_excel_models_use_subscription_description(self):
        for model_id in excel_upstream.MODEL_IDS:
            with self.subTest(model_id=model_id):
                self.assertEqual(
                    _model_token_pricing_description(model_id),
                    "ChatGPT subscription usage; not API-token billing",
                )

    def test_gpt6_excel_models_survive_missing_capabilities(self):
        self.service._model_capabilities = lambda: {}
        models = self.service._build_codex_model_catalog_payload()["models"]
        model_ids = {model["slug"] for model in models}
        self.assertIn("gpt-6-sol-excel", model_ids)
        self.assertIn("gpt-6-luna-excel", model_ids)


if __name__ == "__main__":
    unittest.main()
