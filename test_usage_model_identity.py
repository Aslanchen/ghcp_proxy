import unittest

from dashboard import _prepare_usage_event
from constants import MODEL_PRICING
from util import _normalize_model_name, _usage_event_cost_breakdown, _usage_event_model_name


class UsageModelIdentityTests(unittest.TestCase):
    def test_excel_alias_wins_over_base_response_model(self):
        for base_model in (
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-5.6-sol",
        ):
            excel_model = f"{base_model}-excel"
            event = {
                "requested_model": excel_model,
                "resolved_model": excel_model,
                "response_model": base_model,
                "finished_at": "2026-09-17T12:00:00Z",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            }

            with self.subTest(excel_model=excel_model):
                self.assertEqual(_usage_event_model_name(event), excel_model)
                self.assertEqual(_prepare_usage_event(event)["model_name"], excel_model)

    def test_non_credit_model_keeps_response_model_precedence(self):
        event = {
            "requested_model": "gpt-5.4",
            "resolved_model": "gpt-5.4",
            "response_model": "gpt-5.5",
        }

        self.assertEqual(_usage_event_model_name(event), "gpt-5.5")

    def test_gpt6_excel_pricing_matches_existing_base_model_rates(self):
        for base_model in ("gpt-6-sol", "gpt-6-luna"):
            with self.subTest(base_model=base_model):
                base_pricing = MODEL_PRICING[base_model]
                excel_pricing = MODEL_PRICING[f"{base_model}-excel"]
                self.assertEqual(excel_pricing["provider"], "OpenAI Excel")
                self.assertEqual(excel_pricing["credit_unit_usd"], 0.04)
                self.assertEqual(
                    {key: value for key, value in excel_pricing.items() if key not in {"provider", "credit_unit_usd"}},
                    {key: value for key, value in base_pricing.items() if key != "provider"},
                )
                for input_tokens in (1_000, 300_000):
                    usage = {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": 500,
                        "cache_creation_input_tokens": 100,
                        "output_tokens": 100,
                    }
                    self.assertEqual(
                        _usage_event_cost_breakdown(f"{base_model}-excel", usage),
                        _usage_event_cost_breakdown(base_model, usage),
                    )

    def test_gpt6_excel_pricing_aliases_are_normalized(self):
        for variant in ("sol", "luna"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    _normalize_model_name(f"gpt-6 {variant} excel"),
                    f"gpt-6-{variant}-excel",
                )


if __name__ == "__main__":
    unittest.main()
