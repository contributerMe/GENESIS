import unittest
from unittest.mock import patch
from pydantic import ValidationError

from ai.state import CompanyInputs, redact_state_for_persistence
from ai.llm import LiteLLMRouter
from ai.settings import Settings


class CompanyInputsTests(unittest.TestCase):
    def test_company_inputs_normalizes_and_validates_enums(self):
        inputs = CompanyInputs(company_name=" Tesla ", industry=" EV ", provider="GROQ")

        self.assertEqual(inputs.company_name, "Tesla")
        self.assertEqual(inputs.industry, "EV")
        self.assertEqual(inputs.provider, "groq")

        with self.assertRaises(ValidationError):
            CompanyInputs(company_name="Tesla", industry="EV", provider="unknown")

    def test_gemini_uses_the_configured_current_model_identifier(self):
        self.assertEqual(LiteLLMRouter.get_model_identifier("gemini"), "gemini/gemini-3.6-flash")

    def test_primary_model_drops_unsupported_provider_parameters_when_enabled(self):
        settings = Settings(_env_file=None, litellm_drop_params=True)
        with (
            patch("ai.llm.get_settings", return_value=settings),
            patch.object(LiteLLMRouter, "_execute_with_retries", return_value="test") as completion,
        ):
            LiteLLMRouter.completion(provider="gemini", model_name="gemini-3.5-flash-lite", prompt="test")
        self.assertTrue(completion.call_args.args[0]["drop_params"])


class StateRedactionTests(unittest.TestCase):
    def test_redact_state_for_persistence_removes_secrets_and_local_paths(self):
        original = {
            "inputs": {"api_key": "secret", "uploaded_files": ["/private/report.txt"]},
            "uploaded_documents": [{"url": "file:///private/report.txt", "file_path": "/private/report.txt"}],
            "citations": [{"url_or_path": "file:///private/report.txt"}],
        }

        redacted = redact_state_for_persistence(original)

        self.assertEqual(redacted["inputs"], {})
        self.assertEqual(redacted["uploaded_documents"], [{"url": ""}])
        self.assertEqual(redacted["citations"], [{"url_or_path": ""}])
        self.assertEqual(original["inputs"]["api_key"], "secret")
