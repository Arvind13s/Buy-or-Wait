import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parents[1]))

from extraction import (
    LLMResponse,
    UsageTracker,
    explain_decision,
    extract_blank_event_amounts,
    interpret_relevant_messages,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, *, system_prompt, user_prompt, image_paths=()):
        self.calls.append((system_prompt, user_prompt, tuple(image_paths)))
        content = next(self.responses)
        return LLMResponse(
            content=content,
            input_tokens=11,
            output_tokens=7,
            provider="fake",
            model="fake-model",
        )


class ExtractionTests(unittest.TestCase):
    def test_image_calls_once_per_blank_event_and_tracks_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image_01.png"
            image_path.write_bytes(b"fake-image")
            client = FakeClient(['{"amount": 123.45, "currency": "USD", "confidence": "high", "notes": "total"}'])
            usage = UsageTracker()
            results = extract_blank_event_amounts(
                [
                    {"event_id": "event_1", "user_id": "user_1", "amount": None, "description": "bill"},
                    {"event_id": "event_2", "user_id": "user_1", "amount": "50", "description": "paid"},
                ],
                [{"related_event_id": "event_1", "image_path": str(image_path)}],
                client,
                usage,
            )
            self.assertEqual(results["event_1"].amount, Decimal("123.45"))
            self.assertEqual(len(client.calls), 1)
            self.assertIn("event_1", client.calls[0][1])
            self.assertEqual(usage.input_tokens, 11)
            self.assertEqual(usage.output_tokens, 7)

    def test_image_parser_recovers_json_object_substring(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image_01.png"
            image_path.write_bytes(b"fake-image")
            client = FakeClient([
                'Here is the result: {"amount": 123.45, "currency": "USD", "confidence": "high", "notes": "total"}'
            ])
            result = extract_blank_event_amounts(
                [{"event_id": "event_1", "user_id": "user_1", "amount": None, "description": "bill"}],
                [{"related_event_id": "event_1", "image_path": str(image_path)}],
                client,
            )
            self.assertEqual(result["event_1"].amount, Decimal("123.45"))

    def test_image_parser_recovers_labeled_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image_01.png"
            image_path.write_bytes(b"fake-image")
            client = FakeClient(["Total is shown below. Amount: 1995.00 Currency: INR Confidence: High"])
            result = extract_blank_event_amounts(
                [{"event_id": "event_1", "user_id": "user_1", "amount": None, "description": "bill"}],
                [{"related_event_id": "event_1", "image_path": str(image_path)}],
                client,
            )
            self.assertEqual(result["event_1"].amount, Decimal("1995.00"))
            self.assertEqual(result["event_1"].currency, "INR")

    def test_image_parser_returns_null_amount_when_unrecoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image_01.png"
            image_path.write_bytes(b"fake-image")
            client = FakeClient(["I cannot determine the amount from this image."])
            result = extract_blank_event_amounts(
                [{"event_id": "event_1", "user_id": "user_1", "amount": None, "description": "bill"}],
                [{"related_event_id": "event_1", "image_path": str(image_path)}],
                client,
            )
            self.assertIsNone(result["event_1"].amount)
            self.assertIn("parse failure", result["event_1"].notes)

    def test_message_calls_once_per_row_and_preserves_blank_event_link(self):
        client = FakeClient([
            '{"related_event_id": null, "action": "clarify", "new_amount": null, "new_date": null, "reasoning": "request context only"}',
            '{"related_event_id": "event_2", "action": "amend", "new_amount": 90, "new_date": "2026-09-20", "reasoning": "explicit amendment"}',
        ])
        usage = UsageTracker()
        results = interpret_relevant_messages(
            [
                {"message_id": "message_1", "related_event_id": None, "message_text": "Can I delay this?"},
                {"message_id": "message_2", "related_event_id": "event_2", "message_text": "The amount is now 90."},
            ],
            {"message_1": {"request_id": "request_1"}, "message_2": {"event_id": "event_2"}},
            client,
            usage,
        )
        self.assertEqual(len(client.calls), 2)
        self.assertIsNone(results["message_1"].related_event_id)
        self.assertEqual(results["message_2"].new_amount, 90)
        self.assertEqual(usage.input_tokens, 22)
        self.assertEqual(usage.output_tokens, 14)

    def test_blank_related_event_cannot_be_invented(self):
        client = FakeClient([
            '{"related_event_id": "event_9", "action": "confirm", "new_amount": null, "new_date": null, "reasoning": "invented"}'
        ])
        with self.assertRaises(ValueError):
            interpret_relevant_messages(
                [{"message_id": "message_1", "related_event_id": None, "message_text": "pay it"}],
                {"message_1": {}},
                client,
            )

    def test_usage_report_contains_call_and_token_totals(self):
        usage = UsageTracker()
        usage.records.append(
            type("Record", (), {
                "call_type": "image_amount_extraction",
                "provider": "fake",
                "model": "fake-model",
                "input_tokens": 3,
                "output_tokens": 2,
            })()
        )
        report = usage.to_markdown()
        self.assertIn("Total input tokens: 3", report)
        self.assertIn("Total output tokens: 2", report)
        self.assertIn("Average tokens per call: 5.00", report)

    def test_explanation_is_grounded_and_has_separate_usage_type(self):
        client = FakeClient([
            "The balance is USD 900 and the minimum balance is USD 300. "
            "Use payment_option_01 for the upcoming expense on 2026-09-20."
        ])
        usage = UsageTracker()
        facts = {
            "balance": "900",
            "minimum_balance": "300",
            "upcoming_expense": {"amount": "400", "date": "2026-09-20"},
            "payment_option_id": "payment_option_01",
        }
        explanation = explain_decision(facts, client, usage)
        self.assertIn("payment_option_01", explanation)
        self.assertEqual(usage.records[0].call_type, "decision_explanation")
        self.assertEqual(usage.input_tokens, 11)

    def test_explanation_rejects_invented_number(self):
        client = FakeClient(["The balance is 900. The minimum balance is 3000."])
        with self.assertRaises(ValueError):
            explain_decision({"balance": "900", "minimum_balance": "300"}, client)

    def test_nvidia_retries_429_and_503_with_exponential_backoff(self):
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from extraction import OpenAICompatibleClient

        response_body = b'{"model":"nvidia/test-model","choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":1,"completion_tokens":2}}'
        failures = [
            HTTPError("https://example.invalid", 429, "busy", {}, None),
            HTTPError("https://example.invalid", 503, "busy", {}, None),
        ]

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return response_body

        calls = iter([*failures, Response()])

        def next_response(*args, **kwargs):
            response = next(calls)
            if isinstance(response, HTTPError):
                raise response
            return response

        client = OpenAICompatibleClient("key", "nvidia/test-model", "https://example.invalid/v1")
        with patch("extraction.urllib.request.urlopen", side_effect=next_response) as urlopen, patch("extraction.time.sleep") as sleep, patch("builtins.print") as print_call:
            response = client.complete(system_prompt="system", user_prompt="user")

        self.assertEqual(response.model, "nvidia/test-model")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        self.assertEqual(print_call.call_count, 4)

    def test_nvidia_does_not_retry_other_http_errors(self):
        from extraction import OpenAICompatibleClient

        error = HTTPError("https://example.invalid", 400, "bad request", {}, None)
        client = OpenAICompatibleClient("key", "nvidia/test-model", "https://example.invalid/v1")
        with patch("extraction.urllib.request.urlopen", side_effect=error) as urlopen, patch("extraction.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                client.complete(system_prompt="system", user_prompt="user")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_nvidia_stops_after_five_retries(self):
        from extraction import OpenAICompatibleClient

        error = HTTPError("https://example.invalid", 503, "busy", {}, None)
        client = OpenAICompatibleClient("key", "nvidia/test-model", "https://example.invalid/v1")
        with patch("extraction.urllib.request.urlopen", side_effect=error) as urlopen, patch("extraction.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                client.complete(system_prompt="system", user_prompt="user")
        self.assertEqual(urlopen.call_count, 6)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8, 16, 30])


if __name__ == "__main__":
    unittest.main()
