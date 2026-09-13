import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from validate_output import OutputValidationError, REQUIRED_COLUMNS, validate_output_rows


class ValidateOutputTests(unittest.TestCase):
    def setUp(self):
        self.requests = [
            {
                "request_id": f"request_{index:03d}",
                "user_id": "user_1",
                "request_date": "2026-01-01",
                "requested_amount": "100",
            }
            for index in range(250)
        ]
        self.profiles = [{"user_id": "user_1", "max_installment_months": "2"}]
        self.options = []
        self.events = []
        self.rows = [
            {
                "request_id": request["request_id"],
                "amount_safe_to_pay": "100",
                "affordability_status": "affordable_now",
                "recommended_payment_method": "full_payment",
                "payment_plan": "2026-01-01:100",
                "earliest_date_for_full_payment": "2026-01-01",
                "spending_changes_needed": "none",
                "decision_explanation": "The balance supports the request while preserving the minimum balance.",
            }
            for request in self.requests
        ]

    def test_valid_250_row_output_passes(self):
        validate_output_rows(self.rows, self.requests, self.options, self.profiles, self.events)

    def test_duplicate_request_id_fails(self):
        rows = list(self.rows)
        rows[-1] = {**rows[-1], "request_id": rows[0]["request_id"]}
        with self.assertRaises(OutputValidationError):
            validate_output_rows(rows, self.requests, self.options, self.profiles, self.events)

    def test_partial_plan_sum_and_affordable_now_date_are_checked(self):
        rows = list(self.rows)
        rows[0] = {
            **rows[0],
            "affordability_status": "affordable_now",
            "recommended_payment_method": "partial_payment",
            "payment_plan": "2026-01-01:40|2026-01-02:50",
            "earliest_date_for_full_payment": "2026-01-02",
        }
        with self.assertRaises(OutputValidationError):
            validate_output_rows(rows, self.requests, self.options, self.profiles, self.events)

    def test_installment_requires_matching_option_and_limit(self):
        rows = list(self.rows)
        rows[0] = {
            **rows[0],
            "affordability_status": "affordable_with_plan",
            "recommended_payment_method": "installments",
            "payment_plan": "2026-01-01:50|2026-01-08:50",
            "earliest_date_for_full_payment": "2026-01-01",
        }
        with self.assertRaises(OutputValidationError):
            validate_output_rows(rows, self.requests, self.options, self.profiles, self.events)

    def test_spending_changes_require_flexible_events_and_unique_operation(self):
        rows = list(self.rows)
        rows[0] = {**rows[0], "spending_changes_needed": "stop:event_1|reduce_to:event_1:10"}
        events = [{"event_id": "event_1", "user_id": "user_1", "flexibility": "fixed"}]
        with self.assertRaises(OutputValidationError):
            validate_output_rows(rows, self.requests, self.options, self.profiles, events)


if __name__ == "__main__":
    unittest.main()
