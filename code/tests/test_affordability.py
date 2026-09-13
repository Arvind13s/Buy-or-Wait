import csv
import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from affordability import amount_safe_to_pay, earliest_date_for_full_payment
from forecast import BalanceSnapshot, Forecast


class AffordabilityTests(unittest.TestCase):
    def setUp(self):
        start = date(2026, 1, 1)
        balances = [Decimal("1000") for _ in range(10)]
        balances[0:3] = [Decimal("1000"), Decimal("900"), Decimal("700")]
        balances[3:] = [Decimal("800") for _ in balances[3:]]
        self.forecast = Forecast(
            request_date=start,
            end_date=start + timedelta(days=9),
            initial_balance=Decimal("1000"),
            snapshots=tuple(
                BalanceSnapshot(
                    start + timedelta(days=index),
                    balance,
                    Decimal("0"),
                )
                for index, balance in enumerate(balances)
            ),
        )
        self.request = {
            "request_date": "2026-01-01",
            "requested_amount": "450",
            "financial_profile": {"minimum_balance_to_keep": "300"},
        }

    def test_amount_safe_is_headroom_capped_by_request(self):
        self.assertEqual(amount_safe_to_pay(self.request, self.forecast), Decimal("400"))
        capped = {**self.request, "requested_amount": "250"}
        self.assertEqual(amount_safe_to_pay(capped, self.forecast), Decimal("250"))

    def test_earliest_full_payment_uses_remaining_forward_window(self):
        self.assertEqual(
            earliest_date_for_full_payment(self.request, self.forecast),
            date(2026, 1, 4),
        )

    def test_sample_requests_are_reference_only_contract_regression(self):
        sample_path = Path(__file__).parents[2] / "dataset" / "sample_requests.csv"
        with sample_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 25)
        for row in rows:
            requested = Decimal(row["requested_amount"])
            safe = Decimal(row["amount_safe_to_pay"])
            self.assertGreaterEqual(safe, Decimal("0"))
            self.assertLessEqual(safe, requested)
            self.assertIn(
                row["affordability_status"],
                {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"},
            )
            self.assertIn(
                row["recommended_payment_method"],
                {"full_payment", "partial_payment", "installments", "wait", "not_recommended"},
            )
        # The sample rows are not passed to or used to label evaluation requests.
        self.assertNotIn("evaluation_requests", rows[0])


if __name__ == "__main__":
    unittest.main()
