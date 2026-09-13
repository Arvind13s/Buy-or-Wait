import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from forecast import build_forecast


def event(event_id, event_type, direction, amount, event_date, **extra):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "category": extra.pop("category", event_type),
        "direction": direction,
        "amount": str(amount) if amount is not None else None,
        "event_date": event_date,
        "settlement_date": event_date,
        "status": extra.pop("status", "settled"),
        "linked_event_id": extra.pop("linked_event_id", None),
        **extra,
    }


class ForecastTests(unittest.TestCase):
    def test_weekly_and_monthly_recurrences_are_projected_day_by_day(self):
        events = [
            event("weekly-1", "expense", "debit", 10, "2026-01-01", category="weekly"),
            event("weekly-2", "expense", "debit", 10, "2026-01-08", category="weekly"),
            event("rent-1", "expense", "debit", 100, "2025-12-01", category="rent"),
            event("rent-2", "expense", "debit", 100, "2026-01-01", category="rent"),
            event("salary", "income", "credit", 500, "2026-01-15", category="salary"),
        ]
        forecast = build_forecast("2026-01-10", 1000, events, days=40)
        self.assertEqual(forecast.forecast_balance("2026-01-10"), Decimal("1000"))
        self.assertEqual(forecast.forecast_balance("2026-01-15"), Decimal("1490"))
        self.assertEqual(forecast.forecast_balance("2026-02-01"), Decimal("1370"))
        self.assertEqual(
            forecast.min_balance_between("2026-01-15", "2026-02-18"),
            Decimal("1350"),
        )

    def test_pending_debit_is_reserved_and_pending_credit_is_not_counted(self):
        events = [
            event("debit", "expense", "debit", 40, "2026-01-12", status="pending"),
            event("credit", "refund", "credit", 90, "2026-01-12", status="pending"),
        ]
        forecast = build_forecast("2026-01-10", 100, events, days=5)
        self.assertEqual(forecast.forecast_balance("2026-01-12"), Decimal("60"))


if __name__ == "__main__":
    unittest.main()