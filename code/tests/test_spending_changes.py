import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from forecast import BalanceSnapshot, Forecast
from plan_selector import Payment, PlanCandidate
from spending_changes import select_with_spending_changes


class SpendingChangesTests(unittest.TestCase):
    def setUp(self):
        start = date(2026, 1, 1)
        self.forecast = Forecast(
            start,
            start + timedelta(days=4),
            Decimal("500"),
            tuple(
                BalanceSnapshot(start + timedelta(days=index), Decimal("500"), Decimal("0"))
                for index in range(5)
            ),
        )
        self.candidate = PlanCandidate(
            "full_payment",
            "affordable_now",
            (Payment(start, Decimal("300")),),
            Decimal("300"),
            "",
            safe=False,
        )
        self.request = {
            "minimum_balance_to_keep": "300",
            "financial_profile": {
                "expense_categories_to_protect": ("rent",),
                "expense_categories_user_is_willing_to_stop": ("dining",),
                "expense_categories_user_is_willing_to_reduce": ("groceries",),
            },
        }

    def expense(self, event_id, category, amount, **extra):
        return {
            "event_id": event_id,
            "category": category,
            "amount": str(amount),
            "direction": "debit",
            "flexibility": "flexible",
            "recurrence_dates": ["2026-01-01"],
            **extra,
        }

    def test_stop_flexible_expense_unlocks_plan(self):
        selected = select_with_spending_changes(
            self.request,
            self.forecast,
            [self.candidate],
            [self.expense("event_dining", "dining", 100)],
        )
        self.assertEqual(selected.spending_changes, ("stop:event_dining",))
        self.assertEqual(selected.affordability_status, "affordable_with_plan")

    def test_reduce_flexible_expense_uses_minimum_allowed_amount(self):
        selected = select_with_spending_changes(
            self.request,
            self.forecast,
            [self.candidate],
            [self.expense("event_groceries", "groceries", 100, minimum_allowed_amount="0")],
        )
        self.assertEqual(selected.spending_changes, ("reduce_to:event_groceries:0",))

    def test_protected_category_cannot_be_changed(self):
        selected = select_with_spending_changes(
            self.request,
            self.forecast,
            [self.candidate],
            [self.expense("event_rent", "rent", 100)],
        )
        self.assertEqual(selected.payment_method, "not_recommended")

    def test_safe_as_is_plan_skips_spending_changes(self):
        safe = PlanCandidate(
            self.candidate.payment_method,
            self.candidate.affordability_status,
            self.candidate.payments,
            self.candidate.total_paid,
            self.candidate.payment_option_id,
            safe=True,
        )
        selected = select_with_spending_changes(
            self.request,
            self.forecast,
            [safe],
            [self.expense("event_dining", "dining", 100)],
        )
        self.assertEqual(selected.spending_changes, ())

    def test_stop_and_reduce_same_event_are_never_combined(self):
        request = {
            **self.request,
            "financial_profile": {
                **self.request["financial_profile"],
                "expense_categories_user_is_willing_to_stop": ("dining",),
                "expense_categories_user_is_willing_to_reduce": ("dining",),
            },
        }
        selected = select_with_spending_changes(
            request,
            self.forecast,
            [self.candidate],
            [self.expense("event_dining", "dining", 100, minimum_allowed_amount="0")],
        )
        self.assertEqual(len(selected.spending_changes), 1)

    def test_no_more_than_three_changes_are_used(self):
        expenses = [self.expense(f"event_{index}", "dining", 100) for index in range(4)]
        candidate = PlanCandidate(
            "full_payment",
            "affordable_now",
            (Payment(date(2026, 1, 1), Decimal("500")),),
            Decimal("500"),
            "",
            safe=False,
        )
        selected = select_with_spending_changes(
            self.request,
            self.forecast,
            [candidate],
            expenses,
        )
        self.assertLessEqual(len(selected.spending_changes), 3)
        self.assertEqual(len(selected.spending_changes), 3)


if __name__ == "__main__":
    unittest.main()
