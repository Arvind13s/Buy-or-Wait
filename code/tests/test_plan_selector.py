import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from forecast import BalanceSnapshot, Forecast
from plan_selector import (
    Payment,
    PlanCandidate,
    build_eligible_candidates,
    rank_safe_candidates,
    select_plan,
)


class PlanSelectorTests(unittest.TestCase):
    def setUp(self):
        start = date(2026, 1, 1)
        self.forecast = Forecast(
            start,
            start + timedelta(days=20),
            Decimal("1000"),
            tuple(
                BalanceSnapshot(start + timedelta(days=index), Decimal("1000"), Decimal("0"))
                for index in range(21)
            ),
        )

    def candidate(self, *, method="full_payment", deadline=True, changes=(), total="100", start="2026-01-01", count=1, option=""):
        first = date.fromisoformat(start)
        payments = tuple(Payment(first + timedelta(days=index), Decimal(total) / count) for index in range(count))
        return PlanCandidate(
            method,
            "affordable_with_plan",
            payments,
            Decimal(total),
            option,
            tuple(changes),
            deadline,
            True,
        )

    def test_rank_prefers_completion_by_deadline(self):
        late = self.candidate(deadline=False)
        on_time = self.candidate(deadline=True, option="b")
        self.assertIs(rank_safe_candidates([late, on_time])[0], on_time)

    def test_rank_prefers_no_spending_changes(self):
        changed = self.candidate(changes=("stop:event_1",))
        unchanged = self.candidate(option="b")
        self.assertIs(rank_safe_candidates([changed, unchanged])[0], unchanged)

    def test_rank_prefers_lower_total_paid(self):
        expensive = self.candidate(total="110", option="a")
        cheaper = self.candidate(total="100", option="b")
        self.assertIs(rank_safe_candidates([expensive, cheaper])[0], cheaper)

    def test_rank_prefers_earlier_start(self):
        late = self.candidate(start="2026-01-03", option="a")
        early = self.candidate(start="2026-01-01", option="b")
        self.assertIs(rank_safe_candidates([late, early])[0], early)

    def test_rank_prefers_fewer_payments(self):
        many = self.candidate(count=2, option="a")
        few = self.candidate(count=1, option="b")
        self.assertIs(rank_safe_candidates([many, few])[0], few)

    def test_rank_uses_lowest_payment_option_id_last(self):
        high = self.candidate(option="payment_option_02")
        low = self.candidate(option="payment_option_01")
        self.assertIs(rank_safe_candidates([high, low])[0], low)

    def test_partial_payment_has_exactly_two_payments(self):
        request = {
            "request_date": "2026-01-01",
            "requested_amount": "500",
            "desired_completion_date": "2026-01-10",
            "allows_partial_payment": "true",
            "financial_profile": {
                "minimum_balance_to_keep": "100",
                "payment_methods_user_will_consider": ("partial_payment",),
            },
        }
        candidates = build_eligible_candidates(
            request,
            self.forecast,
            [],
            safe_amount=Decimal("200"),
            full_payment_date=date(2026, 1, 5),
        )
        partial = next(candidate for candidate in candidates if candidate.payment_method == "partial_payment")
        self.assertEqual(len(partial.payments), 2)
        self.assertEqual(sum(payment.amount for payment in partial.payments), Decimal("500"))

    def test_blank_max_installment_months_excludes_existing_option(self):
        request = {
            "request_date": "2026-01-01",
            "requested_amount": "500",
            "desired_completion_date": "2026-01-20",
            "allows_partial_payment": "false",
            "financial_profile": {
                "minimum_balance_to_keep": "100",
                "payment_methods_user_will_consider": ("installments",),
                "max_installment_months": None,
            },
        }
        options = [{
            "payment_option_id": "payment_option_01",
            "payment_method": "installments",
            "payment_amount": "250",
            "number_of_payments": "2",
            "first_payment_date": "2026-01-02",
            "payment_frequency_days": "7",
            "total_payable_amount": "500",
        }]
        candidates = build_eligible_candidates(request, self.forecast, options)
        self.assertEqual([candidate.payment_method for candidate in candidates], ["not_recommended"])

    def test_select_plan_falls_back_when_no_safe_candidate_exists(self):
        fallback = PlanCandidate("not_recommended", "not_affordable", (), Decimal("0"), "", safe=False)
        self.assertIs(select_plan([fallback]), fallback)


if __name__ == "__main__":
    unittest.main()
