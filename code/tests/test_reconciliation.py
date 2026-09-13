import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from reconciliation import reconcile_events



def event(
    event_id,
    *,
    status="settled",
    direction="debit",
    amount="100",
    linked_event_id=None,
    event_type="expense",
    settlement_date="2026-09-10",
    **extra,
):
    return {
        "event_id": event_id,
        "status": status,
        "direction": direction,
        "amount": amount,
        "event_type": event_type,
        "event_date": "2026-09-09",
        "settlement_date": settlement_date,
        "linked_event_id": linked_event_id,
        **extra,
    }


class ReconciliationTests(unittest.TestCase):
    def test_settled_debit_counts_on_settlement_date(self):
        result = reconcile_events([event("settled")])
        included = result.included_events[0]
        self.assertEqual(included.cash_flow, Decimal("-100"))
        self.assertEqual(included.effective_date.isoformat(), "2026-09-10")

    def test_pending_debit_reserved_but_pending_credit_excluded(self):
        result = reconcile_events([
            event("pending-debit", status="pending"),
            event("pending-credit", status="pending", direction="credit"),
        ])
        self.assertEqual([item.event_id for item in result.included_events], ["pending-debit"])
        self.assertEqual(result.classifications[1].reason, "pending credit excluded")

    def test_scheduled_event_uses_scheduled_settlement_date(self):
        result = reconcile_events([
            event("scheduled", status="scheduled", settlement_date="2026-09-15")
        ])
        self.assertEqual(result.included_events[0].effective_date.isoformat(), "2026-09-15")

    def test_failed_cancelled_and_non_cash_rows_are_excluded(self):
        result = reconcile_events([
            event("failed", status="failed"),
            event("cancelled", status="cancelled"),
            event("unrealized", status="unrealized", event_type="investment_valuation"),
            event("valuation", event_type="investment_valuation"),
            event("non-cash", direction="non-cash"),
        ])
        self.assertEqual(result.included_events, ())
        self.assertEqual(len(result.excluded_events), 5)

    def test_excluded_rows_do_not_require_an_amount(self):
        result = reconcile_events([
            event("cancelled-without-amount", status="cancelled", amount=None),
            event(
                "valuation-without-amount",
                event_type="investment_valuation",
                amount=None,
            ),
        ])
        self.assertEqual(result.included_events, ())

    def test_explicit_cancellation_wins_and_linked_cancelled_event_is_dropped(self):
        result = reconcile_events([
            event("original", status="settled"),
            event("cancel", status="cancelled", linked_event_id="original"),
        ])
        self.assertEqual(result.included_events, ())
        self.assertFalse(result.classifications[0].included)
        self.assertFalse(result.classifications[1].included)
        self.assertEqual(result.classifications[1].reason, "cancelled event excluded")

    def test_explicit_amendment_wins(self):
        result = reconcile_events([
            event("original", status="settled", amount="100"),
            event(
                "amendment",
                status="settled",
                amount="80",
                linked_event_id="original",
                amends_event_id="original",
                recorded_at="2026-09-11T10:00:00+00:00",
            ),
        ])
        self.assertEqual([item.event_id for item in result.included_events], ["amendment"])
        self.assertEqual(result.included_events[0].cash_flow, Decimal("-80"))

    def test_newer_record_from_same_source_wins(self):
        result = reconcile_events([
            event(
                "old",
                status="pending",
                linked_event_id=None,
                source="bank",
                recorded_at="2026-09-09T10:00:00+00:00",
            ),
            event(
                "new",
                status="pending",
                amount="125",
                linked_event_id="old",
                source="bank",
                recorded_at="2026-09-10T10:00:00+00:00",
            ),
        ])
        self.assertEqual([item.event_id for item in result.included_events], ["new"])
        self.assertEqual(result.included_events[0].cash_flow, Decimal("-125"))

    def test_settled_beats_estimate_or_forecast(self):
        result = reconcile_events([
            event("forecast", status="forecast", amount="200"),
            event("settled", status="settled", amount="100", linked_event_id="forecast"),
        ])
        self.assertEqual([item.event_id for item in result.included_events], ["settled"])

    def test_ambiguous_conflict_uses_financially_safer_lower_cash_flow(self):
        result = reconcile_events([
            event("credit", direction="credit", amount="500"),
            event("debit", direction="debit", amount="100", linked_event_id="credit"),
        ])
        self.assertEqual([item.event_id for item in result.included_events], ["debit"])


if __name__ == "__main__":
    unittest.main()
