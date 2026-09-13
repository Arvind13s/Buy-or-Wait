"""Evaluate permitted flexible recurring-expense changes for payment plans."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any, Mapping, Sequence

from forecast import Forecast
from plan_selector import Payment, PlanCandidate, rank_safe_candidates, select_plan


class SpendingChangeError(ValueError):
    """Raised when a flexible recurring expense is malformed."""


def _date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _money(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SpendingChangeError(f"{field} must be a valid amount") from error
    if result < 0:
        raise SpendingChangeError(f"{field} must not be negative")
    return result


def _profile_value(request: Mapping[str, Any], field: str) -> Any:
    if field in request:
        return request[field]
    profile = request.get("financial_profile")
    if isinstance(profile, Mapping):
        return profile.get(field)
    return None


def _categories(request: Mapping[str, Any], field: str) -> set[str]:
    value = _profile_value(request, field)
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split("|") if item.strip()}
    return {str(item) for item in value}


def _recurrence_dates(expense: Mapping[str, Any], forecast: Forecast) -> tuple[date, ...]:
    supplied = expense.get("recurrence_dates") or expense.get("occurrence_dates")
    if supplied:
        return tuple(
            _date(value)
            for value in supplied
            if forecast.request_date <= _date(value) <= forecast.end_date
        )

    frequency = expense.get("recurrence_frequency_days") or expense.get("frequency_days")
    if frequency in (None, ""):
        raise SpendingChangeError(
            f"Flexible expense {expense.get('event_id', '<missing-id>')} has no recurrence dates or frequency"
        )
    try:
        interval = int(frequency)
    except (TypeError, ValueError) as error:
        raise SpendingChangeError("recurrence frequency must be an integer") from error
    if interval <= 0:
        raise SpendingChangeError("recurrence frequency must be positive")
    anchor = expense.get("next_occurrence_date") or expense.get("event_date")
    if anchor is None:
        raise SpendingChangeError(
            f"Flexible expense {expense.get('event_id', '<missing-id>')} has no recurrence anchor"
        )
    current = _date(anchor)
    while current < forecast.request_date:
        current += timedelta(days=interval)
    dates: list[date] = []
    while current <= forecast.end_date:
        dates.append(current)
        current += timedelta(days=interval)
    return tuple(dates)


def _change_options(
    request: Mapping[str, Any],
    expense: Mapping[str, Any],
    forecast: Forecast,
) -> tuple[tuple[str, Decimal, tuple[date, ...]], ...]:
    event_id = str(expense.get("event_id") or "")
    if not event_id:
        raise SpendingChangeError("Flexible expense is missing event_id")
    if str(expense.get("direction") or "").lower() != "debit":
        return ()
    if str(expense.get("flexibility") or "").lower() != "flexible":
        return ()
    category = str(expense.get("category") or "")
    protected = _categories(request, "expense_categories_to_protect")
    if category in protected:
        return ()

    amount = _money(expense.get("amount"), "expense amount")
    dates = _recurrence_dates(expense, forecast)
    changes: list[tuple[str, Decimal, tuple[date, ...]]] = []
    if category in _categories(request, "expense_categories_user_is_willing_to_stop"):
        changes.append((f"stop:{event_id}", amount, dates))

    minimum = expense.get("minimum_allowed_amount")
    if category in _categories(request, "expense_categories_user_is_willing_to_reduce") and minimum not in (None, ""):
        reduced = _money(minimum, "minimum_allowed_amount")
        if reduced < amount:
            changes.append((f"reduce_to:{event_id}:{reduced}", amount - reduced, dates))
    return tuple(changes)


def _safe_with_savings(
    candidate: PlanCandidate,
    forecast: Forecast,
    minimum_balance: Decimal,
    savings: Sequence[tuple[Decimal, tuple[date, ...]]],
) -> bool:
    for snapshot in forecast.snapshots:
        saved = sum(
            amount
            for amount, dates in savings
            if any(event_date <= snapshot.day for event_date in dates)
        )
        paid = sum(
            payment.amount
            for payment in candidate.payments
            if payment.payment_date <= snapshot.day
        )
        if snapshot.balance + saved - paid < minimum_balance:
            return False
    return True


def _minimum_balance(request: Mapping[str, Any]) -> Decimal:
    value = _profile_value(request, "minimum_balance_to_keep")
    if value is None:
        raise SpendingChangeError("request is missing minimum_balance_to_keep")
    return _money(value, "minimum_balance_to_keep")


def select_with_spending_changes(
    request: Mapping[str, Any],
    forecast: Forecast,
    candidates: Sequence[PlanCandidate],
    flexible_recurring_expenses: Sequence[Mapping[str, Any]],
) -> PlanCandidate:
    """Return the best safe plan, using up to three permitted changes if needed."""
    safe_as_is = rank_safe_candidates(candidates)
    if safe_as_is:
        return safe_as_is[0]

    options: list[tuple[str, Decimal, tuple[date, ...]]] = []
    for expense in flexible_recurring_expenses:
        options.extend(_change_options(request, expense, forecast))

    minimum = _minimum_balance(request)
    changed: list[PlanCandidate] = []
    for count in range(1, min(3, len(options)) + 1):
        for selected in combinations(options, count):
            event_ids = [change[0].split(":", 2)[1] for change in selected]
            if len(event_ids) != len(set(event_ids)):
                continue
            changes = tuple(change[0] for change in selected)
            savings = tuple((change[1], change[2]) for change in selected)
            for candidate in candidates:
                if candidate.payment_method == "not_recommended" or not candidate.payments:
                    continue
                if _safe_with_savings(candidate, forecast, minimum, savings):
                    changed.append(
                        replace(
                            candidate,
                            affordability_status="affordable_with_plan",
                            spending_changes=changes,
                            safe=True,
                        )
                    )
    ranked = rank_safe_candidates(changed)
    if ranked:
        return ranked[0]
    return select_plan(candidates)


__all__ = ["SpendingChangeError", "select_with_spending_changes"]
