"""Deterministic 90-day day-by-day balance forecasting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Mapping, Sequence

from reconciliation import EventClassification, ReconciliationResult, reconcile_events


FORECAST_DAYS = 90


@dataclass(frozen=True)
class BalanceSnapshot:
    day: date
    balance: Decimal
    cash_flow: Decimal


@dataclass(frozen=True)
class Forecast:
    request_date: date
    end_date: date
    initial_balance: Decimal
    snapshots: tuple[BalanceSnapshot, ...]

    def forecast_balance(self, day: date | str) -> Decimal:
        """Return the projected end-of-day balance for a date in the horizon."""
        target = _parse_date(day)
        if target < self.request_date or target > self.end_date:
            raise ValueError(
                f"Date {target.isoformat()} is outside forecast horizon "
                f"{self.request_date.isoformat()}..{self.end_date.isoformat()}"
            )
        return self.snapshots[(target - self.request_date).days].balance

    def min_balance_between(self, start: date | str, end: date | str) -> Decimal:
        """Return the minimum projected end-of-day balance in an inclusive range."""
        first, last = _parse_date(start), _parse_date(end)
        if first > last:
            raise ValueError("start must not be after end")
        if first < self.request_date or last > self.end_date:
            raise ValueError("requested range is outside forecast horizon")
        start_index = (first - self.request_date).days
        end_index = (last - self.request_date).days + 1
        return min(snapshot.balance for snapshot in self.snapshots[start_index:end_index])


def _parse_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _event_date(event: EventClassification) -> date:
    if event.effective_date is None:
        raise ValueError(f"Included event {event.event_id} has no effective date")
    return event.effective_date


def _signature(event: EventClassification) -> tuple[str, str, str, str, str]:
    row = event.row
    return (
        str(row.get("user_id") or ""),
        str(row.get("event_type") or ""),
        str(row.get("category") or ""),
        str(row.get("direction") or ""),
        str(row.get("currency") or ""),
    )


def _cadence(dates: Sequence[date]) -> int | None:
    if len(dates) < 2:
        return None
    intervals = [(right - left).days for left, right in zip(dates, dates[1:])]
    if not intervals:
        return None
    average = sum(intervals) / len(intervals)
    if 6 <= average <= 8 and all(5 <= interval <= 9 for interval in intervals):
        return 7
    if 27 <= average <= 31 and all(25 <= interval <= 35 for interval in intervals):
        return -1
    return None


def _next_month(day: date) -> date:
    year = day.year + (1 if day.month == 12 else 0)
    month = 1 if day.month == 12 else day.month + 1
    # Dataset recurrences use valid calendar dates; clamp month-end schedules.
    if month == 2:
        max_day = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    elif month in {4, 6, 9, 11}:
        max_day = 30
    else:
        max_day = 31
    return date(year, month, min(day.day, max_day))


def _recurring_flows(
    classifications: Sequence[EventClassification],
    request_date: date,
    end_date: date,
    explicit_dates: set[tuple[tuple[str, str, str, str, str], date]],
) -> dict[date, list[Decimal]]:
    historical: dict[tuple[str, str, str, str, str], list[EventClassification]] = defaultdict(list)
    for event in classifications:
        if not event.included or event.effective_date is None:
            continue
        if event.effective_date < request_date:
            historical[_signature(event)].append(event)

    projected: dict[date, list[Decimal]] = defaultdict(list)
    for signature, events in historical.items():
        events.sort(key=_event_date)
        cadence = _cadence([_event_date(event) for event in events])
        if cadence is None:
            continue
        latest = events[-1]
        next_day = _next_month(_event_date(latest)) if cadence == -1 else _event_date(latest) + timedelta(days=cadence)
        while next_day <= end_date:
            if next_day >= request_date and (signature, next_day) not in explicit_dates:
                projected[next_day].append(latest.cash_flow)
            next_day = _next_month(next_day) if cadence == -1 else next_day + timedelta(days=cadence)
    return projected


def build_forecast(
    request_date: date | str,
    initial_balance: Decimal | int | str,
    events: Sequence[Mapping[str, Any]] | ReconciliationResult,
    days: int = FORECAST_DAYS,
) -> Forecast:
    """Build an inclusive day-by-day forecast starting on ``request_date``.

    Events are reconciled first when raw rows are provided. Pending debits are
    therefore reserved, pending credits are excluded, and future salary rows
    count only on their settlement date as determined by reconciliation.
    """
    start = _parse_date(request_date)
    if days < 1:
        raise ValueError("days must be positive")
    end = start + timedelta(days=days - 1)
    result = events if isinstance(events, ReconciliationResult) else reconcile_events(events)
    included = tuple(result.included_events)

    explicit: dict[date, list[Decimal]] = defaultdict(list)
    explicit_dates: set[tuple[tuple[str, str, str, str, str], date]] = set()
    for event in included:
        event_day = _event_date(event)
        if start <= event_day <= end:
            explicit[event_day].append(event.cash_flow)
            explicit_dates.add((_signature(event), event_day))

    recurring = _recurring_flows(included, start, end, explicit_dates)
    initial = _decimal(initial_balance)
    balance = initial
    snapshots: list[BalanceSnapshot] = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        cash_flow = sum(explicit.get(day, ()), Decimal("0"))
        cash_flow += sum(recurring.get(day, ()), Decimal("0"))
        balance += cash_flow
        snapshots.append(BalanceSnapshot(day, balance, cash_flow))
    return Forecast(start, end, initial, tuple(snapshots))


def forecast_balance(
    request_date: date | str,
    initial_balance: Decimal | int | str,
    events: Sequence[Mapping[str, Any]] | ReconciliationResult,
    target_date: date | str,
    days: int = FORECAST_DAYS,
) -> Decimal:
    """Convenience wrapper returning one projected balance."""
    return build_forecast(request_date, initial_balance, events, days).forecast_balance(target_date)


def min_balance_between(
    request_date: date | str,
    initial_balance: Decimal | int | str,
    events: Sequence[Mapping[str, Any]] | ReconciliationResult,
    start: date | str,
    end: date | str,
    days: int = FORECAST_DAYS,
) -> Decimal:
    """Convenience wrapper returning a projected minimum balance."""
    return build_forecast(request_date, initial_balance, events, days).min_balance_between(start, end)


__all__ = [
    "BalanceSnapshot",
    "FORECAST_DAYS",
    "Forecast",
    "build_forecast",
    "forecast_balance",
    "min_balance_between",
]
