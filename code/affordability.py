"""Deterministic affordability calculations over a balance forecast."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from forecast import Forecast


class MissingMinimumBalanceError(ValueError):
    """Raised when a request has no minimum-balance policy."""


def _date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _money(value: Any, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid amount") from error
    if result < 0:
        raise ValueError(f"{field_name} must not be negative")
    return result


def _request_date(request: Mapping[str, Any]) -> date:
    value = request.get("request_date")
    if value is None:
        raise ValueError("request is missing request_date")
    return _date(value)


def _requested_amount(request: Mapping[str, Any]) -> Decimal:
    value = request.get("requested_amount")
    if value is None:
        raise ValueError("request is missing requested_amount")
    return _money(value, "requested_amount")


def _minimum_balance(request: Mapping[str, Any]) -> Decimal:
    value = request.get("minimum_balance_to_keep")
    if value is None:
        profile = request.get("financial_profile")
        if isinstance(profile, Mapping):
            value = profile.get("minimum_balance_to_keep")
    if value is None:
        raise MissingMinimumBalanceError(
            "request must provide minimum_balance_to_keep or financial_profile.minimum_balance_to_keep"
        )
    return _money(value, "minimum_balance_to_keep")


def _validate_forecast_start(request: Mapping[str, Any], forecast: Forecast) -> date:
    request_day = _request_date(request)
    if request_day != forecast.request_date:
        raise ValueError(
            "request_date must match forecast.request_date for affordability calculations"
        )
    return request_day


def amount_safe_to_pay(request: Mapping[str, Any], forecast: Forecast) -> Decimal:
    """Return the largest no-spending-change payment safe on request_date.

    The forecast already includes all modeled cash flows. A payment made on the
    request date reduces every remaining balance in the forecast by that amount,
    so the available headroom is the forecast minimum minus the protected floor.
    """
    request_day = _validate_forecast_start(request, forecast)
    requested = _requested_amount(request)
    minimum = _minimum_balance(request)
    headroom = forecast.min_balance_between(request_day, forecast.end_date) - minimum
    if headroom <= 0:
        return Decimal("0")
    return min(requested, headroom)


def earliest_date_for_full_payment(
    request: Mapping[str, Any],
    forecast: Forecast,
) -> date | None:
    """Return the first forecast date safe for one full lump-sum payment."""
    request_day = _validate_forecast_start(request, forecast)
    requested = _requested_amount(request)
    minimum = _minimum_balance(request)
    current = request_day
    while current <= forecast.end_date:
        remaining_minimum = forecast.min_balance_between(current, forecast.end_date)
        if remaining_minimum - requested >= minimum:
            return current
        current += timedelta(days=1)
    return None


__all__ = [
    "MissingMinimumBalanceError",
    "amount_safe_to_pay",
    "earliest_date_for_full_payment",
]
