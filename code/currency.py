"""Deterministic dated currency conversion for financial event amounts."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


class MissingExchangeRateError(LookupError):
    """Raised when an exact dated currency pair is not available."""


class DuplicateExchangeRateError(ValueError):
    """Raised when the rate table contains more than one exact key."""


class InvalidExchangeRateError(ValueError):
    """Raised when a rate row cannot be used for conversion."""


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as error:
        raise ValueError(f"Invalid exchange-rate date: {value!r}") from error


def _as_decimal(value: Any, field_name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise InvalidExchangeRateError(
            f"Invalid decimal value for {field_name}: {value!r}"
        ) from error


def _currency(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _rate_key(rate_date: date | str, from_currency: Any, to_currency: Any) -> tuple[date, str, str]:
    return (
        _as_date(rate_date),
        _currency(from_currency, "from_currency"),
        _currency(to_currency, "to_currency"),
    )


class CurrencyConverter:
    """Convert amounts using only exact rates supplied at construction time."""

    def __init__(self, exchange_rates: Sequence[Mapping[str, Any]]) -> None:
        rates: dict[tuple[date, str, str], Decimal] = {}
        for row_number, row in enumerate(exchange_rates, start=2):
            try:
                key = _rate_key(
                    row.get("rate_date"),
                    row.get("from_currency"),
                    row.get("to_currency"),
                )
                rate = _as_decimal(row.get("rate"), "rate")
            except (TypeError, AttributeError) as error:
                raise InvalidExchangeRateError(
                    f"Malformed exchange-rate row {row_number}: {row!r}"
                ) from error

            if rate <= 0:
                raise InvalidExchangeRateError(
                    f"Exchange rate must be positive in row {row_number}: {rate}"
                )
            if key in rates:
                raise DuplicateExchangeRateError(
                    "Duplicate exchange rate for "
                    f"{key[0].isoformat()} {key[1]}->{key[2]}"
                )
            rates[key] = rate
        self._rates = rates

    @classmethod
    def from_csv(cls, path: str | Path) -> "CurrencyConverter":
        """Build a converter from an exchange_rates.csv file."""
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            return cls(tuple(csv.DictReader(handle)))

    def convert(
        self,
        amount: Decimal | int | float | str,
        currency: str,
        settlement_date: date | str,
        home_currency: str,
    ) -> Decimal:
        """Convert ``amount`` on the exact settlement date into home currency."""
        source = _currency(currency, "currency")
        target = _currency(home_currency, "home_currency")
        decimal_amount = _as_decimal(amount, "amount")
        if source == target:
            return decimal_amount

        key = _rate_key(settlement_date, source, target)
        try:
            rate = self._rates[key]
        except KeyError as error:
            raise MissingExchangeRateError(
                "No exact exchange rate for "
                f"{key[0].isoformat()} {key[1]}->{key[2]}"
            ) from error
        return decimal_amount * rate


def convert_to_home_currency(
    amount: Decimal | int | float | str,
    currency: str,
    settlement_date: date | str,
    home_currency: str,
    exchange_rates: Sequence[Mapping[str, Any]],
) -> Decimal:
    """Convert one amount using a supplied exchange-rate row collection."""
    return CurrencyConverter(exchange_rates).convert(
        amount=amount,
        currency=currency,
        settlement_date=settlement_date,
        home_currency=home_currency,
    )


__all__ = [
    "CurrencyConverter",
    "DuplicateExchangeRateError",
    "InvalidExchangeRateError",
    "MissingExchangeRateError",
    "convert_to_home_currency",
]
