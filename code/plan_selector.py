"""Generate and rank safe payment plans deterministically."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from affordability import amount_safe_to_pay, earliest_date_for_full_payment
from forecast import Forecast


METHODS = {"full_payment", "partial_payment", "installments", "wait"}


@dataclass(frozen=True)
class Payment:
    payment_date: date
    amount: Decimal

    def as_text(self) -> str:
        return f"{self.payment_date.isoformat()}:{self.amount}"


@dataclass(frozen=True)
class PlanCandidate:
    payment_method: str
    affordability_status: str
    payments: tuple[Payment, ...]
    total_paid: Decimal
    payment_option_id: str
    spending_changes: tuple[str, ...] = ()
    completes_by_deadline: bool = True
    safe: bool = True

    @property
    def start_date(self) -> date:
        return self.payments[0].payment_date

    @property
    def payment_count(self) -> int:
        return len(self.payments)

    @property
    def payment_plan(self) -> str:
        return "|".join(payment.as_text() for payment in self.payments) or "none"

    @property
    def spending_changes_needed(self) -> str:
        return "|".join(self.spending_changes) or "none"


class PlanSelectionError(ValueError):
    """Raised when required request or profile plan data is missing."""


def _date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _money(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise PlanSelectionError(f"{field} must be a valid amount") from error
    if result < 0:
        raise PlanSelectionError(f"{field} must not be negative")
    return result


def _requested(request: Mapping[str, Any]) -> Decimal:
    if request.get("requested_amount") is None:
        raise PlanSelectionError("request is missing requested_amount")
    return _money(request["requested_amount"], "requested_amount")


def _request_date(request: Mapping[str, Any]) -> date:
    if request.get("request_date") is None:
        raise PlanSelectionError("request is missing request_date")
    return _date(request["request_date"])


def _deadline(request: Mapping[str, Any]) -> date:
    if request.get("desired_completion_date") is None:
        raise PlanSelectionError("request is missing desired_completion_date")
    return _date(request["desired_completion_date"])


def _profile_value(request: Mapping[str, Any], field: str) -> Any:
    if field in request:
        return request[field]
    profile = request.get("financial_profile")
    if isinstance(profile, Mapping):
        return profile.get(field)
    return None


def _methods(request: Mapping[str, Any]) -> set[str]:
    value = _profile_value(request, "payment_methods_user_will_consider")
    if value is None:
        raise PlanSelectionError("request is missing payment method preferences")
    if isinstance(value, str):
        return {item.strip() for item in value.split("|") if item.strip()}
    return {str(item) for item in value}


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _minimum_balance(request: Mapping[str, Any]) -> Decimal:
    value = _profile_value(request, "minimum_balance_to_keep")
    if value is None:
        raise PlanSelectionError("request is missing minimum_balance_to_keep")
    return _money(value, "minimum_balance_to_keep")


def _safe_plan(
    plan: Sequence[Payment],
    forecast: Forecast,
    minimum_balance: Decimal,
) -> bool:
    if not plan:
        return False
    if plan[0].payment_date < forecast.request_date or plan[-1].payment_date > forecast.end_date:
        return False
    cumulative = Decimal("0")
    plan_by_day: dict[date, Decimal] = {}
    for payment in plan:
        cumulative += payment.amount
        plan_by_day[payment.payment_date] = plan_by_day.get(payment.payment_date, Decimal("0")) + payment.amount
    balance = forecast.initial_balance
    for snapshot in forecast.snapshots:
        balance = snapshot.balance
        paid = sum(
            amount
            for payment_date, amount in plan_by_day.items()
            if payment_date <= snapshot.day
        )
        if balance - paid < minimum_balance:
            return False
    return True


def _option_plan(option: Mapping[str, Any]) -> tuple[Payment, ...]:
    count = int(option.get("number_of_payments"))
    if count < 1:
        raise PlanSelectionError("number_of_payments must be positive")
    first = _date(option.get("first_payment_date"))
    amount = _money(option.get("payment_amount"), "payment_amount")
    frequency = option.get("payment_frequency_days")
    frequency_days = int(frequency) if frequency not in (None, "") else 0
    return tuple(
        Payment(first + timedelta(days=index * frequency_days), amount)
        for index in range(count)
    )


def _option_total(option: Mapping[str, Any], plan: Sequence[Payment]) -> Decimal:
    if option.get("total_payable_amount") not in (None, ""):
        return _money(option["total_payable_amount"], "total_payable_amount")
    return sum((payment.amount for payment in plan), Decimal("0"))


def _installment_allowed(option: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    max_months = _profile_value(request, "max_installment_months")
    if max_months in (None, ""):
        return False
    limit_days = _money(max_months, "max_installment_months") * Decimal("31")
    count = int(option.get("number_of_payments"))
    frequency = int(option.get("payment_frequency_days") or 0)
    return Decimal(str(max(0, count - 1) * frequency)) <= limit_days


def _candidate(
    method: str,
    status: str,
    payments: Sequence[Payment],
    total_paid: Decimal,
    option_id: str = "",
    safe: bool = True,
    deadline: date | None = None,
    spending_changes: Sequence[str] = (),
) -> PlanCandidate:
    return PlanCandidate(
        payment_method=method,
        affordability_status=status,
        payments=tuple(payments),
        total_paid=total_paid,
        payment_option_id=option_id,
        spending_changes=tuple(spending_changes),
        completes_by_deadline=deadline is None or payments[-1].payment_date <= deadline,
        safe=safe,
    )


def build_eligible_candidates(
    request: Mapping[str, Any],
    forecast: Forecast,
    payment_options: Sequence[Mapping[str, Any]],
    safe_amount: Decimal | None = None,
    full_payment_date: date | None = None,
) -> tuple[PlanCandidate, ...]:
    """Build safe candidates after preferences and schedule eligibility filters."""
    request_day = _request_date(request)
    deadline = _deadline(request)
    requested = _requested(request)
    methods = _methods(request)
    minimum = _minimum_balance(request)
    safe_amount = amount_safe_to_pay(request, forecast) if safe_amount is None else _money(safe_amount, "amount_safe_to_pay")
    full_payment_date = (
        earliest_date_for_full_payment(request, forecast)
        if full_payment_date is None
        else _date(full_payment_date)
    )
    candidates: list[PlanCandidate] = []

    if "full_payment" in methods and safe_amount >= requested:
        payment = Payment(request_day, requested)
        candidates.append(_candidate("full_payment", "affordable_now", (payment,), requested, "", deadline=deadline))

    if (
        "partial_payment" in methods
        and _bool(request.get("allows_partial_payment"))
        and Decimal("0") < safe_amount < requested
        and full_payment_date is not None
        and full_payment_date <= deadline
    ):
        payments = (
            Payment(request_day, safe_amount),
            Payment(full_payment_date, requested - safe_amount),
        )
        candidates.append(
            _candidate("partial_payment", "affordable_with_plan", payments, requested, "", deadline=deadline)
        )

    if "installments" in methods and _profile_value(request, "max_installment_months") not in (None, ""):
        for option in payment_options:
            if str(option.get("payment_method") or "") != "installments":
                continue
            if not _installment_allowed(option, request):
                continue
            payments = _option_plan(option)
            total = _option_total(option, payments)
            if payments[-1].payment_date > deadline:
                continue
            if _safe_plan(payments, forecast, minimum):
                candidates.append(
                    _candidate(
                        "installments",
                        "affordable_with_plan",
                        payments,
                        total,
                        str(option.get("payment_option_id") or ""),
                        deadline=deadline,
                    )
                )

    if (
        "full_payment" in methods
        and full_payment_date is not None
        and full_payment_date > request_day
        and full_payment_date <= deadline
    ):
        payment = Payment(full_payment_date, requested)
        candidates.append(_candidate("wait", "affordable_later", (payment,), requested, deadline=deadline))

    if not candidates:
        candidates.append(
            PlanCandidate(
                payment_method="not_recommended",
                affordability_status="not_affordable",
                payments=(),
                total_paid=Decimal("0"),
                payment_option_id="",
                safe=False,
            )
        )
    return tuple(candidates)


def _rank_key(candidate: PlanCandidate) -> tuple[Any, ...]:
    return (
        not candidate.completes_by_deadline,
        bool(candidate.spending_changes),
        candidate.total_paid,
        candidate.start_date if candidate.payments else date.max,
        candidate.payment_count,
        candidate.payment_option_id or "~",
    )


def rank_safe_candidates(candidates: Sequence[PlanCandidate]) -> tuple[PlanCandidate, ...]:
    """Return safe candidates ordered by the six required tie-break levels."""
    return tuple(sorted((candidate for candidate in candidates if candidate.safe), key=_rank_key))


def select_plan(candidates: Sequence[PlanCandidate]) -> PlanCandidate:
    """Select the best safe candidate, or the not-recommended fallback."""
    ranked = rank_safe_candidates(candidates)
    if ranked:
        return ranked[0]
    for candidate in candidates:
        if candidate.payment_method == "not_recommended":
            return candidate
    return PlanCandidate("not_recommended", "not_affordable", (), Decimal("0"), "", safe=False)


__all__ = [
    "Payment",
    "PlanCandidate",
    "PlanSelectionError",
    "build_eligible_candidates",
    "rank_safe_candidates",
    "select_plan",
]
