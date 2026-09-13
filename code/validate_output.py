"""Fail-closed validation for the final Buy or Wait? output.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)
VALID_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


class OutputValidationError(ValueError):
    """Raised when one or more output contract checks fail."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("Output validation failed:\n" + "\n".join(self.errors))


def _money(value: Any, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} is not a valid decimal") from error


def _date(value: Any) -> date:
    return date.fromisoformat(str(value).strip())


def _profile_value(profile: Mapping[str, Any], field: str) -> Any:
    return profile.get(field)


def _split_plan(value: Any, row_number: int) -> list[tuple[date, Decimal]]:
    text = str(value or "").strip()
    if text == "none":
        return []
    if not text:
        raise ValueError(f"row {row_number}: payment_plan is blank")
    payments: list[tuple[date, Decimal]] = []
    for item in text.split("|"):
        if item.count(":") != 1:
            raise ValueError(f"row {row_number}: malformed payment_plan entry {item!r}")
        date_text, amount_text = item.split(":")
        payment_date = _date(date_text)
        amount = _money(amount_text, "payment amount")
        if amount < 0:
            raise ValueError(f"row {row_number}: payment amount is negative")
        payments.append((payment_date, amount))
    if any(left > right for (left, _), (right, _) in zip(payments, payments[1:])):
        raise ValueError(f"row {row_number}: payment_plan dates are not chronological")
    return payments


def _option_plan(option: Mapping[str, Any]) -> list[tuple[date, Decimal]]:
    count = int(option["number_of_payments"])
    first = _date(option["first_payment_date"])
    amount = _money(option["payment_amount"], "payment_option payment_amount")
    frequency = int(option.get("payment_frequency_days") or 0)
    return [(first + timedelta(days=index * frequency), amount) for index in range(count)]


def _same_plan(left: Sequence[tuple[date, Decimal]], right: Sequence[tuple[date, Decimal]]) -> bool:
    return len(left) == len(right) and all(
        left_item[0] == right_item[0] and left_item[1] == right_item[1]
        for left_item, right_item in zip(left, right)
    )


def _max_installment_ok(option: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    maximum = profile.get("max_installment_months")
    if maximum in (None, ""):
        return False
    duration_days = max(0, int(option["number_of_payments"]) - 1) * int(option.get("payment_frequency_days") or 0)
    return Decimal(str(duration_days)) <= _money(maximum, "max_installment_months") * Decimal("31")


def _validate_changes(
    value: Any,
    request: Mapping[str, Any],
    events_by_id: Mapping[str, Mapping[str, Any]],
    row_number: int,
    errors: list[str],
) -> None:
    text = str(value or "").strip()
    if text == "none":
        return
    changes = text.split("|")
    if len(changes) > 3:
        errors.append(f"row {row_number}: more than three spending changes")
    seen: set[str] = set()
    for change in changes:
        if change.startswith("stop:"):
            event_id = change[len("stop:"):]
        elif change.startswith("reduce_to:"):
            parts = change.split(":")
            if len(parts) != 3:
                errors.append(f"row {row_number}: malformed spending change {change!r}")
                continue
            event_id = parts[1]
            try:
                if _money(parts[2], "reduced amount") < 0:
                    errors.append(f"row {row_number}: reduced amount is negative")
            except ValueError as error:
                errors.append(f"row {row_number}: {error}")
            
        else:
            errors.append(f"row {row_number}: malformed spending change {change!r}")
            continue
        if event_id in seen:
            errors.append(f"row {row_number}: stop/reduce_to repeated for {event_id}")
        seen.add(event_id)
        event = events_by_id.get(event_id)
        if event is None or str(event.get("user_id")) != str(request.get("user_id")):
            errors.append(f"row {row_number}: spending change targets unknown user event {event_id}")
            continue
        if str(event.get("flexibility") or "").lower() != "flexible":
            errors.append(f"row {row_number}: spending change targets non-flexible event {event_id}")


def validate_output_rows(
    output_rows: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    payment_options: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    expected_count: int = 250,
) -> None:
    """Raise OutputValidationError unless all output rows satisfy the contract."""
    errors: list[str] = []
    if len(output_rows) != expected_count:
        errors.append(f"expected exactly {expected_count} data rows, found {len(output_rows)}")
    request_by_id = {str(row.get("request_id")): row for row in requests}
    profile_by_user = {str(row.get("user_id")): row for row in profiles}
    options_by_request: dict[str, list[Mapping[str, Any]]] = {}
    for option in payment_options:
        options_by_request.setdefault(str(option.get("request_id")), []).append(option)
    events_by_id = {str(event.get("event_id")): event for event in events}

    actual_ids = [str(row.get("request_id") or "") for row in output_rows]
    if len(set(actual_ids)) != len(actual_ids):
        errors.append("output contains duplicate request_id values")
    expected_ids = set(request_by_id)
    actual_id_set = set(actual_ids)
    if actual_id_set != expected_ids:
        errors.append(
            f"request_id mismatch: missing={sorted(expected_ids - actual_id_set)[:5]}, "
            f"unexpected={sorted(actual_id_set - expected_ids)[:5]}"
        )

    for row_number, row in enumerate(output_rows, start=2):
        request_id = str(row.get("request_id") or "")
        request = request_by_id.get(request_id)
        if request is None:
            continue
        prefix = f"row {row_number} ({request_id})"
        try:
            requested = _money(request.get("requested_amount"), "requested_amount")
            safe_amount = _money(row.get("amount_safe_to_pay"), "amount_safe_to_pay")
            if not Decimal("0") <= safe_amount <= requested:
                errors.append(f"{prefix}: amount_safe_to_pay is outside request bounds")
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
            continue

        status = str(row.get("affordability_status") or "")
        method = str(row.get("recommended_payment_method") or "")
        if status not in VALID_STATUSES:
            errors.append(f"{prefix}: invalid affordability_status {status!r}")
        if method not in VALID_METHODS:
            errors.append(f"{prefix}: invalid recommended_payment_method {method!r}")
        try:
            payments = _split_plan(row.get("payment_plan"), row_number)
        except (ValueError, InvalidOperation) as error:
            errors.append(f"{prefix}: {error}")
            payments = []

        if status == "affordable_now" and str(row.get("earliest_date_for_full_payment") or "") != str(request.get("request_date") or ""):
            errors.append(f"{prefix}: affordable_now earliest date must equal request_date")
        if method == "partial_payment":
            if len(payments) != 2:
                errors.append(f"{prefix}: partial_payment must have exactly two payments")
            elif sum((amount for _, amount in payments), Decimal("0")) != requested:
                errors.append(f"{prefix}: partial_payment payments do not sum to requested_amount")
        if method == "installments":
            profile = profile_by_user.get(str(request.get("user_id")), {})
            matching = [
                option for option in options_by_request.get(request_id, ())
                if str(option.get("payment_method")) == "installments"
                and _same_plan(payments, _option_plan(option))
                and _max_installment_ok(option, profile)
            ]
            if not matching:
                errors.append(f"{prefix}: installment plan does not match an allowed payment option")

        _validate_changes(
            row.get("spending_changes_needed"),
            request,
            events_by_id,
            row_number,
            errors,
        )

    if errors:
        raise OutputValidationError(errors)


def _read_csv(path: Path) -> list[dict[str, str | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_output_file(output_path: str | Path, dataset_dir: str | Path = "dataset") -> None:
    """Validate one output CSV against the real evaluation dataset."""
    output_file = Path(output_path)
    with output_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise OutputValidationError([
                f"header must be exactly {list(REQUIRED_COLUMNS)}, found {reader.fieldnames}"
            ])
        output_rows = list(reader)
    root = Path(dataset_dir)
    validate_output_rows(
        output_rows,
        _read_csv(root / "requests.csv"),
        _read_csv(root / "request_payment_options.csv"),
        _read_csv(root / "financial_profiles.csv"),
        _read_csv(root / "financial_events.csv"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Buy or Wait? output.csv")
    parser.add_argument("output", nargs="?", default="output.csv")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args(argv)
    try:
        validate_output_file(args.output, args.dataset_dir)
    except (OSError, OutputValidationError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Validated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
