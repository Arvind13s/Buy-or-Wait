"""Buy or Wait? entry point.

Run from the repository root with:

    python3 code/main.py

Set API_KEY, BASE_URL, and MODEL before running. The
program writes output.csv at the
repository root and code/evaluation/usage_report.md.
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import csv
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from affordability import amount_safe_to_pay, earliest_date_for_full_payment
from currency import CurrencyConverter
from extraction import (
    LLMClient,
    OpenAICompatibleClient,
    UsageTracker,
    explain_decision,
    extract_blank_event_amounts,
    interpret_relevant_messages,
)
from forecast import build_forecast
from loader import LoadedDataset, load_dataset
from plan_selector import PlanCandidate, build_eligible_candidates
from spending_changes import select_with_spending_changes
from validate_output import validate_output_rows


ROOT = Path(__file__).resolve().parents[1]


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _apply_message_effects(
    events: Sequence[Mapping[str, Any]],
    interpretations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    updated = [dict(event) for event in events]
    by_id = {str(event.get("event_id")): event for event in updated}
    for message_id, interpretation in interpretations.items():
        event_id = interpretation.related_event_id
        if not event_id or event_id not in by_id:
            continue
        event = by_id[event_id]
        if interpretation.action == "cancel":
            event["status"] = "cancelled"
            event["state"] = "cancelled"
        elif interpretation.action == "confirm":
            event["status"] = "settled"
            event["state"] = "settled"
        elif interpretation.action == "amend":
            if interpretation.new_amount is not None:
                event["amount"] = str(interpretation.new_amount)
            if interpretation.new_date:
                event["settlement_date"] = interpretation.new_date
                event["event_date"] = interpretation.new_date
        elif interpretation.action == "delay" and interpretation.new_date:
            event["settlement_date"] = interpretation.new_date
    return updated


def _enrich_amounts(
    events: Sequence[Mapping[str, Any]],
    images: Sequence[Mapping[str, Any]],
    client: LLMClient,
    usage: UsageTracker,
) -> list[dict[str, Any]]:
    extracted = extract_blank_event_amounts(events, images, client, usage)
    enriched: list[dict[str, Any]] = []
    for event in events:
        copy = dict(event)
        event_id = str(event.get("event_id") or "")
        if event.get("amount") in (None, ""):
            result = extracted.get(event_id)
            if result is None or result.amount is None:
                print(
                    f"Image extraction produced no amount for {event_id}; "
                    "excluding the unresolved event from cash flow.",
                    flush=True,
                )
                copy["status"] = "failed"
                copy["state"] = "failed"
                copy["extraction_notes"] = (
                    result.notes if result is not None else "No extraction result"
                )
                enriched.append(copy)
                continue
            copy["amount"] = str(result.amount)
            if result.currency:
                copy["currency"] = result.currency
        enriched.append(copy)
    return enriched


def _convert_events(
    events: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Any]],
    converter: CurrencyConverter,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for event in events:
        copy = dict(event)
        user_id = str(event.get("user_id") or "")
        profile = profiles[user_id]
        if event.get("amount") not in (None, ""):
            copy["amount"] = str(
                converter.convert(
                    event["amount"],
                    str(event.get("currency") or profile["home_currency"]),
                    event.get("settlement_date") or event.get("event_date"),
                    profile["home_currency"],
                )
            )
            copy["currency"] = profile["home_currency"]
        converted.append(copy)
    return converted


def _flexible_recurring_expenses(
    events: Sequence[Mapping[str, Any]],
    request_date: date,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if (
            str(event.get("direction") or "").lower() == "debit"
            and str(event.get("flexibility") or "").lower() == "flexible"
            and str(event.get("status") or event.get("state") or "").lower() == "settled"
            and event.get("event_date")
            and date.fromisoformat(str(event["event_date"])) < request_date
        ):
            key = (
                str(event.get("event_type") or ""),
                str(event.get("category") or ""),
                str(event.get("direction") or ""),
            )
            groups[key].append(event)
    recurring: list[dict[str, Any]] = []
    for events_in_group in groups.values():
        latest = dict(events_in_group[-1])
        latest["recurrence_dates"] = [event["event_date"] for event in events_in_group]
        recurring.append(latest)
    return recurring


def _message_contexts(data: LoadedDataset) -> dict[str, Any]:
    contexts: dict[str, Any] = {}
    for request_id, context in data.request_contexts.items():
        for message in context.messages:
            message_id = str(message.get("message_id") or "")
            contexts[message_id] = {
                "request": context.request,
                "related_event_id": message.get("related_event_id"),
                "user_id": context.request.get("user_id"),
            }
    return contexts


def _request_with_profile(request: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(request)
    enriched["financial_profile"] = profile
    enriched["minimum_balance_to_keep"] = profile["minimum_balance_to_keep"]
    enriched["payment_methods_user_will_consider"] = profile["payment_methods_user_will_consider"]
    enriched["max_installment_months"] = profile["max_installment_months"]
    return enriched


def _facts(
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    forecast: Any,
    safe_amount: Decimal,
    earliest: date | None,
    candidate: PlanCandidate,
) -> dict[str, Any]:
    return {
        "request_id": request["request_id"],
        "balance": str(forecast.forecast_balance(request["request_date"])),
        "requested_amount": str(request["requested_amount"]),
        "upcoming_expense": str(request["requested_amount"]),
        "minimum_balance": str(profile["minimum_balance_to_keep"]),
        "amount_safe_to_pay": str(safe_amount),
        "earliest_date_for_full_payment": earliest.isoformat() if earliest else None,
        "recommended_payment_method": candidate.payment_method,
        "payment_plan": candidate.payment_plan,
        "spending_changes_needed": candidate.spending_changes_needed,
    }


OUTPUT_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)
SAFE_DEFAULT_EXPLANATION = "Not fully processed before submission deadline."


def _safe_default(request_id: str) -> dict[str, str]:
    return {
        "request_id": request_id,
        "amount_safe_to_pay": "0",
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": SAFE_DEFAULT_EXPLANATION,
    }


def _write_output(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, client: LLMClient, request_id: str | None = None) -> Path:
    data = load_dataset(root / "dataset")
    if request_id is not None:
        selected = [request for request in data.requests if request.get("request_id") == request_id]
        if not selected:
            raise ValueError(f"Unknown request_id: {request_id}")
        data = data.__class__(
            dataset_dir=data.dataset_dir,
            requests=tuple(selected),
            sample_requests=data.sample_requests,
            financial_profiles=data.financial_profiles,
            financial_events=data.financial_events,
            payment_options=data.payment_options,
            messages=data.messages,
            images=data.images,
            exchange_rates=data.exchange_rates,
            output_template=data.output_template,
            request_contexts={request_id: data.request_contexts[request_id]},
            unresolved_references=data.unresolved_references,
        )
    usage = UsageTracker()
    output_path = root / ("output.csv" if request_id is None else f"single_request_{request_id}.csv")
    output_rows = [_safe_default(str(request["request_id"])) for request in data.requests]
    _write_output(output_path, output_rows)
    print(f"Initialized {len(output_rows)} safe default rows in {output_path}", flush=True)
    profiles = {str(profile["user_id"]): profile for profile in data.financial_profiles}
    converter = CurrencyConverter(data.exchange_rates)
    all_converted_events: list[dict[str, Any]] = []
    processed = 0
    try:
        for index, request in enumerate(data.requests):
            current_id = str(request["request_id"])
            try:
                user_id = str(request["user_id"])
                profile = profiles[user_id]
                context = data.request_contexts[current_id]
                enriched_events = _enrich_amounts(
                    context.financial_events,
                    context.images,
                    client,
                    usage,
                )
                interpretations = interpret_relevant_messages(
                    context.messages,
                    _message_contexts(data),
                    client,
                    usage,
                )
                enriched_events = _apply_message_effects(enriched_events, interpretations)
                converted_events = _convert_events(enriched_events, profiles, converter)
                all_converted_events.extend(converted_events)
                context_request = _request_with_profile(request, profile)
                forecast = build_forecast(
                    request["request_date"],
                    profile["current_available_balance"],
                    converted_events,
                )
                safe_amount = amount_safe_to_pay(context_request, forecast)
                earliest = earliest_date_for_full_payment(context_request, forecast)
                candidates = build_eligible_candidates(
                    context_request,
                    forecast,
                    context.payment_options,
                    safe_amount,
                    earliest,
                )
                candidate = select_with_spending_changes(
                    context_request,
                    forecast,
                    candidates,
                    _flexible_recurring_expenses(
                        converted_events,
                        date.fromisoformat(request["request_date"]),
                    ),
                )
                explanation = explain_decision(
                    _facts(context_request, profile, forecast, safe_amount, earliest, candidate),
                    client,
                    usage,
                )
                output_rows[index] = {
                    "request_id": current_id,
                    "amount_safe_to_pay": str(safe_amount),
                    "affordability_status": candidate.affordability_status,
                    "recommended_payment_method": candidate.payment_method,
                    "payment_plan": candidate.payment_plan,
                    "earliest_date_for_full_payment": earliest.isoformat() if earliest else "",
                    "spending_changes_needed": candidate.spending_changes_needed,
                    "decision_explanation": explanation,
                }
                processed += 1
                _write_output(output_path, output_rows)
                print(f"Processed {processed}/{len(data.requests)} rows", flush=True)
            except Exception as error:
                print(
                    f"Request {current_id} failed; keeping safe default: {error}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        try:
            validate_output_rows(
                output_rows,
                data.requests,
                data.payment_options,
                data.financial_profiles,
                all_converted_events,
                expected_count=len(data.requests),
            )
        except Exception as error:
            print(f"Partial output validation warning: {error}", file=sys.stderr, flush=True)
        _write_output(output_path, output_rows)
        usage.write_markdown(root / "evaluation" / "usage_report.md")
        usage.write_markdown(root / "code" / "evaluation" / "usage_report.md")
        print(f"Run finished with {processed}/{len(data.requests)} real rows", flush=True)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Buy or Wait? output.csv")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--request-id", help="Process one request instead of the full evaluation set")
    args = parser.parse_args(argv)
    try:
        client = OpenAICompatibleClient.from_env()
        output_path = run(args.root.resolve(), client, args.request_id)
    except Exception as error:
        print(f"Buy or Wait? run failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
