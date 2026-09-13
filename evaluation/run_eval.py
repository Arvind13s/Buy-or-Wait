"""Per-field regression scoring for output.csv against sample reference rows."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence


FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize(field: str, value: Any) -> str:
    text = str(value or "").strip()
    if field == "amount_safe_to_pay":
        try:
            return str(Decimal(text).normalize())
        except (InvalidOperation, ValueError):
            return text
    return text


def score(
    output_rows: Sequence[dict[str, str]],
    sample_rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    output_by_id = {row.get("request_id", ""): row for row in output_rows}
    sample_by_id = {row.get("request_id", ""): row for row in sample_rows}
    shared_ids = sorted(set(output_by_id) & set(sample_by_id))
    output_only = sorted(set(output_by_id) - set(sample_by_id))
    sample_only = sorted(set(sample_by_id) - set(output_by_id))
    matches = Counter()
    totals = Counter()
    mismatches: list[dict[str, str]] = []
    for request_id in shared_ids:
        output = output_by_id[request_id]
        sample = sample_by_id[request_id]
        for field in FIELDS:
            totals[field] += 1
            if _normalize(field, output.get(field)) == _normalize(field, sample.get(field)):
                matches[field] += 1
            else:
                mismatches.append({
                    "request_id": request_id,
                    "field": field,
                    "expected": sample.get(field, ""),
                    "actual": output.get(field, ""),
                })
    return {
        "shared_request_count": len(shared_ids),
        "output_only_request_count": len(output_only),
        "sample_only_request_count": len(sample_only),
        "output_only_request_ids": output_only,
        "sample_only_request_ids": sample_only,
        "matches": dict(matches),
        "totals": dict(totals),
        "mismatches": mismatches,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Per-field Evaluation Regression",
        "",
        "This compares only request IDs present in both files. Sample rows are reference data and are not used as labels for the real evaluation requests.",
        "",
        f"Shared request IDs: {result['shared_request_count']}",
        f"Output-only request IDs: {result['output_only_request_count']}",
        f"Sample-only request IDs: {result['sample_only_request_count']}",
        "",
        "| Field | Matches | Compared | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field in FIELDS:
        matches = result["matches"].get(field, 0)
        total = result["totals"].get(field, 0)
        accuracy = matches / total if total else 0
        lines.append(f"| {field} | {matches} | {total} | {accuracy:.2%} |")
    lines.extend(["", f"Mismatched field values: {len(result['mismatches'])}"])
    for mismatch in result["mismatches"]:
        lines.append(
            f"- `{mismatch['request_id']}` `{mismatch['field']}`: "
            f"expected `{mismatch['expected']}`; actual `{mismatch['actual']}`"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score output.csv against sample_requests.csv")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--samples", default="dataset/sample_requests.csv")
    parser.add_argument("--report", default="evaluation/regression_report.md")
    args = parser.parse_args(argv)
    result = score(_read(Path(args.output)), _read(Path(args.samples)))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8", newline="\n")
    print(render_report(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
