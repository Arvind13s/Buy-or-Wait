"""Reconcile financial-event lifecycle rows into deterministic cash flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


class UnknownEventStateError(ValueError):
    """Raised when an event has no supported financial state."""


@dataclass(frozen=True)
class EventClassification:
    event_id: str
    lifecycle_id: str
    row: Mapping[str, Any]
    included: bool
    cash_flow: Decimal
    effective_date: date | None
    reason: str


@dataclass(frozen=True)
class ReconciliationResult:
    classifications: tuple[EventClassification, ...]
    included_events: tuple[EventClassification, ...]

    @property
    def excluded_events(self) -> tuple[EventClassification, ...]:
        return tuple(event for event in self.classifications if not event.included)


def _text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    return str(value).strip() if value is not None else ""


def _state(row: Mapping[str, Any]) -> str:
    value = _text(row, "state") or _text(row, "status")
    if not value:
        raise UnknownEventStateError(
            f"Event {row.get('event_id', '<missing-id>')} has no state"
        )
    return value.lower()


def _parse_date(value: Any) -> date | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError as error:
            raise ValueError(f"Invalid event date: {value!r}") from error


def _amount(row: Mapping[str, Any]) -> Decimal:
    try:
        value = Decimal(str(row.get("amount")))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(
            f"Event {row.get('event_id', '<missing-id>')} has invalid amount"
        ) from error
    if value < 0:
        raise ValueError(
            f"Event {row.get('event_id', '<missing-id>')} has negative amount"
        )
    return value


def _is_non_cash(row: Mapping[str, Any]) -> bool:
    direction = _text(row, "direction").lower().replace("_", "-")
    event_type = _text(row, "event_type").lower()
    return direction in {"non-cash", "noncash"} or event_type == "investment_valuation"


def _effective_date(row: Mapping[str, Any], state: str) -> date | None:
    if state == "scheduled":
        values = (row.get("scheduled_date"), row.get("settlement_date"), row.get("event_date"))
    else:
        values = (row.get("settlement_date"), row.get("event_date"))
    for value in values:
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _classify(row: Mapping[str, Any]) -> EventClassification:
    state = _state(row)
    event_id = _text(row, "event_id") or "<missing-id>"
    lifecycle_id = event_id
    direction = _text(row, "direction").lower()
    is_debit = direction == "debit"
    effective_date = _effective_date(row, state)

    if _is_non_cash(row) or state == "unrealized":
        return EventClassification(
            event_id, lifecycle_id, row, False, Decimal("0"), effective_date,
            "non-cash or unrealized investment value",
        )
    if state in {"failed", "cancelled"}:
        return EventClassification(
            event_id, lifecycle_id, row, False, Decimal("0"), effective_date,
            f"{state} event excluded",
        )
    if state == "pending" and not is_debit:
        return EventClassification(
            event_id, lifecycle_id, row, False, Decimal("0"), effective_date,
            "pending credit excluded",
        )
    if state not in {"settled", "pending", "scheduled", "estimate", "forecast"}:
        raise UnknownEventStateError(f"Event {event_id} has unsupported state {state!r}")
    amount = _amount(row)
    cash_flow = -amount if is_debit else amount
    return EventClassification(
        event_id, lifecycle_id, row, True, cash_flow, effective_date,
        f"{state} cash flow included",
    )


def _lifecycle_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    parent: list[int] = list(range(len(rows)))
    event_indexes = {
        _text(row, "event_id"): index
        for index, row in enumerate(rows)
        if _text(row, "event_id")
    }

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, row in enumerate(rows):
        linked_id = _text(row, "linked_event_id")
        if linked_id in event_indexes:
            union(index, event_indexes[linked_id])

    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _is_amendment(row: Mapping[str, Any]) -> bool:
    if row.get("amends_event_id") or row.get("amendment") or row.get("is_amendment"):
        return True
    return "amend" in _text(row, "description").lower()


def _source(row: Mapping[str, Any]) -> str:
    return _text(row, "source") or _text(row, "source_type")


def _recorded_at(row: Mapping[str, Any]) -> datetime | None:
    for field in ("recorded_at", "updated_at", "created_at", "observed_at"):
        value = _text(row, field)
        if not value:
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Invalid {field} timestamp: {value!r}") from error
    return None


def _choose_winner(
    indexes: Sequence[int],
    classifications: Sequence[EventClassification],
    rows: Sequence[Mapping[str, Any]],
) -> int | None:
    if any(_state(rows[index]) == "cancelled" for index in indexes):
        return None

    active = [index for index in indexes if classifications[index].included]
    if not active:
        return None

    amendments = [index for index in active if _is_amendment(rows[index])]
    if amendments:
        return max(
            amendments,
            key=lambda index: (_recorded_at(rows[index]) or datetime.min, index),
        )

    settled = [index for index in active if _state(rows[index]) == "settled"]
    if settled:
        active = settled

    sources = {_source(rows[index]) for index in active}
    recorded = [_recorded_at(rows[index]) for index in active]
    if len(sources) == 1 and "" not in sources and all(value is not None for value in recorded):
        return max(active, key=lambda index: (_recorded_at(rows[index]), index))

    if settled:
        return min(active, key=lambda index: (classifications[index].cash_flow, index))

    # When evidence remains ambiguous, prefer the interpretation that lowers cash.
    return min(active, key=lambda index: (classifications[index].cash_flow, index))


def reconcile_events(events: Sequence[Mapping[str, Any]]) -> ReconciliationResult:
    """Classify every row, then reconcile duplicates/amendments by lifecycle."""
    rows = tuple(events)
    classifications = [_classify(row) for row in rows]
    for group in _lifecycle_groups(rows):
        winner = _choose_winner(group, classifications, rows)
        for index in group:
            current = classifications[index]
            if winner == index:
                continue
            if current.included:
                reason = "superseded by linked lifecycle record"
            elif _state(rows[index]) == "cancelled":
                reason = "cancelled event excluded"
            else:
                reason = current.reason
            classifications[index] = EventClassification(
                current.event_id,
                current.lifecycle_id,
                current.row,
                False,
                Decimal("0"),
                current.effective_date,
                reason,
            )

    included = tuple(event for event in classifications if event.included)
    return ReconciliationResult(tuple(classifications), included)


__all__ = [
    "EventClassification",
    "ReconciliationResult",
    "UnknownEventStateError",
    "reconcile_events",
]
