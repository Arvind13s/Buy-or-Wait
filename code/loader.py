"""Load the Buy or Wait? dataset into request-scoped contexts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


Row = dict[str, Any]


@dataclass(frozen=True)
class UnresolvedReference:
    source_file: str
    source_id: str
    related_event_id: str


@dataclass(frozen=True)
class RequestContext:
    request: Row
    financial_profile: Row
    financial_events: tuple[Row, ...]
    payment_options: tuple[Row, ...]
    messages: tuple[Row, ...]
    images: tuple[Row, ...]
    exchange_rates: tuple[Row, ...]
    unresolved_references: tuple[UnresolvedReference, ...]


@dataclass(frozen=True)
class LoadedDataset:
    dataset_dir: Path
    requests: tuple[Row, ...]
    sample_requests: tuple[Row, ...]
    financial_profiles: tuple[Row, ...]
    financial_events: tuple[Row, ...]
    payment_options: tuple[Row, ...]
    messages: tuple[Row, ...]
    images: tuple[Row, ...]
    exchange_rates: tuple[Row, ...]
    output_template: tuple[Row, ...]
    request_contexts: Mapping[str, RequestContext]
    unresolved_references: tuple[UnresolvedReference, ...]

    @property
    def evaluation_requests(self) -> tuple[Row, ...]:
        """Return the real evaluation set, excluding sample rows."""
        return self.requests


def _clean_row(row: Mapping[str, str | None]) -> Row:
    return {
        key: value.strip() if isinstance(value, str) and value.strip() else None
        for key, value in row.items()
    }


def _read_csv(dataset_dir: Path, filename: str) -> tuple[Row, ...]:
    path = dataset_dir / filename
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(_clean_row(row) for row in csv.DictReader(handle))


def _split_pipe(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in str(value).split("|") if part.strip())


def _parse_profile(row: Row) -> Row:
    profile = dict(row)
    profile["financial_priorities"] = _split_pipe(row.get("financial_priorities"))
    profile["expense_categories_to_protect"] = _split_pipe(
        row.get("expense_categories_to_protect")
    )
    profile["expense_categories_user_is_willing_to_reduce"] = _split_pipe(
        row.get("expense_categories_user_is_willing_to_reduce")
    )
    profile["expense_categories_user_is_willing_to_stop"] = _split_pipe(
        row.get("expense_categories_user_is_willing_to_stop")
    )
    profile["payment_methods_user_will_consider"] = _split_pipe(
        row.get("payment_methods_user_will_consider")
    )

    max_months = row.get("max_installment_months")
    profile["max_installment_months"] = int(max_months) if max_months else None
    return profile


def _add_event_state(rows: Sequence[Row]) -> tuple[Row, ...]:
    return tuple(
        {**row, "state": row.get("status")}
        for row in rows
    )


def _reference_issues(
    filename: str,
    rows: Sequence[Row],
    id_field: str,
    event_ids: set[str],
) -> tuple[UnresolvedReference, ...]:
    issues: list[UnresolvedReference] = []
    for row in rows:
        related_event_id = row.get("related_event_id")
        if related_event_id and related_event_id not in event_ids:
            issues.append(
                UnresolvedReference(
                    source_file=filename,
                    source_id=str(row.get(id_field) or "<missing-id>"),
                    related_event_id=str(related_event_id),
                )
            )
    return tuple(issues)


def _index_by(rows: Sequence[Row], field: str) -> dict[str, tuple[Row, ...]]:
    index: dict[str, list[Row]] = {}
    for row in rows:
        value = row.get(field)
        if value:
            index.setdefault(str(value), []).append(row)
    return {key: tuple(value) for key, value in index.items()}


def _image_path(dataset_dir: Path, image_id: Any) -> str | None:
    if not image_id:
        return None
    return str(dataset_dir / "media" / "images" / f"{image_id}.png")


def load_dataset(dataset_dir: str | Path = "dataset") -> LoadedDataset:
    """Load all dataset CSVs and build a context for every evaluation request.

    ``requests.csv`` and ``sample_requests.csv`` remain separate. Samples are
    available for format/reference work but are never included in evaluation
    contexts.
    """
    root = Path(dataset_dir).resolve()
    requests = _read_csv(root, "requests.csv")
    sample_requests = _read_csv(root, "sample_requests.csv")
    profiles = tuple(_parse_profile(row) for row in _read_csv(root, "financial_profiles.csv"))
    events = _add_event_state(_read_csv(root, "financial_events.csv"))
    payment_options = _read_csv(root, "request_payment_options.csv")
    messages = _read_csv(root, "messages.csv")
    images = tuple(
        {**row, "image_path": _image_path(root, row.get("image_id"))}
        for row in _read_csv(root, "images.csv")
    )
    exchange_rates = _read_csv(root, "exchange_rates.csv")
    output_template = _read_csv(root, "output.csv")

    event_ids = {
        str(event["event_id"])
        for event in events
        if event.get("event_id")
    }
    unresolved = (
        _reference_issues("messages.csv", messages, "message_id", event_ids)
        + _reference_issues("images.csv", images, "image_id", event_ids)
    )

    profiles_by_user = {
        str(profile["user_id"]): profile
        for profile in profiles
        if profile.get("user_id")
    }
    events_by_user = _index_by(events, "user_id")
    options_by_request = _index_by(payment_options, "request_id")
    messages_by_request = _index_by(messages, "request_id")
    images_by_request = _index_by(images, "request_id")

    contexts: dict[str, RequestContext] = {}
    for request in requests:
        request_id = str(request.get("request_id") or "")
        user_id = str(request.get("user_id") or "")
        if not request_id:
            raise ValueError("requests.csv contains a row without request_id")
        if user_id not in profiles_by_user:
            raise ValueError(f"No financial profile found for user_id={user_id}")

        user_events = events_by_user.get(user_id, ())
        user_event_ids = {str(event["event_id"]) for event in user_events if event.get("event_id")}
        request_messages = list(messages_by_request.get(request_id, ()))
        request_images = list(images_by_request.get(request_id, ()))

        for message in messages:
            related_event_id = message.get("related_event_id")
            if related_event_id and related_event_id in user_event_ids and message not in request_messages:
                request_messages.append(message)
        for image in images:
            related_event_id = image.get("related_event_id")
            if related_event_id and related_event_id in user_event_ids and image not in request_images:
                request_images.append(image)

        contexts[request_id] = RequestContext(
            request=request,
            financial_profile=profiles_by_user[user_id],
            financial_events=user_events,
            payment_options=options_by_request.get(request_id, ()),
            messages=tuple(request_messages),
            images=tuple(request_images),
            exchange_rates=exchange_rates,
            unresolved_references=tuple(
                issue
                for issue in unresolved
                if issue.source_id in {
                    str(message.get("message_id")) for message in request_messages
                }
                or issue.source_id in {
                    str(image.get("image_id")) for image in request_images
                }
            ),
        )

    return LoadedDataset(
        dataset_dir=root,
        requests=requests,
        sample_requests=sample_requests,
        financial_profiles=profiles,
        financial_events=events,
        payment_options=payment_options,
        messages=messages,
        images=images,
        exchange_rates=exchange_rates,
        output_template=output_template,
        request_contexts=contexts,
        unresolved_references=unresolved,
    )


__all__ = [
    "LoadedDataset",
    "RequestContext",
    "UnresolvedReference",
    "load_dataset",
]
