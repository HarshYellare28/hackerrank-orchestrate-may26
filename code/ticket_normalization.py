"""Normalize raw support-ticket CSV rows into a stable internal shape."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import List, Mapping, Optional, Tuple, Union


SUPPORTED_COMPANY_HINTS = frozenset({"hackerrank", "claude", "visa"})

_FIELD_ALIASES = {
    "issue": ("issue", "Issue", "ISSUE"),
    "subject": ("subject", "Subject", "SUBJECT"),
    "company": ("company", "Company", "COMPANY"),
}

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class NormalizedTicket:
    """Clean ticket representation used by downstream agent stages."""

    row_index: int
    issue: str
    subject: str
    company_raw: str
    company_hint: Optional[str]
    text: str
    searchable_text: str
    tokens: Tuple[str, ...]
    is_empty: bool
    is_acknowledgement: bool
    raw_row: Mapping[str, str] = field(repr=False, compare=False)

    def as_row_context(self) -> Mapping[str, Union[str, int, bool, None]]:
        """Return serializable fields useful for logs, prompts, or debugging."""

        return {
            "row_index": self.row_index,
            "issue": self.issue,
            "subject": self.subject,
            "company_raw": self.company_raw,
            "company_hint": self.company_hint,
            "text": self.text,
            "searchable_text": self.searchable_text,
            "is_empty": self.is_empty,
            "is_acknowledgement": self.is_acknowledgement,
        }


def normalize_csv_row(row: Mapping[str, object], row_index: int) -> NormalizedTicket:
    """Normalize one CSV row without inferring support-domain intent.

    The normalizer cleans formatting noise and canonicalizes an explicit company
    hint when one is provided. It deliberately does not classify the ticket,
    detect product area, or decide whether the row is answerable.
    """

    raw_row = {str(key): "" if value is None else str(value) for key, value in row.items()}
    issue = normalize_text(_get_field(raw_row, "issue"))
    subject = normalize_text(_get_field(raw_row, "subject"))
    company_raw = normalize_text(_get_field(raw_row, "company"))
    company_hint = normalize_company_hint(company_raw)

    text_parts = [part for part in (subject, issue) if part]
    text = normalize_text(" ".join(text_parts))
    searchable_text = text.casefold()
    tokens = tuple(_WORD_RE.findall(searchable_text))

    return NormalizedTicket(
        row_index=row_index,
        issue=issue,
        subject=subject,
        company_raw=company_raw,
        company_hint=company_hint,
        text=text,
        searchable_text=searchable_text,
        tokens=tokens,
        is_empty=not bool(text),
        is_acknowledgement=is_acknowledgement(text),
        raw_row=raw_row,
    )


def normalize_rows(rows: List[Mapping[str, object]]) -> List[NormalizedTicket]:
    """Normalize a batch of CSV rows using one-based row indexes."""

    return [normalize_csv_row(row, row_index=index) for index, row in enumerate(rows, start=1)]


def normalize_text(value: object) -> str:
    """Convert user-provided text into compact, Unicode-normalized text."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\ufeff", "")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_company_hint(value: object) -> Optional[str]:
    """Return a canonical company hint from the CSV field, if present."""

    normalized = normalize_text(value).casefold().strip(" .,:;-_")
    if not normalized or normalized in {"none", "null", "n/a", "na", "unknown"}:
        return None
    if normalized in SUPPORTED_COMPANY_HINTS:
        return normalized
    if "hacker" in normalized and "rank" in normalized:
        return "hackerrank"
    if "claude" in normalized or "anthropic" in normalized:
        return "claude"
    if "visa" in normalized:
        return "visa"
    return None


def is_acknowledgement(text: object) -> bool:
    """Detect low-information courtesy messages before retrieval/classification."""

    normalized = normalize_text(text).casefold().strip("!.?, ")
    if not normalized:
        return False
    acknowledgement_phrases = {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "ok",
        "okay",
        "got it",
        "appreciate it",
        "thank you for helping me",
    }
    return normalized in acknowledgement_phrases


def _get_field(row: Mapping[str, str], canonical_name: str) -> str:
    for key in _FIELD_ALIASES[canonical_name]:
        if key in row:
            return row[key]

    wanted = canonical_name.casefold()
    for key, value in row.items():
        if key.casefold().strip() == wanted:
            return value
    return ""
