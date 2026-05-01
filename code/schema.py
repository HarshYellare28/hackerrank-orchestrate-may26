"""Output schema validation for evaluator-compatible support predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


ALLOWED_STATUSES = frozenset({"replied", "escalated"})
ALLOWED_REQUEST_TYPES = frozenset({"product_issue", "feature_request", "bug", "invalid"})
OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


class SchemaValidationError(ValueError):
    """Raised when an agent output cannot be converted to evaluator schema."""


@dataclass(frozen=True)
class SupportResponse:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
        }


def validate_support_response(payload: Mapping[str, Any]) -> SupportResponse:
    status = _clean_enum(payload.get("status", ""))
    request_type = _clean_enum(payload.get("request_type", ""))
    product_area = _clean_text(payload.get("product_area", ""))
    response = _clean_text(payload.get("response", ""))
    justification = _clean_text(payload.get("justification", ""))

    if status not in ALLOWED_STATUSES:
        raise SchemaValidationError("status must be one of: %s" % sorted(ALLOWED_STATUSES))
    if request_type not in ALLOWED_REQUEST_TYPES:
        raise SchemaValidationError(
            "request_type must be one of: %s" % sorted(ALLOWED_REQUEST_TYPES)
        )
    if not response:
        raise SchemaValidationError("response is required")
    if not justification:
        raise SchemaValidationError("justification is required")

    return SupportResponse(
        status=status,
        product_area=product_area,
        response=response,
        justification=justification,
        request_type=request_type,
    )


def safe_escalation_response(
    product_area: str,
    request_type: str,
    reason: str,
    response: str = "Escalate to a human support specialist.",
) -> SupportResponse:
    if request_type not in ALLOWED_REQUEST_TYPES:
        request_type = "product_issue"
    return SupportResponse(
        status="escalated",
        product_area=product_area,
        response=response,
        justification=reason,
        request_type=request_type,
    )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean_enum(value: Any) -> str:
    return _clean_text(value).casefold()
