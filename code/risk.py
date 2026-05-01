"""Deterministic request classification and risk gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ticket_normalization import NormalizedTicket


@dataclass(frozen=True)
class RiskAssessment:
    request_type: str
    high_risk: bool
    reasons: List[str]

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "No deterministic risk trigger matched."


def classify_request(ticket: NormalizedTicket) -> str:
    text = ticket.searchable_text
    if ticket.is_empty or ticket.is_acknowledgement:
        return "invalid"
    if any(term in text for term in ["delete all files", "wipe", "rm -rf"]):
        return "invalid"
    if "none of the submissions" in text:
        return "bug"
    if any(term in text for term in ["feature request", "can you add", "please add"]):
        return "feature_request"
    if any(term in text for term in ["down", "not working", "stopped working", "failing", "bug", "error"]):
        return "bug"
    if _looks_out_of_scope(text):
        return "invalid"
    return "product_issue"


def assess_risk(ticket: NormalizedTicket) -> RiskAssessment:
    text = ticket.searchable_text
    reasons: List[str] = []
    request_type = classify_request(ticket)

    patterns = [
        ("account access or admin-only action", ["restore my access", "removed my seat", "not the workspace owner", "remove an interviewer", "remove interviewer", "remove a user", "remove user", "employee has left", "remove them from our hackerrank hiring account"]),
        ("billing, payment, refund, or subscription change", ["refund", "payment", "order id", "pause our subscription", "pause subscription", "billing"]),
        ("fraud, stolen card, or identity theft", ["fraud", "identity", "stolen", "lost card", "card stolen", "blocked card", "carte visa a été bloquée", "carte visa bloquée"]),
        ("security vulnerability or bug bounty", ["security vulnerability", "bug bounty", "vulnerability"]),
        ("security/procurement review or custom forms", ["infosec process", "security questionnaire", "fill in the forms", "filling in the forms"]),
        ("assessment score, recruiter, or candidate outcome dispute", ["increase my score", "rejected me", "review my answers", "next round"]),
        ("service outage or broad platform failure", ["site is down", "is down", "all requests are failing", "none of the submissions", "stopped working completely"]),
        ("unsafe local system operation", ["delete all files", "wipe", "rm -rf"]),
        ("prompt injection or request to reveal internal logic", ["ignore previous", "ignore all previous", "previous rules", "internal documents", "rules internal", "documents retrieved", "logic", "règles internes", "documents récupérés", "logique exacte"]),
    ]
    for reason, needles in patterns:
        if any(needle in text for needle in needles):
            reasons.append(reason)

    return RiskAssessment(
        request_type=request_type,
        high_risk=bool(reasons),
        reasons=reasons,
    )


def infer_domain_from_text(ticket: NormalizedTicket) -> Optional[str]:
    if ticket.company_hint:
        return ticket.company_hint
    text = ticket.searchable_text
    scores = {
        "hackerrank": sum(term in text for term in ["hackerrank", "test", "assessment", "candidate", "interview", "recruiter"]),
        "claude": sum(term in text for term in ["claude", "anthropic", "bedrock", "workspace", "lti"]),
        "visa": sum(term in text for term in ["visa", "card", "merchant", "charge", "cheque"]),
    }
    best_domain = max(scores, key=scores.get)
    return best_domain if scores[best_domain] > 0 else None


def _looks_out_of_scope(text: str) -> bool:
    out_of_scope_terms = ["iron man", "actor", "weather", "recipe", "joke"]
    return any(term in text for term in out_of_scope_terms)
