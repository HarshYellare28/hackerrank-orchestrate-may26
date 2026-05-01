"""Semantic routing hints and scoring for domain/risk selection.

Generated hints are routing-only data. They must never be passed as grounded
support evidence for final answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from corpus import CorpusChunk, SUPPORTED_DOMAINS
from retrieval import BaseRetriever, RetrievalResult
from risk import RiskAssessment
from ticket_normalization import NormalizedTicket


SUPPORTED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
ROUTING_HINT_CACHE_VERSION = 2


@dataclass(frozen=True)
class RoutingHint:
    domain: str
    product_area: str
    kind: str
    terms: List[str]
    risk_category: str = ""
    generated: bool = True

    def to_chunk(self, index: int) -> CorpusChunk:
        title = "%s %s routing hint" % (self.domain, self.kind)
        if self.product_area:
            title = "%s %s" % (title, self.product_area)
        text = " ".join([title, self.risk_category] + self.terms)
        return CorpusChunk(
            chunk_id="routing:%s:%s:%s" % (self.domain, self.kind, index),
            domain=self.domain,
            product_area=self.product_area or "general_support",
            source_path="generated:routing_hints",
            title=title,
            text=text,
            metadata_extra={
                "generated": self.generated,
                "routing_only": True,
                "kind": self.kind,
                "risk_category": self.risk_category,
            },
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "domain": self.domain,
            "product_area": self.product_area,
            "kind": self.kind,
            "terms": list(self.terms),
            "risk_category": self.risk_category,
            "generated": self.generated,
        }


@dataclass(frozen=True)
class RoutingAssessment:
    domain: Optional[str]
    domain_scores: Mapping[str, float]
    high_risk: bool
    risk_reasons: List[str]
    expanded_query: str
    used_llm_triage: bool
    confidence: float


def load_or_build_routing_hints(
    chunks: Sequence[CorpusChunk],
    cache_path: Path,
    rebuild: bool,
    llm_judge: object,
) -> List[RoutingHint]:
    if cache_path.exists() and not rebuild:
        cached = _read_hints(cache_path)
        if cached:
            return cached

    hints: List[RoutingHint] = []
    if rebuild and hasattr(llm_judge, "generate_routing_hints"):
        try:
            hints = getattr(llm_judge, "generate_routing_hints")(_domain_context(chunks))
        except Exception:
            hints = []
    hints = _merge_hints(default_routing_hints(chunks), hints)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": ROUTING_HINT_CACHE_VERSION,
                "hints": [hint.as_dict() for hint in hints],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return hints


def route_ticket(
    ticket: NormalizedTicket,
    risk: RiskAssessment,
    direct_domain: Optional[str],
    first_pass_evidence: Sequence[RetrievalResult],
    routing_retriever: Optional[BaseRetriever],
    llm_judge: object,
    top_k: int = 8,
) -> RoutingAssessment:
    text = ticket.text
    seed_scores = _seed_domain_scores(ticket, direct_domain)
    evidence_scores = _evidence_domain_scores(first_pass_evidence)
    hint_results = routing_retriever.search(text, top_k=top_k) if routing_retriever and text.strip() else []
    hint_scores = _hint_domain_scores(hint_results)

    has_direct = bool(direct_domain or any(seed_scores.values()))
    weights = (0.45, 0.30, 0.25) if has_direct else (0.10, 0.30, 0.60)
    scores = {}
    for domain in sorted(SUPPORTED_DOMAINS):
        scores[domain] = (
            weights[0] * seed_scores.get(domain, 0.0)
            + weights[1] * evidence_scores.get(domain, 0.0)
            + weights[2] * hint_scores.get(domain, 0.0)
        )

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_domain, best_score = sorted_scores[0]
    runner_up = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    confidence = max(0.0, min(1.0, best_score - runner_up))
    risk_reasons = _risk_reasons_from_hints(hint_results, ticket.searchable_text)
    used_llm_triage = False

    low_confidence = best_score <= 0 or confidence < 0.08
    if (low_confidence or (risk_reasons and not risk.high_risk)) and hasattr(llm_judge, "triage_ticket"):
        try:
            triage = getattr(llm_judge, "triage_ticket")(ticket, sorted_scores, risk_reasons)
        except Exception:
            triage = None
        if isinstance(triage, Mapping):
            used_llm_triage = True
            llm_domain = str(triage.get("domain") or "").casefold()
            if llm_domain in SUPPORTED_DOMAINS:
                best_domain = llm_domain
                scores[best_domain] = max(scores.get(best_domain, 0.0), float(triage.get("confidence") or 0.0))
            for reason in triage.get("risk_categories") or []:
                if isinstance(reason, str) and reason and reason not in risk_reasons:
                    risk_reasons.append(reason)

    # Query expansion is most useful when the domain is not already explicit.
    # When the CSV gives a company hint, generic routing words such as "risk",
    # "domain", or a broad product area can drown out precise support evidence.
    semantic_terms = []
    if not has_direct or low_confidence:
        semantic_terms = _top_hint_terms(
            [
                item for item in hint_results
                if not direct_domain or item.domain == direct_domain
            ],
            limit=10,
        )
    expanded_query = " ".join([ticket.text] + semantic_terms).strip()
    domain = best_domain if scores.get(best_domain, 0.0) > 0 else None
    return RoutingAssessment(
        domain=domain,
        domain_scores=scores,
        high_risk=bool(risk_reasons),
        risk_reasons=risk_reasons,
        expanded_query=expanded_query or ticket.text,
        used_llm_triage=used_llm_triage,
        confidence=confidence,
    )


def default_routing_hints(chunks: Sequence[CorpusChunk]) -> List[RoutingHint]:
    product_areas = _product_areas(chunks)
    hints = [
        RoutingHint(
            "visa",
            "general_support",
            "domain",
            [
                "visa",
                "credit card",
                "debit card",
                "cardholder",
                "merchant",
                "charge",
                "payment network",
                "checkout",
                "travellers cheque",
                "exchange rate",
                "unauthorized transaction",
            ],
            generated=False,
        ),
        RoutingHint(
            "hackerrank",
            "general_support",
            "domain",
            [
                "hackerrank",
                "assessment",
                "coding test",
                "candidate",
                "recruiter",
                "interview",
                "test invite",
                "extra time",
                "score",
                "question library",
                "proctoring",
                "ats integration",
            ],
            generated=False,
        ),
        RoutingHint(
            "claude",
            "general_support",
            "domain",
            [
                "claude",
                "anthropic",
                "workspace",
                "conversation",
                "chat",
                "model",
                "bedrock",
                "console",
                "api key",
                "projects",
                "account",
                "subscription",
            ],
            generated=False,
        ),
        RoutingHint("visa", "fraud_protection", "risk", ["fraud", "stolen card", "lost card", "identity theft", "unauthorized transaction", "card compromised"], "fraud, stolen card, or identity theft", False),
        RoutingHint("visa", "dispute_resolution", "risk", ["dispute charge", "wrong product", "merchant issue", "merchant complaint", "refund request", "billing change", "contact issuer", "card issuer"], "billing, payment, refund, or subscription change", False),
        RoutingHint("hackerrank", "screen", "risk", ["increase my score", "review my answers", "rejected me", "next round", "candidate outcome", "score dispute"], "assessment score, recruiter, or candidate outcome dispute", False),
        RoutingHint("hackerrank", "settings", "risk", ["restore access", "admin permission", "removed my seat", "workspace owner", "remove interviewer", "remove user", "remove employee", "employee left", "user roles", "teams management"], "account access or admin-only action", False),
        RoutingHint("hackerrank", "community", "product_area", ["delete my account", "delete account", "google login account deletion", "github login account deletion", "account password reset before deletion"], "", False),
        RoutingHint("claude", "account_management", "risk", ["restore account", "workspace owner", "admin access"], "account access or admin-only action", False),
        RoutingHint("claude", "conversation_management", "product_area", ["delete conversation", "rename conversation", "private conversation", "temporary chat", "remove private info from chat"], "", False),
        RoutingHint("claude", "safeguards", "risk", ["security vulnerability", "bug bounty", "responsible disclosure", "public vulnerability reporting", "model safety bug bounty", "hackerone"], "security vulnerability or bug bounty", False),
        RoutingHint("hackerrank", "general_support", "risk", ["site is down", "platform outage", "all requests failing", "submissions unavailable"], "service outage or broad platform failure", False),
        RoutingHint("hackerrank", "interviews", "product_area", ["compatibility check", "compatible check", "zoom connectivity", "zoom domains", "system compatibility", "browser network device issues"], "", False),
        RoutingHint("hackerrank", "general_help", "product_area", ["reschedule assessment", "reschedule test", "recruiter hiring team", "not authorized to reschedule", "candidate support workflow"], "", False),
        RoutingHint("hackerrank", "community", "product_area", ["apply tab", "search and apply jobs", "job application", "find jobs", "apply for jobs"], "", False),
        RoutingHint("claude", "amazon_bedrock", "product_area", ["amazon bedrock", "aws bedrock", "all requests failing", "aws support", "customer support inquiries"], "", False),
        RoutingHint("claude", "troubleshooting", "risk", ["service outage", "all chats failing", "model unavailable", "workspace down", "all requests failing"], "service outage or broad platform failure", False),
    ]
    for domain, areas in product_areas.items():
        for area in sorted(areas):
            if area == "general_support":
                continue
            hints.append(RoutingHint(domain, area, "product_area", [area.replace("_", " "), area], generated=False))
    return hints


def _read_hints(path: Path) -> List[RoutingHint]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, Mapping) or raw.get("version") != ROUTING_HINT_CACHE_VERSION:
        return []
    raw_hints = raw.get("hints")
    hints = []
    for item in raw_hints if isinstance(raw_hints, list) else []:
        if not isinstance(item, Mapping):
            continue
        domain = str(item.get("domain") or "").casefold()
        if domain not in SUPPORTED_DOMAINS:
            continue
        terms = [str(term).casefold() for term in item.get("terms") or [] if str(term).strip()]
        if not terms:
            continue
        hints.append(
            RoutingHint(
                domain=domain,
                product_area=str(item.get("product_area") or "general_support"),
                kind=str(item.get("kind") or "domain"),
                terms=terms,
                risk_category=str(item.get("risk_category") or ""),
                generated=bool(item.get("generated", True)),
            )
        )
    return hints


def _merge_hints(defaults: Sequence[RoutingHint], generated: Sequence[object]) -> List[RoutingHint]:
    merged: Dict[tuple, RoutingHint] = {}
    for hint in list(defaults) + _coerce_generated_hints(generated):
        key = (hint.domain, hint.product_area, hint.kind, hint.risk_category)
        existing = merged.get(key)
        if existing is None:
            merged[key] = hint
        else:
            terms = sorted(set(existing.terms + hint.terms))
            merged[key] = RoutingHint(hint.domain, hint.product_area, hint.kind, terms, hint.risk_category, hint.generated)
    return list(merged.values())


def _coerce_generated_hints(items: Sequence[object]) -> List[RoutingHint]:
    hints = []
    for item in items:
        if isinstance(item, RoutingHint):
            hints.append(item)
        elif isinstance(item, Mapping):
            domain = str(item.get("domain") or "").casefold()
            terms = [str(term).casefold() for term in item.get("terms") or [] if str(term).strip()]
            if domain in SUPPORTED_DOMAINS and terms:
                hints.append(
                    RoutingHint(
                        domain=domain,
                        product_area=str(item.get("product_area") or "general_support"),
                        kind=str(item.get("kind") or "domain"),
                        terms=terms,
                        risk_category=str(item.get("risk_category") or ""),
                        generated=True,
                    )
                )
    return hints


def _domain_context(chunks: Sequence[CorpusChunk]) -> Mapping[str, object]:
    product_areas = _product_areas(chunks)
    titles: Dict[str, List[str]] = {domain: [] for domain in SUPPORTED_DOMAINS}
    for chunk in chunks:
        bucket = titles.setdefault(chunk.domain, [])
        if chunk.title not in bucket and len(bucket) < 60:
            bucket.append(chunk.title)
    return {
        domain: {
            "product_areas": sorted(product_areas.get(domain, [])),
            "sample_titles": titles.get(domain, []),
        }
        for domain in sorted(SUPPORTED_DOMAINS)
    }


def _product_areas(chunks: Sequence[CorpusChunk]) -> Dict[str, set]:
    areas: Dict[str, set] = {domain: set() for domain in SUPPORTED_DOMAINS}
    for chunk in chunks:
        areas.setdefault(chunk.domain, set()).add(chunk.product_area)
    return areas


def _seed_domain_scores(ticket: NormalizedTicket, direct_domain: Optional[str]) -> Dict[str, float]:
    scores = {domain: 0.0 for domain in SUPPORTED_DOMAINS}
    if ticket.company_hint:
        scores[ticket.company_hint] = 1.0
    elif direct_domain:
        scores[direct_domain] = 1.0
    return scores


def _evidence_domain_scores(evidence: Sequence[RetrievalResult]) -> Dict[str, float]:
    scores = {domain: 0.0 for domain in SUPPORTED_DOMAINS}
    for item in evidence:
        if item.domain in scores:
            scores[item.domain] += max(0.0, item.score)
    return _normalize(scores)


def _hint_domain_scores(hints: Sequence[RetrievalResult]) -> Dict[str, float]:
    scores = {domain: 0.0 for domain in SUPPORTED_DOMAINS}
    for item in hints:
        if item.domain in scores:
            scores[item.domain] += max(0.0, item.score)
    return _normalize(scores)


def _risk_reasons_from_hints(hints: Sequence[RetrievalResult], query_text: str) -> List[str]:
    triggers = {
        "fraud, stolen card, or identity theft": [
            "fraud",
            "stolen",
            "lost card",
            "identity theft",
            "unauthorized transaction",
            "compromised",
        ],
        "billing, payment, refund, or subscription change": [
            "refund",
            "unauthorized payment",
            "billing change",
            "pause subscription",
            "cancel subscription",
        ],
        "assessment score, recruiter, or candidate outcome dispute": [
            "increase my score",
            "review my answers",
            "rejected",
            "next round",
            "score dispute",
            "candidate outcome",
        ],
        "account access or admin-only action": [
            "restore access",
            "remove interviewer",
            "remove an interviewer",
            "remove user",
            "remove a user",
            "remove employee",
            "employee has left",
            "admin",
            "workspace owner",
            "removed my seat",
        ],
        "security vulnerability or bug bounty": [
            "security vulnerability",
            "bug bounty",
            "responsible disclosure",
            "api key compromised",
            "vulnerability",
        ],
        "service outage or broad platform failure": [
            "site is down",
            "outage",
            "all requests",
            "all chats",
            "unavailable",
            "none of the submissions",
        ],
    }
    reasons = []
    for item in hints:
        if item.score < 0.20 or item.source_path != "generated:routing_hints":
            continue
        text = " ".join([item.title, item.text]).casefold()
        for reason, needles in triggers.items():
            if reason in text and any(needle in query_text for needle in needles) and reason not in reasons:
                reasons.append(reason)
    return reasons


def _top_hint_terms(hints: Sequence[RetrievalResult], limit: int = 10) -> List[str]:
    terms = []
    for item in hints[:5]:
        for token in item.text.split():
            token = token.strip(" ,.;:()[]{}").casefold()
            if len(token) < 4 or token in terms:
                continue
            terms.append(token)
            if len(terms) >= limit:
                return terms
    return terms


def _normalize(scores: Mapping[str, float]) -> Dict[str, float]:
    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}
