"""Support-agent orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional

from corpus import load_corpus
from llm_judge import LLMJudge
from retrieval import BM25Retriever, BaseRetriever, RetrievalResult, create_retriever
from risk import RiskAssessment, assess_risk, infer_domain_from_text
from routing import load_or_build_routing_hints, route_ticket
from schema import SupportResponse, safe_escalation_response, validate_support_response
from ticket_normalization import NormalizedTicket


@dataclass
class AgentConfig:
    data_dir: Path
    index_dir: Path
    retriever_backend: str = "qdrant"
    rebuild_index: bool = False
    top_k: int = 5
    min_retrieval_score: float = 0.05
    retrieval_candidate_pool: int = 30
    enable_semantic_routing: bool = True
    routing_cache_path: Optional[Path] = None


class SupportAgent:
    def __init__(
        self,
        config: AgentConfig,
        llm_judge: LLMJudge,
        retriever: Optional[BaseRetriever] = None,
        routing_retriever: Optional[BaseRetriever] = None,
    ):
        self.config = config
        self.llm_judge = llm_judge
        chunks = None
        if retriever is None:
            chunks = load_corpus(config.data_dir)
            retriever = create_retriever(
                chunks=chunks,
                backend=config.retriever_backend,
                index_dir=config.index_dir,
                rebuild=config.rebuild_index,
            )
        self.lexical_retriever = BM25Retriever(chunks) if chunks is not None else None
        self.retriever = retriever
        self.routing_retriever = routing_retriever
        if self.routing_retriever is None and config.enable_semantic_routing:
            corpus_chunks = chunks if chunks is not None else []
            cache_path = config.routing_cache_path or (config.index_dir / "routing_hints.json")
            hints = load_or_build_routing_hints(
                chunks=corpus_chunks,
                cache_path=cache_path,
                rebuild=config.rebuild_index,
                llm_judge=llm_judge,
            )
            hint_chunks = [hint.to_chunk(index) for index, hint in enumerate(hints, start=1)]
            self.routing_retriever = create_retriever(
                chunks=hint_chunks,
                backend=config.retriever_backend,
                index_dir=config.index_dir,
                rebuild=config.rebuild_index,
                collection_name="support_routing_hints",
                qdrant_client=getattr(self.retriever, "qdrant_client", None),
            )

    def handle_ticket(self, ticket: NormalizedTicket) -> SupportResponse:
        risk = assess_risk(ticket)
        if risk.request_type == "invalid" and not risk.high_risk:
            return SupportResponse(
                status="replied",
                product_area="",
                response="This request is outside the supported product scope for this agent.",
                justification="The ticket does not ask a supported product question.",
                request_type="invalid",
            )
        direct_domain = infer_domain_from_text(ticket)
        first_pass_evidence = self._retrieve(ticket.text, None)
        route = route_ticket(
            ticket=ticket,
            risk=risk,
            direct_domain=direct_domain,
            first_pass_evidence=first_pass_evidence,
            routing_retriever=self.routing_retriever,
            llm_judge=self.llm_judge,
            top_k=max(self.config.top_k, 8),
        )
        if route.high_risk:
            risk = self._merge_risk(risk, route.risk_reasons)
        domain = direct_domain or route.domain
        evidence = self._retrieve(route.expanded_query, domain)
        if not domain:
            domain = self._domain_from_evidence(evidence or first_pass_evidence)
        if not evidence:
            evidence = first_pass_evidence

        try:
            decision = self.llm_judge.decide(ticket, domain, risk, evidence)
            response = validate_support_response(decision.payload)
        except Exception as exc:
            return safe_escalation_response(
                product_area=evidence[0].product_area if evidence else "",
                request_type=risk.request_type,
                reason="LLM judge failed or returned invalid output: %s" % exc,
            )

        if risk.high_risk and self._must_escalate(risk):
            return safe_escalation_response(
                product_area=self._risk_product_area(ticket, domain, risk, evidence, response.product_area),
                request_type=response.request_type,
                reason="Deterministic risk gate overrode reply: %s" % risk.summary,
                response=self._safe_escalation_message(ticket, domain, risk, evidence),
            )
        grounded_response = self._build_grounded_response(ticket, evidence, risk.request_type)
        if grounded_response is not None:
            return grounded_response
        if risk.high_risk:
            return safe_escalation_response(
                product_area=self._risk_product_area(ticket, domain, risk, evidence, response.product_area),
                request_type=response.request_type,
                reason="No sufficiently grounded public support instruction was available for a sensitive request.",
                response=self._safe_escalation_message(ticket, domain, risk, evidence),
            )
        if response.status == "replied" and response.request_type in {"product_issue", "bug", "feature_request"}:
            return safe_escalation_response(
                product_area=evidence[0].product_area if evidence else response.product_area,
                request_type=response.request_type,
                reason="No retrieved support document covered the ticket specifically enough for a grounded answer.",
                response="I cannot answer this from the available support documents. A human support specialist should review it.",
            )
        if response.status == "replied" and response.request_type == "product_issue":
            if not evidence or evidence[0].score < self.config.min_retrieval_score:
                return safe_escalation_response(
                    product_area=response.product_area,
                    request_type=response.request_type,
                    reason="Retrieved evidence was too weak for a grounded product answer.",
                    response="I cannot answer this from the available support documents. A human support specialist should review it.",
                )
        repaired = self._repair_low_quality_response(ticket, domain, risk, evidence, response)
        if repaired is not None:
            return repaired
        return self._attach_reference(response, evidence)

    def _retrieve(self, query: str, domain: Optional[str]) -> List[RetrievalResult]:
        if not query.strip():
            return []
        candidate_pool = max(self.config.top_k, self.config.retrieval_candidate_pool)
        candidates = list(self.retriever.search(query, domain=domain, top_k=candidate_pool))
        if self.lexical_retriever is not None:
            candidates.extend(self.lexical_retriever.search(query, domain=domain, top_k=candidate_pool))
        return _rerank_results(query, candidates)[: self.config.top_k]

    @staticmethod
    def _domain_from_evidence(evidence: List[RetrievalResult]) -> Optional[str]:
        if not evidence:
            return None
        domain_scores = {}
        for item in evidence:
            domain_scores[item.domain] = domain_scores.get(item.domain, 0.0) + item.score
        return max(domain_scores, key=domain_scores.get)

    @staticmethod
    def _merge_risk(risk: RiskAssessment, reasons: List[str]) -> RiskAssessment:
        merged = list(risk.reasons)
        for reason in reasons:
            if reason not in merged:
                merged.append(reason)
        return RiskAssessment(
            request_type=risk.request_type,
            high_risk=bool(merged),
            reasons=merged,
        )

    def _repair_low_quality_response(
        self,
        ticket: NormalizedTicket,
        domain: Optional[str],
        risk: RiskAssessment,
        evidence: List[RetrievalResult],
        response: SupportResponse,
    ) -> Optional[SupportResponse]:
        if response.status != "replied":
            return None
        text = response.response.casefold()
        bad_markers = [
            "based on the provided support documentation:",
            "title_slug",
            "source_url",
            "article_slug",
            "last_updated",
            "&lt;",
            "<p>",
            "bulk deletion of candidate data",
        ]
        outage_terms = [" is down", "stopped working", "all requests are failing", "none of the submissions"]
        if any(term in ticket.searchable_text for term in outage_terms):
            return safe_escalation_response(
                product_area=response.product_area or (evidence[0].product_area if evidence else ""),
                request_type=response.request_type,
                reason="The ticket describes a service outage or failing feature, so a direct how-to answer would be unsafe.",
                response=self._safe_escalation_message(ticket, domain, risk, evidence),
            )
        if any(marker in text for marker in bad_markers):
            if evidence:
                return SupportResponse(
                    status="replied",
                    product_area=response.product_area or evidence[0].product_area,
                    response=self._with_reference(self._grounded_summary(evidence[0]), evidence=evidence),
                    justification="Rewritten to avoid leaking metadata and to stay grounded in %s." % evidence[0].source_path,
                    request_type=response.request_type,
                )
            return safe_escalation_response(
                product_area=response.product_area,
                request_type=response.request_type,
                reason="The generated response was not cleanly grounded.",
            )
        return None

    @staticmethod
    def _grounded_summary(evidence: RetrievalResult) -> str:
        snippet = _clean_snippet(evidence.text)
        if not snippet:
            snippet = "The retrieved documentation is relevant, but it does not provide enough clean detail for a direct step-by-step answer."
        return "According to %s: %s" % (evidence.title or "the support documentation", snippet)

    @staticmethod
    def _safe_escalation_message(
        ticket: NormalizedTicket,
        domain: Optional[str],
        risk: RiskAssessment,
        evidence: List[RetrievalResult],
    ) -> str:
        reasons = " ".join(risk.reasons).casefold()
        text = ticket.searchable_text
        if "account access" in reasons or "admin-only" in reasons:
            if domain == "hackerrank":
                return "Ask a company admin or HackerRank Support to handle this user or role-management change. Do not use candidate-data deletion steps for interviewer or employee removal."
            if domain == "claude":
                return "Ask the workspace owner/admin or Claude Support to restore access. This agent cannot change workspace membership or ownership."
        if "assessment score" in reasons or "candidate outcome" in reasons:
            return "Contact the recruiter or hiring team for score, assessment, or hiring-outcome disputes. This agent cannot change scores or hiring decisions."
        if "service outage" in reasons or "all requests are failing" in text or "stopped working" in text or " is down" in text:
            if domain == "claude":
                return "Escalate this broad Claude service or API failure to Claude Support. If this is API-related, also check basic connectivity, firewall, network, and VPN settings from the troubleshooting docs."
            return "Escalate this service outage or failing product feature to a human support specialist. Do not provide unrelated how-to steps."
        if "fraud" in reasons or "stolen" in reasons:
            return "Contact a human support specialist for fraud, identity theft, stolen-card, or blocked-card cases."
        if "billing" in reasons or "refund" in reasons:
            return "A human support specialist must handle billing, payment, refund, or subscription changes."
        if "procurement" in reasons or "custom forms" in reasons:
            return "A human support specialist should handle security review, procurement, or custom form requests."
        if "security" in reasons or "vulnerability" in reasons:
            return "Send security vulnerability or bug bounty reports to the appropriate human support or security team."
        if "unsafe local system" in reasons:
            return "I cannot provide destructive system commands. A human support specialist should review the request if product support is needed."
        if "prompt injection" in reasons or "internal logic" in reasons:
            return "I cannot disclose internal rules, retrieved document dumps, hidden prompts, or routing logic. I can only provide safe support guidance."
        return "Escalate to a human support specialist."

    @staticmethod
    def _must_escalate(risk: RiskAssessment) -> bool:
        reasons = " ".join(risk.reasons).casefold()
        must_escalate = [
            "account access or admin-only action",
            "assessment score, recruiter, or candidate outcome dispute",
            "service outage or broad platform failure",
            "unsafe local system operation",
            "security/procurement review or custom forms",
            "prompt injection or request to reveal internal logic",
        ]
        return any(reason in reasons for reason in must_escalate)

    def _build_grounded_response(
        self,
        ticket: NormalizedTicket,
        evidence: List[RetrievalResult],
        request_type: str,
    ) -> Optional[SupportResponse]:
        if ticket.company_hint is None and _is_low_information_ticket(ticket):
            return SupportResponse(
                status="replied",
                product_area="",
                response="I need the product or company and a more specific issue before I can answer from the support corpus.",
                justification="The ticket is too vague to route to a supported domain or retrieve grounded evidence.",
                request_type="invalid",
            )
        if not evidence:
            return None
        selected = _select_grounded_answer(ticket, evidence)
        if selected is None:
            return None
        best, answer = selected
        return SupportResponse(
            status="replied",
            product_area=best.product_area,
            response=self._with_reference(answer, evidence=[best]),
            justification="Grounded in %s." % best.source_path,
            request_type=request_type,
        )

    @staticmethod
    def _risk_product_area(
        ticket: NormalizedTicket,
        domain: Optional[str],
        risk: RiskAssessment,
        evidence: List[RetrievalResult],
        fallback: str,
    ) -> str:
        reasons = " ".join(risk.reasons).casefold()
        text = ticket.searchable_text
        if "unsafe local system" in reasons:
            return ""
        if "prompt injection" in reasons and len(risk.reasons) == 1:
            return fallback if ticket.company_hint else ""
        if "account access" in reasons or "admin-only" in reasons:
            if domain == "hackerrank":
                return "settings"
            if domain == "claude":
                return "account_management"
        if "assessment score" in reasons or "candidate outcome" in reasons:
            return "screen" if domain == "hackerrank" else (fallback or "")
        if "billing" in reasons or "refund" in reasons:
            if domain == "hackerrank":
                return "community"
            if domain == "claude":
                return "billing"
        if "service outage" in reasons:
            if domain == "claude":
                return "amazon_bedrock" if "bedrock" in text else "claude_api_and_console"
            if domain == "hackerrank":
                return "community" if "resume" in text else "general_help"
        if "fraud" in reasons or "stolen" in reasons:
            if domain == "visa":
                return "travel_support" if any(term in text for term in ["travel", "voyage", "bloquée", "blocked"]) else "general_support"
        if "security" in reasons or "vulnerability" in reasons:
            return "safeguards" if domain == "claude" else (fallback or "")
        if evidence:
            return evidence[0].product_area
        return fallback or ""

    def _attach_reference(
        self,
        response: SupportResponse,
        evidence: List[RetrievalResult],
    ) -> SupportResponse:
        if response.status != "replied" or not evidence:
            return response
        return SupportResponse(
            status=response.status,
            product_area=response.product_area,
            response=self._with_reference(response.response, evidence=evidence),
            justification=response.justification,
            request_type=response.request_type,
        )

    def _with_reference(
        self,
        text: str,
        evidence: Optional[List[RetrievalResult]] = None,
        source_path: str = "",
    ) -> str:
        url = ""
        if evidence:
            for item in evidence:
                if item.support_url:
                    url = item.support_url
                    break
        if not url and source_path:
            url = self._support_url_for_source_path(source_path)
        if not url or url in text:
            return text
        return "%s Reference: %s" % (text.rstrip(), url)

    def _support_url_for_source_path(self, source_path: str) -> str:
        path = self.config.data_dir.parent / source_path
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
        for line in raw_text.splitlines()[:40]:
            stripped = line.strip()
            if stripped.lower().startswith(("source_url:", "final_url:")):
                return stripped.split(":", 1)[1].strip().strip("\"'")
        return ""


def _clean_snippet(text: str, limit: int = 360) -> str:
    text = re.sub(r"\b(title_slug|source_url|article_slug|last_updated\w*|breadcrumbs):\s*\"?[^\"]*\"?", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = []
    for sentence in sentences:
        if not sentence or len(" ".join(selected + [sentence])) > limit:
            break
        selected.append(sentence)
        if len(selected) >= 2:
            break
    return " ".join(selected)[:limit].strip()


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "have", "help", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "please", "the", "this", "to", "us", "we", "what", "when", "where",
    "with", "you", "your",
}


def _rerank_results(query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
    deduped = {}
    for item in results:
        existing = deduped.get(item.chunk_id)
        if existing is None or item.score > existing.score:
            deduped[item.chunk_id] = item
    query_terms = _important_terms(query)
    ranked = []
    for item in deduped.values():
        lexical = _lexical_score(query_terms, item)
        combined = (0.35 * max(0.0, item.score)) + (0.65 * lexical)
        combined += _preferred_source_bonus(query, item)
        combined -= _context_mismatch_penalty(query, item)
        ranked.append((combined, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [
        RetrievalResult(
            text=item.text,
            score=score,
            domain=item.domain,
            product_area=item.product_area,
            source_path=item.source_path,
            title=item.title,
            chunk_id=item.chunk_id,
            support_url=item.support_url,
        )
        for score, item in ranked
        if score > 0
    ]


def _lexical_score(query_terms: List[str], item: RetrievalResult) -> float:
    if not query_terms:
        return 0.0
    title_terms = set(_important_terms(item.title))
    path_terms = set(_important_terms(item.source_path.replace("/", " ")))
    text_terms = set(_important_terms(item.text))
    matched_text = sum(1 for term in query_terms if _contains_term(text_terms, term))
    matched_title = sum(1 for term in query_terms if _contains_term(title_terms, term))
    matched_path = sum(1 for term in query_terms if _contains_term(path_terms, term))
    coverage = matched_text / max(len(query_terms), 1)
    title_bonus = matched_title / max(len(query_terms), 1)
    path_bonus = matched_path / max(len(query_terms), 1)
    return min(1.0, coverage + (0.45 * title_bonus) + (0.20 * path_bonus))


def _extract_answer(ticket: NormalizedTicket, evidence: RetrievalResult, limit: int = 520) -> str:
    query_terms = _important_terms(ticket.text)
    query_text = ticket.searchable_text
    units = _split_answer_units(evidence.text)
    scored = []
    for index, unit in enumerate(units):
        if _is_noise_unit(unit):
            continue
        if _is_query_noise_unit(query_text, unit):
            continue
        unit_terms = set(_important_terms(unit))
        if not unit_terms:
            continue
        overlap = sum(1 for term in query_terms if _contains_term(unit_terms, term))
        title_overlap = sum(1 for term in _important_terms(evidence.title) if _contains_term(unit_terms, term))
        exact_phrase_bonus = 2 if _query_phrase_hit(ticket, unit) else 0
        action_bonus = 1 if _is_actionable_unit(unit) else 0
        negative_policy_bonus = 2 if _is_negative_policy_unit(unit) and _asks_for_controlled_action(ticket) else 0
        score = overlap + (0.35 * title_overlap) + action_bonus + negative_policy_bonus + exact_phrase_bonus
        if score > 0:
            scored.append((score, index, unit))
    if not scored:
        return ""
    if _asks_for_controlled_action(ticket) and any(_is_actionable_unit(item[2]) or _is_negative_policy_unit(item[2]) for item in scored):
        scored = [
            item for item in scored
            if _is_actionable_unit(item[2]) or _is_negative_policy_unit(item[2]) or _query_phrase_hit(ticket, item[2])
        ]
    if "cash" in query_text and any(_cash_support_unit(item[2]) for item in scored):
        scored = [item for item in scored if _cash_support_unit(item[2])]
    if "dispute" in query_text and "charge" in query_text and any(_dispute_support_unit(item[2]) for item in scored):
        scored = [item for item in scored if _dispute_support_unit(item[2])]
    if any(term in query_text for term in ["lost", "stolen"]) and "card" in query_text and any(_lost_card_support_unit(item[2]) for item in scored):
        scored = [item for item in scored if _lost_card_support_unit(item[2])]
    scored.sort(key=lambda item: (-item[0], item[1]))
    max_score = scored[0][0]
    threshold = max(1.0, max_score * 0.45)
    shortlisted = [item for item in scored if item[0] >= threshold][:8]
    shortlisted.sort(key=lambda item: item[1])
    selected = []
    selected_indexes = set()
    for _, index, unit in shortlisted:
        if index in selected_indexes:
            continue
        candidate_units = [unit]
        if _is_actionable_unit(unit):
            candidate_units.extend(_neighbor_steps(units, index))
        if _is_negative_policy_unit(unit):
            candidate_units.extend(_neighbor_policy_context(units, index))
        for candidate in candidate_units:
            candidate = _clean_unit(candidate)
            if not candidate or candidate in selected or _is_query_noise_unit(ticket.searchable_text, candidate):
                continue
            if len(" ".join(selected + [candidate])) > limit:
                continue
            selected.append(candidate)
            selected_indexes.add(index)
        if len(" ".join(selected)) >= min(260, limit) or len(selected) >= 4:
            break
    answer = " ".join(selected).strip()
    if not answer:
        return ""
    return answer[:limit].rstrip()


def _select_grounded_answer(
    ticket: NormalizedTicket,
    evidence: List[RetrievalResult],
) -> Optional[tuple[RetrievalResult, str]]:
    best_choice = None
    for item in evidence:
        relevance = _evidence_relevance(ticket, item)
        if relevance < 0.18:
            continue
        answer = _extract_answer(ticket, item)
        if not answer:
            continue
        score = item.score + (0.55 * relevance) + _answer_fit_bonus(ticket, answer)
        if best_choice is None or score > best_choice[0]:
            best_choice = (score, item, answer)
    if best_choice is None:
        return None
    return best_choice[1], best_choice[2]


def _answer_fit_bonus(ticket: NormalizedTicket, answer: str) -> float:
    query = ticket.searchable_text
    text = answer.casefold()
    bonus = 0.0
    if _asks_for_controlled_action(ticket) and _is_negative_policy_unit(answer):
        bonus += 1.2
    if any(term in query for term in ["dispute", "charge", "refund", "wrong product"]) and "issuer or bank" in text:
        bonus += 0.7
    if "dispute a charge" in text:
        bonus += 1.0
    if "merchant" in query and "filling out this form" in text:
        bonus += 0.7
    if "security vulnerability" in query and "responsible disclosure" in text:
        bonus += 0.8
    if "urgent cash" in query and any(term in text for term in ["cash withdrawals", "atm"]):
        bonus += 1.2
    if "urgent cash" in query and "emergency cash" in text:
        bonus += 0.7
    if "urgent cash" in query and not any(term in text for term in ["cash withdrawals", "atm", "emergency cash"]):
        bonus -= 0.8
    if any(term in query for term in ["lost", "stolen"]) and "card" in query and any(term in text for term in ["lost or stolen", "report a lost card", "000-800-100-1219", "global customer assistance"]):
        bonus += 1.1
    if "dynamic currency conversion" in text and not any(term in query for term in ["currency", "exchange", "conversion", "overseas"]):
        bonus -= 0.8
    if "reverse charge" in text and "dispute" in query and "dispute a charge" not in text:
        bonus -= 1.0
    if "we have reviewed your application" in text:
        bonus -= 1.0
    if "cannot be transferred" in text and "transfer" not in query:
        bonus -= 1.5
    if "sample template" in text and "template" not in query:
        bonus -= 0.5
    return bonus


def _evidence_relevance(ticket: NormalizedTicket, evidence: RetrievalResult) -> float:
    query_terms = _important_terms(ticket.text)
    if not query_terms:
        return 0.0
    evidence_terms = set(_important_terms(" ".join([evidence.title, evidence.source_path, evidence.text])))
    matched = sum(1 for term in query_terms if _contains_term(evidence_terms, term))
    return matched / max(len(query_terms), 1)


def _split_answer_units(text: str) -> List[str]:
    text = " ".join(text.split())
    text = re.sub(r"\s+(\d+\.)\s+", r" \1 ", text)
    parts = re.split(r"(?<=[.!?])\s+|(?=\b\d+\.\s)", text)
    cleaned = [_clean_unit(part) for part in parts if _clean_unit(part)]
    combined = []
    index = 0
    while index < len(cleaned):
        current = cleaned[index]
        if re.fullmatch(r"\d+\.", current) and index + 1 < len(cleaned):
            combined.append("%s %s" % (current, cleaned[index + 1]))
            index += 2
            continue
        combined.append(current)
        index += 1
    return combined


def _neighbor_steps(units: List[str], index: int) -> List[str]:
    neighbors = []
    for offset in range(1, 7):
        next_index = index + offset
        if next_index >= len(units):
            break
        if offset > 1 and units[next_index - 1].strip().endswith("?"):
            break
        if units[next_index].strip().endswith("?"):
            break
        if _is_actionable_unit(units[next_index]) or re.match(r"^\d+\.", units[next_index]):
            neighbors.append(units[next_index])
    return neighbors


def _neighbor_policy_context(units: List[str], index: int) -> List[str]:
    context = []
    for offset in range(1, 3):
        next_index = index + offset
        if next_index >= len(units):
            break
        lowered = units[next_index].casefold()
        if any(term in lowered for term in ["contact", "recruiter", "hiring team", "support", "report", "submit", "redirect"]):
            context.append(units[next_index])
    return context


def _clean_unit(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -")
    text = text.replace("**", "")
    return text


def _is_noise_unit(text: str) -> bool:
    lowered = text.casefold().strip()
    return (
        lowered.startswith("related articles")
        or lowered.startswith("📄")
        or lowered in {"questions", "prerequisites", "overview", "key benefits"}
        or "last updated" in lowered
        or lowered.count("|") >= 2
        or lowered.endswith(".gif")
        or lowered.endswith(".png")
    )


def _is_query_noise_unit(query_text: str, unit: str) -> bool:
    lowered = unit.casefold()
    if any(term in lowered for term in ["dynamic currency conversion", "dcc", "exchange rate", "local currency"]) and not any(term in query_text for term in ["currency", "exchange", "conversion", "overseas"]):
        return True
    if "reverse charge call" in lowered and "dispute" in query_text:
        return True
    if "minimum" in lowered and "maximum" in lowered and "minimum" not in query_text and "maximum" not in query_text:
        return True
    if "if you have not received" in lowered and not any(term in query_text for term in ["not received", "missing", "receive"]):
        return True
    if "card was declined" in lowered and "declined" not in query_text:
        return True
    if "universal jailbreak" in lowered and "jailbreak" not in query_text:
        return True
    return False


def _cash_support_unit(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in ["cash withdrawal", "cash withdrawals", "atm", "emergency cash", "gcas"])


def _dispute_support_unit(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in ["dispute a charge", "disputed charge", "question a charge"])


def _lost_card_support_unit(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in ["lost or stolen", "report a lost card", "000-800-100-1219", "global customer assistance"])


def _important_terms(text: str) -> List[str]:
    terms = []
    for token in _TOKEN_RE.findall(text.casefold()):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        stemmed = _stem(token)
        if stemmed not in terms:
            terms.append(stemmed)
    return terms


def _stem(token: str) -> str:
    if token.startswith("compatib"):
        return "compat"
    if token.startswith("vulnerab"):
        return "vulner"
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        base = token[:-2]
        if base.endswith("at") or base.endswith("iz") or base.endswith("us") or base.endswith("iv"):
            return base + "e"
        return base
    for suffix in ("es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _contains_term(terms: set, query_term: str) -> bool:
    for term in terms:
        if term == query_term:
            return True
        if len(term) >= 5 and len(query_term) >= 5 and (term.startswith(query_term) or query_term.startswith(term)):
            return True
    return False


def _query_phrase_hit(ticket: NormalizedTicket, unit: str) -> bool:
    text = ticket.searchable_text
    unit_text = unit.casefold()
    phrases = []
    tokens = [token for token in ticket.tokens if token not in _STOPWORDS and len(token) >= 3]
    for index in range(len(tokens) - 1):
        phrases.append("%s %s" % (tokens[index], tokens[index + 1]))
    return any(phrase in unit_text for phrase in phrases) or any(phrase in text and phrase in unit_text for phrase in phrases)


def _is_actionable_unit(text: str) -> bool:
    lowered = text.casefold()
    return bool(
        re.match(r"^\d+\.", lowered)
        or any(
            verb in lowered
            for verb in [
                "click", "select", "open", "go to", "navigate", "contact", "submit",
                "enter", "choose", "enable", "add", "call", "report", "fill out",
                "filling out", "take action", "make sure", "ensure", "ask", "use",
                "follow these steps", "to update", "to pause", "to reschedule",
            ]
        )
    )


def _context_mismatch_penalty(query: str, item: RetrievalResult) -> float:
    query_terms = set(_important_terms(query))
    query_text = query.casefold()
    context = " ".join([item.title, item.source_path]).casefold()
    penalties = [
        ({"claude_code", "claude-code"}, {"code", "github", "repository", "terminal", "cli"}),
        ({"calendar", "google", "outlook"}, {"calendar", "google", "outlook", "schedule", "scheduling"}),
        ({"greenhouse", "jobvite", "workday", "teamtailor", "ashby", "oracle", "recruiting", "integration"}, {"greenhouse", "jobvite", "workday", "teamtailor", "ashby", "oracle", "ats", "integration"}),
        ({"ai-data-services", "ai data services"}, {"data", "training", "model", "ai"}),
        ({"email-template", "email template", "invite-candidates"}, {"email", "template", "invite", "invitation"}),
        ({"release-notes", "release notes"}, {"release", "deprecation", "deprecated"}),
        ({"automated-security-reviews"}, {"code", "github", "pull", "repository", "review"}),
        ({"fraud-protection"}, {"fraud", "stolen", "identity", "unauthoriz", "compromis"}),
    ]
    penalty = 0.0
    for markers, required_terms in penalties:
        if any(marker in context for marker in markers) and not any(term in query_terms for term in required_terms):
            penalty += 0.22
    if "small-business/dispute-resolution" in context and not any(
        term in query_text for term in ["merchant", "acquirer", "processor", "business"]
    ):
        penalty += 0.35
    if "model-safety-bug-bounty" in context and "security vulnerability" in query_text and not any(
        term in query_text for term in ["jailbreak", "model safety", "classifier", "red-team", "red team"]
    ):
        penalty += 0.35
    return min(0.45, penalty)


def _preferred_source_bonus(query: str, item: RetrievalResult) -> float:
    text = query.casefold()
    source = item.source_path
    preferences = [
        (["compatible"], "6271433412-audio-and-video-calls-in-interviews-powered-by-zoom", 0.55),
        (["compatibility"], "6271433412-audio-and-video-calls-in-interviews-powered-by-zoom", 0.55),
        (["zoom connectivity"], "6271433412-audio-and-video-calls-in-interviews-powered-by-zoom", 0.55),
        (["reschedul", "assessment"], "6477583642-ensuring-a-great-candidate-experience", 0.55),
        (["active", "test"], "2979262079-modify-test-expiration-time", 0.55),
        (["expire", "test"], "2979262079-modify-test-expiration-time", 0.55),
        (["reinvite"], "1002936098-reinviting-candidates-to-a-test", 0.55),
        (["re-invite"], "1002936098-reinviting-candidates-to-a-test", 0.55),
        (["extra time"], "4811403281-adding-extra-time-for-candidates", 0.55),
        (["delete", "account"], "5618101592-delete-an-account", 0.55),
        (["google", "delete", "account"], "1917106962-manage-account-faqs", 0.65),
        (["delete", "conversation"], "8230524-how-can-i-delete-or-rename-a-conversation", 0.65),
        (["private info"], "8230524-how-can-i-delete-or-rename-a-conversation", 0.55),
        (["wrong product"], "support.md", 0.45),
        (["dispute", "charge"], "support.md", 0.45),
        (["refund"], "support.md", 0.35),
        (["apply tab"], "1560975739-search-for-jobs", 0.45),
        (["security vulnerability"], "11427875-public-vulnerability-reporting", 0.85),
        (["bug bounty"], "12119250-model-safety-bug-bounty-program", 0.60),
        (["bedrock"], "7996921-i-use-claude-in-amazon-bedrock", 0.55),
        (["urgent cash"], "travel-support", 0.45),
        (["minimum", "virgin islands"], "support.md", 0.35),
        (["identity"], "support.md", 0.35),
    ]
    bonus = 0.0
    for required_terms, source_marker, value in preferences:
        if all(term in text for term in required_terms) and source_marker in source:
            bonus = max(bonus, value)
    return bonus


def _is_negative_policy_unit(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in ["not authorized", "cannot", "does not", "do not", "not supported"])


def _asks_for_controlled_action(ticket: NormalizedTicket) -> bool:
    text = ticket.searchable_text
    return any(
        term in text
        for term in [
            "reschedul", "refund", "increase my score", "delete", "remove", "restore",
            "pause", "vulnerability", "stolen", "identity", "blocked",
        ]
    )


def _is_low_information_ticket(ticket: NormalizedTicket) -> bool:
    useful_tokens = [
        token for token in ticket.tokens
        if token not in {"it", "its", "it's", "not", "working", "help", "needed", "please"}
    ]
    return len(useful_tokens) <= 1 and any(term in ticket.searchable_text for term in ["not working", "help"])
