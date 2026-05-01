import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from agent import AgentConfig, SupportAgent
from corpus import CorpusChunk, clean_markdown, extract_support_url, infer_product_area, load_corpus
from llm_judge import FallbackLLMJudge, HeuristicTestJudge, LLMError, ProviderLLMJudge, create_llm_judge_from_env, parse_json_object
from retrieval import BM25Retriever
from risk import assess_risk, infer_domain_from_text
from routing import RoutingHint, default_routing_hints, route_ticket
from schema import SchemaValidationError, validate_support_response
from ticket_normalization import normalize_csv_row


class CorpusAndRetrievalTest(unittest.TestCase):
    def test_loads_and_chunks_markdown_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "hackerrank" / "screen" / "managing-tests" / "test.md"
            doc.parent.mkdir(parents=True)
            doc.write_text("# Test Settings\n\nTests can have start and end times.", encoding="utf-8")

            chunks = load_corpus(root, chunk_chars=80, overlap_chars=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].domain, "hackerrank")
        self.assertEqual(chunks[0].product_area, "screen")
        self.assertEqual(chunks[0].title, "Test Settings")

    def test_bm25_retriever_returns_relevant_results(self):
        chunks = [
            CorpusChunk("1", "visa", "dispute_resolution", "data/visa/a.md", "Disputes", "Dispute a charge with your bank."),
            CorpusChunk("2", "claude", "account_management", "data/claude/b.md", "Login", "Log in to Claude with your account."),
        ]
        retriever = BM25Retriever(chunks)

        results = retriever.search("How do I dispute a Visa charge?", domain="visa", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].product_area, "dispute_resolution")

    def test_markdown_cleanup_and_product_area_inference(self):
        text = clean_markdown('---\ntitle_slug: "bad"\nsource_url: "https://example.com"\n---\n# Title\n\n<p>See [docs](https://example.com).</p>')
        self.assertIn("See docs.", text)
        self.assertNotIn("title_slug", text)
        self.assertNotIn("source_url", text)
        self.assertEqual(extract_support_url('---\nsource_url: "https://example.com/help"\n---'), "https://example.com/help")
        area = infer_product_area(Path("data"), Path("data/visa/support/consumer/travel-support/help.md"), "visa")
        self.assertEqual(area, "travel_support")


class RiskSchemaAndAgentTest(unittest.TestCase):
    def test_risk_gate_detects_high_risk_score_dispute(self):
        ticket = normalize_csv_row(
            {
                "Issue": "Please increase my score and tell the recruiter to move me to the next round.",
                "Subject": "Test Score Dispute",
                "Company": "HackerRank",
            },
            row_index=1,
        )

        risk = assess_risk(ticket)

        self.assertTrue(risk.high_risk)
        self.assertEqual(risk.request_type, "product_issue")
        self.assertEqual(infer_domain_from_text(ticket), "hackerrank")

    def test_schema_validation_rejects_bad_enum(self):
        with self.assertRaises(SchemaValidationError):
            validate_support_response(
                {
                    "status": "maybe",
                    "product_area": "",
                    "response": "x",
                    "justification": "y",
                    "request_type": "product_issue",
                }
            )

    def test_parse_json_object_handles_fenced_output(self):
        parsed = parse_json_object('```json\n{"status":"replied"}\n```')
        self.assertEqual(parsed["status"], "replied")

    def test_env_can_select_heuristic_no_credit_mode(self):
        with patch.dict(
            "os.environ",
            {"SUPPORT_AGENT_LLM_PROVIDER": "heuristic"},
            clear=True,
        ):
            judge = create_llm_judge_from_env()

        self.assertIsInstance(judge, HeuristicTestJudge)

    def test_env_can_select_openai_compatible_provider(self):
        with patch.dict(
            "os.environ",
            {
                "SUPPORT_AGENT_LLM_PROVIDER": "openai_compatible",
                "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:8000/v1",
                "OPENAI_COMPATIBLE_MODEL": "local-model",
            },
            clear=True,
        ):
            judge = create_llm_judge_from_env()

        self.assertIsInstance(judge, ProviderLLMJudge)
        self.assertEqual(judge.provider, "openai_compatible")
        self.assertEqual(judge.base_url, "http://localhost:8000/v1")

    def test_env_can_select_ollama_provider_without_api_key(self):
        with patch.dict(
            "os.environ",
            {
                "SUPPORT_AGENT_LLM_PROVIDER": "ollama",
                "OLLAMA_MODEL": "qwen2.5:1.5b",
                "OLLAMA_BASE_URL": "http://localhost:11434",
            },
            clear=True,
        ):
            judge = create_llm_judge_from_env()

        self.assertIsInstance(judge, ProviderLLMJudge)
        self.assertEqual(judge.provider, "ollama")
        self.assertEqual(judge.model, "qwen2.5:1.5b")

    def test_provider_failure_can_fall_back_to_heuristic(self):
        class BrokenJudge:
            def decide(self, ticket, domain, risk, evidence):
                raise LLMError("no credits")

        ticket = normalize_csv_row(
            {"Issue": "Thank you for helping me", "Subject": "", "Company": "None"},
            row_index=1,
        )
        decision = FallbackLLMJudge(BrokenJudge(), HeuristicTestJudge()).decide(
            ticket,
            None,
            assess_risk(ticket),
            [],
        )

        self.assertEqual(decision.payload["status"], "replied")
        self.assertEqual(decision.payload["request_type"], "invalid")

    def test_agent_uses_llm_judge_and_retriever_contract(self):
        chunks = [
            CorpusChunk(
                "1",
                "visa",
                "dispute_resolution",
                "data/visa/support/small-business/dispute-resolution.md",
                "Dispute Resolution",
                "Dispute resolution documentation explains how payment disputes are handled.",
            )
        ]
        retriever = BM25Retriever(chunks)
        agent = SupportAgent(
            config=AgentConfig(data_dir=Path("data"), index_dir=Path("data/index/qdrant")),
            llm_judge=HeuristicTestJudge(),
            retriever=retriever,
        )
        ticket = normalize_csv_row(
            {"Issue": "How do I dispute a charge?", "Subject": "Dispute charge", "Company": "Visa"},
            row_index=1,
        )

        response = agent.handle_ticket(ticket)

        self.assertEqual(response.status, "replied")
        self.assertEqual(response.product_area, "dispute_resolution")

    def test_routing_hints_route_synonyms_without_exact_domain_keywords(self):
        hints = default_routing_hints([])
        routing_retriever = BM25Retriever([hint.to_chunk(index) for index, hint in enumerate(hints, start=1)])
        ticket = normalize_csv_row(
            {"Issue": "My debit card has an unauthorized transaction.", "Subject": "", "Company": "None"},
            row_index=1,
        )

        route = route_ticket(
            ticket=ticket,
            risk=assess_risk(ticket),
            direct_domain=infer_domain_from_text(ticket),
            first_pass_evidence=[],
            routing_retriever=routing_retriever,
            llm_judge=HeuristicTestJudge(),
        )

        self.assertEqual(route.domain, "visa")

    def test_routing_hints_route_private_claude_conversation_without_admin_escalation(self):
        hints = default_routing_hints([])
        routing_retriever = BM25Retriever([hint.to_chunk(index) for index, hint in enumerate(hints, start=1)])
        ticket = normalize_csv_row(
            {
                "Issue": "I need to delete a Claude conversation because it has private info.",
                "Subject": "",
                "Company": "None",
            },
            row_index=1,
        )

        route = route_ticket(
            ticket=ticket,
            risk=assess_risk(ticket),
            direct_domain=infer_domain_from_text(ticket),
            first_pass_evidence=[],
            routing_retriever=routing_retriever,
            llm_judge=HeuristicTestJudge(),
        )

        self.assertEqual(route.domain, "claude")
        self.assertFalse(route.high_risk)
        self.assertNotIn("account access or admin-only action", route.risk_reasons)

    def test_generated_routing_hints_are_not_final_answer_evidence(self):
        class RecordingJudge(HeuristicTestJudge):
            def __init__(self):
                self.evidence = []

            def decide(self, ticket, domain, risk, evidence):
                self.evidence = evidence
                return super().decide(ticket, domain, risk, evidence)

        support_chunks = [
            CorpusChunk(
                "1",
                "visa",
                "general_support",
                "data/visa/support/consumer/visa-rules.md",
                "Visa Credit Card Rules",
                "Visa support documentation covers credit cards, debit cards, cardholders, and merchants.",
            )
        ]
        routing_hints = [
            RoutingHint(
                "visa",
                "general_support",
                "domain",
                ["debit card support", "cardholder account help"],
                generated=True,
            )
        ]
        judge = RecordingJudge()
        agent = SupportAgent(
            config=AgentConfig(
                data_dir=Path("data"),
                index_dir=Path("data/index/qdrant"),
                retriever_backend="bm25",
            ),
            llm_judge=judge,
            retriever=BM25Retriever(support_chunks),
            routing_retriever=BM25Retriever([routing_hints[0].to_chunk(1)]),
        )
        ticket = normalize_csv_row(
            {"Issue": "I need help with debit card support.", "Subject": "", "Company": "None"},
            row_index=1,
        )

        response = agent.handle_ticket(ticket)

        self.assertEqual(response.status, "replied")
        self.assertTrue(judge.evidence)
        self.assertTrue(all(item.source_path != "generated:routing_hints" for item in judge.evidence))

    def test_interviewer_removal_escalates_without_candidate_deletion_advice(self):
        chunks = [
            CorpusChunk(
                "1",
                "hackerrank",
                "settings",
                "data/hackerrank/settings/teams-management/9603546665-types-of-user-roles.md",
                "Types of User Roles",
                "Company Admins have authority to manage users, assign roles, modify entitlements, and lock or unlock users.",
            )
        ]
        agent = SupportAgent(
            config=AgentConfig(data_dir=Path("data"), index_dir=Path("data/index/qdrant"), retriever_backend="bm25"),
            llm_judge=HeuristicTestJudge(),
            retriever=BM25Retriever(chunks),
            routing_retriever=BM25Retriever([hint.to_chunk(index) for index, hint in enumerate(default_routing_hints(chunks), start=1)]),
        )
        ticket = normalize_csv_row(
            {
                "Issue": "Hello! I am trying to remove an interviewer from the platform. I am not seeing this as an option.",
                "Subject": "How to Remove a User",
                "Company": "HackerRank",
            },
            row_index=1,
        )

        response = agent.handle_ticket(ticket)

        self.assertEqual(response.status, "escalated")
        self.assertIn("company admin", response.response.casefold())
        self.assertNotIn("candidate data", response.response.casefold())

    def test_feature_down_escalates_instead_of_dumping_how_to_doc(self):
        chunks = [
            CorpusChunk(
                "1",
                "hackerrank",
                "community",
                "data/hackerrank/hackerrank_community/additional-resources/job-search-and-applications/9106957203-create-a-resume-with-resume-builder.md",
                "Create a Resume with Resume Builder",
                'title_slug: "create-a-resume-with-resume-builder" source_url: "https://example.com" The HackerRank Resume Builder helps you create a professional resume.',
            )
        ]
        agent = SupportAgent(
            config=AgentConfig(data_dir=Path("data"), index_dir=Path("data/index/qdrant"), retriever_backend="bm25"),
            llm_judge=HeuristicTestJudge(),
            retriever=BM25Retriever(chunks),
            routing_retriever=BM25Retriever([hint.to_chunk(index) for index, hint in enumerate(default_routing_hints(chunks), start=1)]),
        )
        ticket = normalize_csv_row(
            {"Issue": "Resume Builder is Down", "Subject": "Help in creating resume", "Company": "HackerRank"},
            row_index=1,
        )

        response = agent.handle_ticket(ticket)

        self.assertEqual(response.status, "escalated")
        self.assertNotIn("title_slug", response.response)
        self.assertIn("service outage", response.response.casefold())

    def test_assessment_reschedule_routes_to_recruiter_with_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = root / "data" / "hackerrank" / "general-help" / "additional-resources" / "6477583642-ensuring-a-great-candidate-experience.md"
            doc.parent.mkdir(parents=True)
            doc.write_text(
                '---\nsource_url: "https://support.hackerrank.com/articles/6477583642-ensuring-a-great-candidate-experience"\n---\n'
                "# Ensuring a Great Candidate Experience\n"
                "HackerRank does not participate in hiring decisions. The HackerRank team is not authorized to share test results, reschedule assessments or interviews, grant testing accommodations, or modify your hiring workflow.",
                encoding="utf-8",
            )
            chunks = load_corpus(root / "data")
            agent = SupportAgent(
                config=AgentConfig(data_dir=root / "data", index_dir=root / "data" / "index", retriever_backend="bm25"),
                llm_judge=HeuristicTestJudge(),
                retriever=BM25Retriever(chunks),
                routing_retriever=BM25Retriever([hint.to_chunk(index) for index, hint in enumerate(default_routing_hints(chunks), start=1)]),
            )
            ticket = normalize_csv_row(
                {
                    "Issue": "I would like to request a rescheduling of my HackerRank assessment due to unforeseen circumstances.",
                    "Subject": "",
                    "Company": "HackerRank",
                },
                row_index=1,
            )

            response = agent.handle_ticket(ticket)

        self.assertEqual(response.status, "replied")
        self.assertIn("not authorized", response.response)
        self.assertIn("Reference: https://support.hackerrank.com/articles/6477583642-ensuring-a-great-candidate-experience", response.response)
        self.assertNotIn("we will review", response.response.casefold())


if __name__ == "__main__":
    unittest.main()
