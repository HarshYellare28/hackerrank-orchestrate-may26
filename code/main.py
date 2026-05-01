"""Terminal entry point for the multi-domain support RAG agent."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent import AgentConfig, SupportAgent
from llm_judge import LLMError, create_llm_judge_from_env
from schema import OUTPUT_COLUMNS
from ticket_normalization import normalize_csv_row


def parse_args() -> argparse.Namespace:
    repo_root = CODE_DIR.parent
    parser = argparse.ArgumentParser(description="Run the support-ticket RAG agent.")
    parser.add_argument("--input", default=str(repo_root / "support_tickets" / "support_tickets.csv"))
    parser.add_argument("--output", default=str(repo_root / "support_tickets" / "output.csv"))
    parser.add_argument("--data-dir", default=str(repo_root / "data"))
    parser.add_argument("--index-dir", default=str(repo_root / "data" / "index" / "qdrant"))
    parser.add_argument("--retriever", choices=["qdrant", "bm25"], default="qdrant")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--issue", default="", help="Process one ticket directly from the terminal.")
    parser.add_argument("--subject", default="", help="Subject for --issue single-ticket mode.")
    parser.add_argument("--company", default="", help="Company hint for --issue single-ticket mode.")
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_available()
    args = parse_args()
    try:
        llm_judge = create_llm_judge_from_env()
    except LLMError as exc:
        print("LLM configuration error: %s" % exc, file=sys.stderr)
        print(
            "Copy .env.example to .env and choose openai, anthropic, openai_compatible, ollama, or heuristic.",
            file=sys.stderr,
        )
        return 2
    config = AgentConfig(
        data_dir=Path(args.data_dir),
        index_dir=Path(args.index_dir),
        retriever_backend=args.retriever,
        rebuild_index=args.rebuild_index,
        top_k=args.top_k,
    )
    print(
        "Loading corpus with %s retriever%s..."
        % (args.retriever, " and rebuilding index" if args.rebuild_index else "")
    )
    agent = SupportAgent(config=config, llm_judge=llm_judge)

    if args.issue:
        ticket = normalize_csv_row(
            {
                "Issue": args.issue,
                "Subject": args.subject,
                "Company": args.company,
            },
            row_index=1,
        )
        response = agent.handle_ticket(ticket)
        print(json.dumps(response.to_csv_row(), ensure_ascii=False, indent=2))
        return 0

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print("Processing %s tickets from %s..." % (len(rows), input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            print("Processing ticket %s/%s" % (index, len(rows)))
            ticket = normalize_csv_row(row, row_index=index)
            writer.writerow(agent.handle_ticket(ticket).to_csv_row())

    print("Wrote %s rows to %s" % (len(rows), output_path))
    return 0


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
