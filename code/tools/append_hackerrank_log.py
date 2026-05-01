"""Append AGENTS.md-compatible entries to the shared HackerRank log."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--branch", default="unknown")
    parser.add_argument("--worktree", default="main")
    parser.add_argument("--parent-agent", default="none")
    parser.add_argument("--tool", default="Codex")
    parser.add_argument("--action", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path.home() / "hackerrank_orchestrate" / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")
    actions = args.action or ["No file edits or commands beyond logging"]
    action_lines = "\n".join(f"* {action}" for action in actions)
    entry = f"""## [{now}] {args.title}

User Prompt (verbatim, secrets redacted):
{args.prompt}

Agent Response Summary:
{args.summary}

Actions:
{action_lines}

Context:
tool={args.tool}
branch={args.branch}
repo_root={args.repo_root}
worktree={args.worktree}
parent_agent={args.parent_agent}

"""
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
