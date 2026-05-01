"""Small Ollama connectivity check for the support-agent judge."""

from __future__ import annotations

import json
import os
import sys
from urllib import request
from urllib.error import URLError


def main() -> int:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return only this JSON object with no markdown: "
                    '{"status":"replied","product_area":"","response":"ok",'
                    '"justification":"ok","request_type":"invalid","evidence_supported":false}'
                ),
            }
        ],
        "options": {"temperature": 0},
    }
    req = request.Request(
        "%s/api/chat" % base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        print("Ollama check failed: %s" % exc, file=sys.stderr)
        return 2

    content = body.get("message", {}).get("content", "").strip()
    if not content:
        print("Ollama responded, but message.content was empty.", file=sys.stderr)
        return 3
    try:
        json.loads(content)
    except json.JSONDecodeError:
        print("Ollama responded, but the content was not JSON:", file=sys.stderr)
        print(content, file=sys.stderr)
        return 4
    print("Ollama is reachable at %s with model %s." % (base_url, model))
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
