"""Load and chunk the local support corpus."""

from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re
from typing import Any, Iterable, List, Mapping, Optional


SUPPORTED_DOMAINS = frozenset({"hackerrank", "claude", "visa"})
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_METADATA_LINE_RE = re.compile(
    r"^(title_slug|source_url|final_url|article_slug|article_id|last_updated|last_updated_exact|last_updated_relative|last_updated_iso|description|breadcrumbs):",
    re.IGNORECASE,
)
_URL_METADATA_RE = re.compile(r"^(source_url|final_url):\s*[\"']?([^\"']+)[\"']?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    domain: str
    product_area: str
    source_path: str
    title: str
    text: str
    support_url: str = ""
    metadata_extra: Optional[Mapping[str, Any]] = None

    def metadata(self) -> dict:
        metadata = {
            "chunk_id": self.chunk_id,
            "domain": self.domain,
            "product_area": self.product_area,
            "source_path": self.source_path,
            "title": self.title,
            "text": self.text,
            "support_url": self.support_url,
        }
        if self.metadata_extra:
            metadata.update(dict(self.metadata_extra))
        return metadata


def load_corpus(data_dir: Path, chunk_chars: int = 1800, overlap_chars: int = 220) -> List[CorpusChunk]:
    chunks: List[CorpusChunk] = []
    for path in sorted(data_dir.rglob("*.md")):
        if path.name == "index.md":
            continue
        domain = _domain_for_path(data_dir, path)
        if domain is None:
            continue
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        clean_text = clean_markdown(raw_text)
        if not clean_text:
            continue
        title = extract_title(raw_text, path)
        support_url = extract_support_url(raw_text)
        product_area = infer_product_area(data_dir, path, domain)
        relative = path.relative_to(data_dir.parent).as_posix()
        for index, text in enumerate(chunk_text(clean_text, chunk_chars, overlap_chars)):
            chunks.append(
                CorpusChunk(
                    chunk_id="%s:%s" % (relative, index),
                    domain=domain,
                    product_area=product_area,
                    source_path=relative,
                    title=title,
                    text=text,
                    support_url=support_url,
                )
            )
    return chunks


def clean_markdown(text: str) -> str:
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _MARKDOWN_IMAGE_RE.sub(" ", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    lines = []
    in_frontmatter = False
    first_content_seen = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.casefold().startswith(("## related articles", "# related articles", "related articles")):
            break
        if not first_content_seen and stripped == "---":
            in_frontmatter = True
            first_content_seen = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if not stripped:
            continue
        first_content_seen = True
        if _METADATA_LINE_RE.match(stripped):
            continue
        if stripped.casefold().startswith(("_last updated", "_last modified", "last modified:")):
            continue
        if stripped.startswith("*") and stripped.endswith("*") and len(stripped) <= 80:
            continue
        if stripped.startswith("```"):
            continue
        stripped = stripped.replace("\\", " ")
        lines.append(stripped.lstrip("#-* ").strip())
    return " ".join(" ".join(lines).split())


def extract_title(raw_text: str, path: Path) -> str:
    match = _HEADING_RE.search(raw_text)
    if match:
        return " ".join(match.group(1).strip().split())
    return path.stem.replace("-", " ").replace("_", " ").title()


def extract_support_url(raw_text: str) -> str:
    for line in raw_text.splitlines()[:40]:
        match = _URL_METADATA_RE.match(line.strip())
        if match:
            return match.group(2).strip()
    return ""


def infer_product_area(data_dir: Path, path: Path, domain: str) -> str:
    parts = list(path.relative_to(data_dir / domain).parts)
    if not parts:
        return domain
    if domain == "hackerrank":
        first = parts[0]
        if first == "hackerrank_community":
            return "community"
        return _slug(first)
    if domain == "claude":
        if len(parts) >= 2 and parts[0] == "claude":
            return _slug(parts[1])
        return _slug(parts[0])
    if domain == "visa":
        joined = "/".join(parts)
        if "travel" in joined:
            return "travel_support"
        if "fraud" in joined:
            return "fraud_protection"
        if "dispute" in joined:
            return "dispute_resolution"
        if "merchant" in joined:
            return "merchant_support"
        return "general_support"
    return domain


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> Iterable[str]:
    if len(text) <= chunk_chars:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_chars * 0.65), end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)


def _domain_for_path(data_dir: Path, path: Path) -> Optional[str]:
    try:
        domain = path.relative_to(data_dir).parts[0]
    except (IndexError, ValueError):
        return None
    return domain if domain in SUPPORTED_DOMAINS else None


def _slug(value: str) -> str:
    return _NON_WORD_RE.sub("_", value.casefold()).strip("_") or "general_support"
