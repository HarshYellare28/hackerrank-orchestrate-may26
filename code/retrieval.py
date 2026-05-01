"""Pluggable corpus retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from corpus import CorpusChunk


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class RetrieverUnavailable(RuntimeError):
    """Raised when a configured retriever backend cannot be initialized."""


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    score: float
    domain: str
    product_area: str
    source_path: str
    title: str
    chunk_id: str
    support_url: str = ""

    @classmethod
    def from_chunk(cls, chunk: CorpusChunk, score: float) -> "RetrievalResult":
        return cls(
            text=chunk.text,
            score=score,
            domain=chunk.domain,
            product_area=chunk.product_area,
            source_path=chunk.source_path,
            title=chunk.title,
            chunk_id=chunk.chunk_id,
            support_url=chunk.support_url,
        )


class BaseRetriever:
    def search(self, query: str, domain: Optional[str] = None, top_k: int = 5) -> List[RetrievalResult]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """Dependency-free Okapi BM25 fallback retriever."""

    def __init__(self, chunks: Sequence[CorpusChunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._doc_tokens = [_tokenize(chunk.text + " " + chunk.title) for chunk in self._chunks]
        self._avgdl = sum(len(tokens) for tokens in self._doc_tokens) / max(len(self._doc_tokens), 1)
        self._idf = self._build_idf(self._doc_tokens)

    def search(self, query: str, domain: Optional[str] = None, top_k: int = 5) -> List[RetrievalResult]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scored = []
        for chunk, tokens in zip(self._chunks, self._doc_tokens):
            if domain and chunk.domain != domain:
                continue
            score = self._score(query_tokens, tokens)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return []
        best = scored[0][0] or 1.0
        return [RetrievalResult.from_chunk(chunk, score / best) for score, chunk in scored[:top_k]]

    def _score(self, query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
        freqs: Dict[str, int] = {}
        for token in doc_tokens:
            freqs[token] = freqs.get(token, 0) + 1
        doc_len = len(doc_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in freqs:
                continue
            tf = freqs[token]
            idf = self._idf.get(token, 0.0)
            denom = tf + self._k1 * (1 - self._b + self._b * doc_len / max(self._avgdl, 1))
            score += idf * (tf * (self._k1 + 1)) / denom
        return score

    @staticmethod
    def _build_idf(doc_tokens: Sequence[Sequence[str]]) -> Dict[str, float]:
        doc_count = len(doc_tokens)
        doc_freqs: Dict[str, int] = {}
        for tokens in doc_tokens:
            for token in set(tokens):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1
        return {
            token: log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freqs.items()
        }


class QdrantLocalRetriever(BaseRetriever):
    """Local Qdrant retriever using FastEmbed models when installed."""

    def __init__(
        self,
        chunks: Sequence[CorpusChunk],
        index_dir: Path,
        collection_name: str = "support_corpus",
        rebuild: bool = False,
        client: Optional[Any] = None,
    ):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RetrieverUnavailable("qdrant-client[fastembed] is not installed") from exc

        self._chunks = list(chunks)
        self._collection_name = collection_name
        self._models = models
        self._index_dir = index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or QdrantClient(path=str(index_dir))
        try:
            self._client.set_sparse_model("Qdrant/bm25")
        except Exception as exc:
            raise RetrieverUnavailable("Qdrant sparse BM25 model could not be initialized") from exc
        self._ensure_index(rebuild=rebuild)

    @property
    def qdrant_client(self) -> Any:
        return self._client

    def search(self, query: str, domain: Optional[str] = None, top_k: int = 5) -> List[RetrievalResult]:
        query_filter = None
        if domain:
            query_filter = self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="domain",
                        match=self._models.MatchValue(value=domain),
                    )
                ]
            )
        responses = self._client.query(
            collection_name=self._collection_name,
            query_text=query,
            query_filter=query_filter,
            limit=top_k,
        )
        results = []
        for item in responses:
            metadata = getattr(item, "metadata", None) or getattr(item, "payload", None) or {}
            score = float(getattr(item, "score", 0.0) or 0.0)
            text = getattr(item, "document", None) or metadata.get("text", "")
            results.append(
                RetrievalResult(
                    text=text,
                    score=score,
                    domain=metadata.get("domain", ""),
                    product_area=metadata.get("product_area", ""),
                    source_path=metadata.get("source_path", ""),
                    title=metadata.get("title", ""),
                    chunk_id=metadata.get("chunk_id", ""),
                    support_url=metadata.get("support_url", ""),
                )
            )
        return results

    def _ensure_index(self, rebuild: bool) -> None:
        marker = self._index_dir / (".%s_indexed" % self._collection_name)
        if rebuild:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
        if marker.exists() and not rebuild:
            return
        documents = [chunk.text for chunk in self._chunks]
        metadata = [chunk.metadata() for chunk in self._chunks]
        ids = list(range(1, len(documents) + 1))
        self._client.add(
            collection_name=self._collection_name,
            documents=documents,
            metadata=metadata,
            ids=ids,
            batch_size=32,
        )
        marker.write_text("indexed\n", encoding="utf-8")


def create_retriever(
    chunks: Sequence[CorpusChunk],
    backend: str,
    index_dir: Path,
    rebuild: bool = False,
    collection_name: str = "support_corpus",
    qdrant_client: Optional[Any] = None,
) -> BaseRetriever:
    if backend == "bm25":
        return BM25Retriever(chunks)
    if backend != "qdrant":
        raise ValueError("Unknown retriever backend: %s" % backend)
    try:
        return QdrantLocalRetriever(
            chunks,
            index_dir=index_dir,
            collection_name=collection_name,
            rebuild=rebuild,
            client=qdrant_client,
        )
    except Exception:
        return BM25Retriever(chunks)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.casefold())
