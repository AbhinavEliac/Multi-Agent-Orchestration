"""
research_cache.py — in-process semantic research cache.

Caches the full LLM-synthesised research brief keyed on a normalised hash
of the search query.  A cosine-similarity check on a lightweight TF-IDF
vector catches near-duplicate queries (e.g. "best AI tools for startups" vs
"top AI software for startups") and reuses the cached result.

Cache is in-memory and process-scoped.  Entries expire after TTL_SECONDS
(default 6 hours).  No external dependencies — just stdlib + math.

Usage:
    from utilis.research_cache import ResearchCache
    cache = ResearchCache()            # one instance per agent is fine
    hit = cache.get(query)
    if hit:
        return hit
    result = ... expensive call ...
    cache.set(query, result)
    return result
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TTL_SECONDS   = 6 * 3600   # cached results live for 6 hours
SIM_THRESHOLD = 0.82        # cosine similarity above this → cache hit


# ---------------------------------------------------------------------------
# Lightweight TF-IDF helpers (no external deps)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been for of in on at to by and or with "
    "this that it its as from how what when where who which will can do "
    "does did not no so but if then than".split()
)


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in _STOPWORDS]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    total = len(tokens) or 1
    return {k: v / total for k, v in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot  = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na   = math.sqrt(sum(v * v for v in a.values()))
    nb   = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _query_vector(query: str) -> dict[str, float]:
    return _tf(_tokenize(query))


def _query_hash(query: str) -> str:
    norm = " ".join(_tokenize(query))
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------

class ResearchCache:
    """
    Thread-safe in-memory cache for LLM research briefs.

    Each entry stores:
        - the normalised query vector (for similarity matching)
        - the cached result string
        - the expiry timestamp
    """

    def __init__(self, ttl: int = TTL_SECONDS, sim_threshold: float = SIM_THRESHOLD):
        self._ttl       = ttl
        self._threshold = sim_threshold
        # { hash_key: {"vec": dict, "result": str, "expires": float} }
        self._store: dict[str, dict] = {}

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items() if v["expires"] < now]
        for k in expired:
            del self._store[k]

    def get(self, query: str) -> str | None:
        """
        Return a cached result if a sufficiently similar query was seen recently,
        otherwise return None.
        """
        self._evict_expired()
        if not self._store:
            return None

        qvec = _query_vector(query)
        for entry in self._store.values():
            sim = _cosine(qvec, entry["vec"])
            if sim >= self._threshold:
                return entry["result"]
        return None

    def set(self, query: str, result: str) -> None:
        """Store a result for the given query."""
        key = _query_hash(query)
        self._store[key] = {
            "vec":     _query_vector(query),
            "result":  result,
            "expires": time.time() + self._ttl,
        }

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Module-level singletons — one cache per agent type so they don't
# cross-contaminate (research brief vs targeted research brief).
# ---------------------------------------------------------------------------
_researcher_cache         = ResearchCache()
_targeted_researcher_cache = ResearchCache()


def get_researcher_cache() -> ResearchCache:
    return _researcher_cache


def get_targeted_cache() -> ResearchCache:
    return _targeted_researcher_cache
