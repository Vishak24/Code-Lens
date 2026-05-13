import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import time


_STOP_WORDS: frozenset[str] = frozenset({
    "what", "where", "when", "how", "why", "is", "are", "the", "a", "an",
    "in", "of", "to", "do", "does", "it", "this", "that", "and", "or",
    "for", "with", "can", "you", "me", "i", "my", "your", "its", "be",
    "was", "has", "have", "had", "not", "but", "from", "at", "by",
})


def precision_at_k(
    retrieved: list[dict],
    relevant_sources: list[str],
    k: int = 5,
) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_sources)
    hits = sum(1 for r in top_k if r.get("source", "") in relevant_set)
    return hits / len(top_k)


def answer_relevance_score(query: str, answer: str) -> float:
    query_words = {
        w.lower().strip(".,?!:;'\"()")
        for w in query.split()
        if len(w) > 2 and w.lower().strip(".,?!:;'\"()") not in _STOP_WORDS
    }
    if not query_words:
        return 0.0
    answer_lower = answer.lower()
    matches = sum(1 for word in query_words if word in answer_lower)
    return min(1.0, matches / len(query_words))


def evaluate_retrieval(
    query: str,
    retriever,
    relevant_sources: list[str],
) -> dict:
    start = time.perf_counter()
    results = retriever.retrieve(query)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "precision_at_5": precision_at_k(results, relevant_sources),
        "retrieved_sources": [r.get("source", "") for r in results],
        "latency_ms": round(latency_ms, 1),
    }


def log_query(
    query: str,
    answer: str,
    results: list[dict],
    latency_ms: float,
    relevance: float,
) -> dict:
    if relevance >= 0.7:
        check_state = "passed"
    elif relevance >= 0.4:
        check_state = "flagged"
    else:
        check_state = "failed"

    repository = ""
    if results:
        parts = Path(results[0].get("source", "")).parts
        repository = parts[0] if parts else ""

    return {
        "id": str(uuid.uuid4()),
        "query": query,
        "repository": repository,
        "latency_ms": round(latency_ms, 1),
        "sources_count": len(results),
        "relevance_score": round(relevance, 3),
        "check_state": check_state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class QueryLogger:
    def __init__(self, log_file: str = "query_log.json") -> None:
        self.log_file = Path(log_file)
        if not self.log_file.exists():
            self.log_file.write_text("[]")

    def log(self, entry: dict) -> None:
        entries = self.load()
        entries.append(entry)
        self.log_file.write_text(json.dumps(entries, indent=2))

    def load(self) -> list[dict]:
        if not self.log_file.exists():
            return []
        try:
            return json.loads(self.log_file.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def get_stats(self) -> dict:
        entries = self.load()
        if not entries:
            return {
                "total_queries": 0,
                "pass_rate": 0.0,
                "avg_relevance": 0.0,
                "avg_latency_ms": 0.0,
                "avg_sources": 0.0,
                "flagged_count": 0,
                "hallucination_count": 0,
            }
        total = len(entries)
        passed = sum(1 for e in entries if e.get("check_state") == "passed")
        flagged = sum(1 for e in entries if e.get("check_state") == "flagged")
        failed = sum(1 for e in entries if e.get("check_state") == "failed")
        return {
            "total_queries": total,
            "pass_rate": round(passed / total, 3),
            "avg_relevance": round(
                sum(e.get("relevance_score", 0.0) for e in entries) / total, 3
            ),
            "avg_latency_ms": round(
                sum(e.get("latency_ms", 0.0) for e in entries) / total, 1
            ),
            "avg_sources": round(
                sum(e.get("sources_count", 0) for e in entries) / total, 1
            ),
            "flagged_count": flagged,
            "hallucination_count": failed,
        }
