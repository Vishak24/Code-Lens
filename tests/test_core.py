"""Unit tests for CodeLens indexing and retrieval logic (no network, no model download)."""
import numpy as np

from codelens.chunker import chunk_file
from codelens.evaluator import precision_at_k
from codelens.graph_builder import build_graph
from codelens.models import CodeChunk, make_chunk_id
from codelens.retriever import HybridRetriever


def _chunk(source: str, text: str, idx: int = 0) -> CodeChunk:
    return CodeChunk(
        id=make_chunk_id("repo", source, idx), text=text, source=source,
        language="python", repo_name="repo", chunk_index=idx, total_chunks=1,
        start_line=1, symbols=[],
    )


def test_chunk_file_extracts_symbols_and_stable_ids(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("import os\n\nclass Store:\n    pass\n\ndef load(path):\n    return path\n")
    chunks = chunk_file(str(f), "python", "repo", "mod.py")
    assert len(chunks) == 1
    assert {"Store", "load"} <= set(chunks[0].symbols)
    assert chunks[0].id == make_chunk_id("repo", "mod.py", 0)


def test_build_graph_links_files_symbols_imports_and_cross_file_calls():
    g = build_graph([
        _chunk("a.py", "import json\n\ndef helper(x):\n    return x\n"),
        _chunk("b.py", "def main_flow():\n    return helper(1)\n"),
    ])
    assert g.has_edge("file:a.py", "function:a.py:helper")
    assert g.edges["file:a.py", "import:a.py:json"]["edge_type"] == "IMPORTS"
    assert g.edges["function:b.py:main_flow", "function:a.py:helper"]["edge_type"] == "CALLS"


def test_build_graph_skips_noise_names():
    g = build_graph([_chunk("c.py", "def if():\n    pass\n")])
    assert "function:c.py:if" not in g


def test_precision_at_k():
    retrieved = [{"source": "a.py"}, {"source": "b.py"}, {"source": "c.py"}]
    assert precision_at_k(retrieved, ["a.py", "c.py"], k=3) == 2 / 3
    assert precision_at_k([], ["a.py"]) == 0.0


class _FakeModel:
    def encode(self, texts, show_progress_bar=False):
        return np.array([[1.0, 0.0]])


class _FakeCollection:
    """Semantic search hits the caller (b.py); graph expansion should pull in the callee (a.py)."""

    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [["def main_flow(): helper()"]],
            "metadatas": [[{"source": "b.py", "chunk_index": 0, "symbols": "main_flow"}]],
            "distances": [[0.1]],
        }

    def get(self, where, include):
        assert where == {"source": "a.py"}
        return {
            "documents": ["def helper(): ..."],
            "metadatas": [{"source": "a.py", "chunk_index": 0, "symbols": "helper"}],
            "embeddings": [[0.6, 0.8]],
        }


def test_hybrid_retriever_follows_calls_edges_and_ranks_by_score():
    g = build_graph([
        _chunk("a.py", "def helper():\n    return 1\n"),
        _chunk("b.py", "def main_flow():\n    return helper()\n"),
    ])
    results = HybridRetriever(_FakeCollection(), g, _FakeModel()).retrieve("helper", n_results=5)
    assert [r["source"] for r in results] == ["b.py", "a.py"]
    assert results[0]["score"] == 0.9            # 1 - distance
    assert abs(results[1]["score"] - 0.6) < 1e-6  # cosine([1,0], [0.6,0.8])
    assert results[1]["symbols"] == ["helper"]
