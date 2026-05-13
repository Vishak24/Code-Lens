from functools import lru_cache
from typing import Any

import chromadb
from chromadb import Collection
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from sentence_transformers import SentenceTransformer

from codelens.models import CodeChunk, EmbeddedChunk

_MODEL_NAME = "all-MiniLM-L6-v2"
_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed_chunks(chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
    model = load_model()
    results: list[EmbeddedChunk] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Embedding chunks..."),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("embed", total=len(chunks))
        for i in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[i : i + _BATCH_SIZE]
            vectors = model.encode([c.text for c in batch], show_progress_bar=False)
            for chunk, vec in zip(batch, vectors):
                results.append(EmbeddedChunk(**chunk.model_dump(), embedding=vec.tolist()))
            progress.advance(task, len(batch))

    return results


def init_chromadb(persist_dir: str = ".chromadb") -> Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name="codelens",
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(collection: Collection, embedded_chunks: list[EmbeddedChunk]) -> None:
    if not embedded_chunks:
        return

    collection.upsert(
        ids=[c.id for c in embedded_chunks],
        embeddings=[c.embedding for c in embedded_chunks],
        documents=[c.text for c in embedded_chunks],
        metadatas=[
            {
                "source": c.source,
                "language": c.language,
                "repo_name": c.repo_name,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "start_line": c.start_line,
                "symbols": ",".join(c.symbols),
            }
            for c in embedded_chunks
        ],
    )


def query_similar(
    collection: Collection,
    query_text: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    model = load_model()
    query_vec = model.encode([query_text])[0].tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "language": meta.get("language", ""),
            "score": 1.0 - dist,  # cosine distance → similarity score
            "metadata": meta,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
