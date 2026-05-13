import numpy as np
import networkx as nx
from chromadb import Collection
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(
        self,
        collection: Collection,
        graph: nx.DiGraph,
        model: SentenceTransformer,
    ) -> None:
        self.collection = collection
        self.graph = graph
        self.model = model

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        # Step 1: semantic search
        query_vec: list[float] = self.model.encode(
            [query], show_progress_bar=False
        )[0].tolist()

        semantic = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Step 2: graph hop — neighbors of each result's source file node.
        # 1-hop gives us the file's own symbols (functions/classes/imports).
        # We also check those symbols' neighbors to surface cross-file CALLS targets.
        initial_sources: set[str] = set()
        neighbor_sources: set[str] = set()

        for meta in semantic["metadatas"][0]:
            source = meta.get("source", "")
            initial_sources.add(source)
            file_id = f"file:{source}"
            if file_id not in self.graph:
                continue

            for nbr_id in self.graph.neighbors(file_id):
                nbr_data = self.graph.nodes.get(nbr_id, {})
                nbr_source = nbr_data.get("source", "")
                if nbr_source and nbr_source != source:
                    neighbor_sources.add(nbr_source)

                # Second hop: follow CALLS edges out of local function nodes
                for nbr2_id in self.graph.neighbors(nbr_id):
                    nbr2_data = self.graph.nodes.get(nbr2_id, {})
                    nbr2_source = nbr2_data.get("source", "")
                    if nbr2_source and nbr2_source != source:
                        neighbor_sources.add(nbr2_source)

        neighbor_sources -= initial_sources

        # Step 3: fetch neighbor chunks from ChromaDB using stored embeddings
        # for accurate re-ranking without re-encoding
        graph_items: list[dict] = []
        q_arr = np.array(query_vec, dtype=np.float32)

        for nbr_source in neighbor_sources:
            try:
                resp = self.collection.get(
                    where={"source": nbr_source},
                    include=["documents", "metadatas", "embeddings"],
                )
                for doc, meta, emb in zip(
                    resp.get("documents", []),
                    resp.get("metadatas", []),
                    resp.get("embeddings", []),
                ):
                    emb_arr = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(emb_arr) * np.linalg.norm(q_arr)
                    score = float(emb_arr.dot(q_arr) / (norm + 1e-8))
                    graph_items.append({"doc": doc, "meta": meta, "score": score})
            except Exception:
                pass

        # Step 4: merge + deduplicate by (source, chunk_index), re-rank by score
        seen: set[str] = set()
        merged: list[dict] = []

        def _cid(meta: dict) -> str:
            return f"{meta.get('source', '')}:{meta.get('chunk_index', 0)}"

        def _to_result(doc: str, meta: dict, score: float) -> dict:
            return {
                "text": doc,
                "source": meta.get("source", ""),
                "language": meta.get("language", ""),
                "score": score,
                "chunk_index": meta.get("chunk_index", 0),
                "symbols": [s for s in meta.get("symbols", "").split(",") if s],
            }

        for doc, meta, dist in zip(
            semantic["documents"][0],
            semantic["metadatas"][0],
            semantic["distances"][0],
        ):
            cid = _cid(meta)
            if cid not in seen:
                seen.add(cid)
                merged.append(_to_result(doc, meta, 1.0 - dist))

        for item in graph_items:
            cid = _cid(item["meta"])
            if cid not in seen:
                seen.add(cid)
                merged.append(_to_result(item["doc"], item["meta"], item["score"]))

        # Step 5: sort by semantic score, return top n_results
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:n_results]

    def get_context_string(self, results: list[dict]) -> str:
        return "".join(f"File: {r['source']}\n{r['text']}\n---\n" for r in results)
