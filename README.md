# CodeLens

Ask questions about a GitHub repository and get answers grounded in its source code, with the file paths they came from.

CodeLens indexes a repository two ways: as **vector embeddings** of code chunks, and as a **symbol graph** of files, functions, classes, imports and cross-file calls. At query time it combines both. Semantic search finds the most relevant chunks, then the graph pulls in code those chunks depend on, even when that code doesn't share vocabulary with the question.

## How retrieval works

```
GitHub URL ─► clone (GitPython) ─► walk supported files
                │
                ├─► chunk (LangChain language-aware splitter, 1000 chars / 200 overlap)
                │        └─► embed (all-MiniLM-L6-v2) ─► ChromaDB
                │
                └─► symbol graph (NetworkX DiGraph)
                         file ──CONTAINS──► function / class
                         file ──IMPORTS───► module
                         function ──CALLS──► function in another file

Question ─► HybridRetriever
              1. top-k semantic search in ChromaDB
              2. graph expansion: from each hit's file, follow CONTAINS and CALLS edges (2 hops)
                 to collect related files
              3. fetch those files' chunks and score them by cosine similarity, using the
                 embeddings already stored in ChromaDB (no re-encoding)
              4. merge, de-duplicate by (file, chunk), re-rank, keep top-k
          ─► Groq (llama-3.1-8b-instant) ─► answer with cited files + suggested follow-ups
```

**Design notes**

- **Why a graph as well as vectors?** Pure semantic search returns the chunk that *talks about* a concept but often misses the helper it calls. Following `CALLS` edges brings that definition into context.
- **Re-ranking with stored embeddings.** Graph-expanded chunks are scored against the query with their existing ChromaDB embeddings, so expansion costs no extra model inference.
- **Symbol extraction is regex-based, not a full AST.** The graph builder uses per-language patterns for Python, JavaScript, TypeScript/TSX, Java, Go and C++. This is fast and dependency-free, but it can miss or misattribute calls (dynamic dispatch, same-named functions in different files). A tree-sitter or `ast`-based parser is the natural upgrade.

## Guardrails in the app

- GitHub URL validation and query validation before any work is done.
- Per-session rate limits: 5 repository ingestions per hour, 10 queries per minute.
- The API key is read from the environment (`.env` is git-ignored), never from the UI.

## Run it locally

Requires Python 3.11.

```bash
git clone https://github.com/Vishak24/Code-Lens
cd Code-Lens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then set GROQ_API_KEY
streamlit run app.py
```

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key for answer generation ([console.groq.com](https://console.groq.com)) |

The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB).

## Tests

```bash
pip install pytest
python -m pytest -q
```

The tests cover chunking and symbol extraction, graph construction (`CONTAINS`, `IMPORTS`, cross-file `CALLS`), `precision@k`, and the hybrid retriever's graph expansion and re-ranking (using fake vector-store and model objects, so no network or model download is needed). They run in CI on every push.

## Project layout

```
app.py                  Streamlit UI, validation, rate limiting
codelens/ingestion.py   clone / local path input, file walk, repo stats
codelens/chunker.py     language-aware chunking + symbol tags
codelens/embedder.py    sentence-transformers embeddings, ChromaDB storage
codelens/graph_builder.py  symbol graph (NetworkX)
codelens/retriever.py   hybrid vector + graph retrieval
codelens/llm.py         Groq prompt + answer generation
codelens/evaluator.py   precision@k, answer-relevance heuristic, query log
```

## Limitations and next steps

- No benchmark results yet. `evaluator.py` has `precision@k`, but there is no labelled question set to report numbers against. Building one, and comparing vector-only against hybrid retrieval, is the next step.
- Regex symbol extraction (see above).
- One repository per session. The index is local to the machine running the app.
