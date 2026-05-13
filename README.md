# CodeLens 🔍
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange)
![Groq](https://img.shields.io/badge/Groq-Llama3-purple)

CodeLens is a RAG-based AI code review assistant that ingests any GitHub repository, builds a hybrid vector + AST graph index over it, and answers natural-language questions about the codebase with cited, grounded answers. It combines semantic search with structural code understanding to surface relevant context that pure embedding search misses.

## Architecture

```
GitHub URL ──► Ingestion ──► Chunker ──► Embedder (all-MiniLM-L6-v2) ──► ChromaDB
                         │
                         └──► Graph Builder (AST) ──► NetworkX Graph
                                                            │
Query ──────────────────────────────► HybridRetriever ◄────┘
                                      (vector + graph)
                                            │
                                      Groq LLM (Llama 3)
                                            │
                                    Answer + Follow-ups
```

## Features

- **Hybrid RAG pipeline** — combines dense vector retrieval (ChromaDB + sentence-transformers) with structural graph traversal (NetworkX) for higher-precision context recall
- **AST-aware chunking** — language-aware splitting via LangChain with symbol extraction (functions, classes, imports) preserved as metadata on every chunk
- **Code knowledge graph** — builds a call graph and file dependency graph from AST analysis, enabling graph-based retrieval that follows structural relationships between files
- **Groq inference** — low-latency Llama 3 completions via Groq API with automatic follow-up question generation for deeper exploration
- **Built-in evaluation** — Precision@5 scoring, answer relevance grading, and per-query logging with pass / flagged / failed states for continuous quality tracking
- **Streamlit UI** — interactive web interface with repo ingestion, query history, latency tracking, and source citations rendered alongside every answer

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/CodeLens
cd CodeLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
streamlit run app.py
```

## Tech Stack

| Component   | Technology                          | Purpose                                      |
|-------------|-------------------------------------|----------------------------------------------|
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` | Dense vector representations of code chunks |
| Vector DB   | ChromaDB                            | Persistent vector store with metadata filtering |
| LLM         | Groq API (Llama 3)                  | Low-latency answer generation and follow-ups |
| Chunking    | LangChain RecursiveCharacterTextSplitter | Language-aware, symbol-preserving chunking  |
| Graph       | NetworkX                            | AST call graph and file dependency traversal |
| Frontend    | Streamlit                           | Interactive query UI with source citations   |

## Evaluation Metrics

CodeLens tracks retrieval and generation quality on every query:

- **Precision@5** — fraction of the top-5 retrieved chunks that are genuinely relevant to the query, scored against expected source files
- **Answer relevance** — LLM-graded 0–1 score assessing whether the generated answer is grounded in the retrieved context and directly addresses the question
- **Query logging** — every query is persisted to `query_log.json` with latency, source count, relevance score, and a pass / flagged / failed state, enabling offline analysis and regression detection

## About

CodeLens is a production-grade RAG system built to demonstrate end-to-end applied ML engineering: from repository ingestion and language-aware chunking through hybrid retrieval (combining dense vector search with AST-derived graph traversal) to grounded LLM answer generation. The project showcases practical system design decisions — chunking strategy, embedding model selection, graph construction from static analysis — and includes a full evaluation harness for measuring retrieval quality. Deployed via Streamlit Community Cloud with Groq-powered inference for sub-second latency.
