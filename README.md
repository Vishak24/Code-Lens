# CodeLens 🔍

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange)
![Groq](https://img.shields.io/badge/Groq-Llama3-purple)

RAG-based AI code review assistant — paste any GitHub repo, ask questions, get cited answers.

## How It Works

1. Paste a GitHub repo URL
2. CodeLens clones the repo and builds a hybrid vector + AST graph index
3. Ask natural language questions about the codebase
4. Get grounded answers with cited file paths and line numbers

## Architecture

```
Ingestion:  GitPython clone → file walk → LangChain chunking → AST parsing → ChromaDB embeddings
Query:      semantic search (ChromaDB) + graph traversal (NetworkX) → Groq Llama3 → cited answer
```

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

## Setup

```bash
git clone https://github.com/Vishak24/Code-Lens
cd Code-Lens
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
streamlit run app.py
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description | Where to get it |
|----------|-------------|-----------------|
| `GROQ_API_KEY` | API key for Groq inference (Llama 3) | [console.groq.com](https://console.groq.com) |

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Dense vector representations of code chunks |
| Vector DB | ChromaDB | Persistent vector store with metadata filtering |
| LLM | Groq API (Llama 3) | Low-latency answer generation and follow-ups |
| Chunking | LangChain RecursiveCharacterTextSplitter | Language-aware, symbol-preserving chunking |
| Graph | NetworkX | AST call graph and file dependency traversal |
| Frontend | Streamlit | Interactive query UI with source citations |

## Roadmap

- [ ] Repo Health Score (complexity, doc coverage, test ratio, duplication)
- [ ] Multi-repo workspace support
- [ ] VS Code extension
- [ ] GitHub Actions integration

