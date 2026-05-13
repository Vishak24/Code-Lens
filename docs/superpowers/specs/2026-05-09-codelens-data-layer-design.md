# CodeLens — Data Layer Design

**Date:** 2026-05-09  
**Scope:** `codelens/models.py`, `codelens/ingestion.py`, `codelens/chunker.py`

## Architecture

Three modules form the ingestion + chunking pipeline for the RAG-based code review tool:

```
codelens/
  models.py    — Pydantic v2 data contracts + make_chunk_id()
  ingestion.py — accept_input → (local_path, repo_name); clone; walk → RepoStats
  chunker.py   — chunk_file; chunk_repo → list[CodeChunk]
```

Data flow: `accept_input` → `walk_files` → `chunk_repo` → (downstream: embed, store, retrieve)

## models.py

| Model | Fields |
|-------|--------|
| `CodeChunk` | id, text, source, language, repo_name, chunk_index, total_chunks, start_line, symbols: list[str] |
| `EmbeddedChunk(CodeChunk)` | + embedding: list[float] |
| `GraphNode` | node_id, node_type, name, source, language, line_number: Optional[int] |
| `GraphEdge` | source_id, target_id, edge_type |
| `RepoStats` | repo_name, total_files, total_chunks, languages: dict[str, int], total_lines |

`make_chunk_id(repo_name, source, chunk_index) → str`: sha256 hash, returns first 16 hex chars.

## ingestion.py

- **`accept_input(url_or_path)`** → `tuple[str, str]` — detects `github.com` in string to distinguish URL from local path. Returns `(local_path, repo_name)` always.
- **`clone_repo(github_url)`** → `str` — clones to a temp dir via GitPython, returns local path. Caller owns cleanup.
- **`get_repo_name(url_or_path)`** → `str` — last path segment from URL, or directory name from local path.
- **`walk_files(repo_path)`** → `tuple[list[tuple[str, str]], RepoStats]` — walks tree with rich progress, filters by extension, skips dirs.

Extensions: `.py .js .ts .java .cpp .go .tsx`  
Skipped dirs: `.git node_modules __pycache__ .venv dist build vendor`

## chunker.py

- **`chunk_file(filepath, language, repo_name, rel_path)`** → `list[CodeChunk]` — LangChain `RecursiveCharacterTextSplitter` with language-aware splitting (chunk_size=1000, chunk_overlap=200). Maps language string → `Language` enum. Extracts symbols (function/class names) via regex per language. utf-8 fallback on encoding errors.
- **`chunk_repo(repo_path, file_list)`** → `list[CodeChunk]` — iterates file list from `walk_files`, calls `chunk_file` per file, rich progress bar. Does not produce `RepoStats`.

Symbol extraction patterns:
- Python: `def \w+`, `class \w+`
- JS/TS/TSX: `function \w+`, `class \w+`, `const \w+ =`, `export default \w+`
- Java: `class \w+`, `interface \w+`, `(public|private|protected) \w+ \w+`
- Go: `func \w+`, `type \w+`
- C++: `\w+ \w+\(`, `\w+::\w+`

## Key Decisions

- `accept_input` always returns a tuple — caller never branches on input type.
- `RepoStats` is produced in the ingestion layer only; `chunk_repo` returns only `list[CodeChunk]`.
- Language-aware splitting uses LangChain's `Language` enum where available; falls back to generic recursive splitting for unmapped languages.
