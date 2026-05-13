import hashlib
from typing import Optional
from pydantic import BaseModel


def make_chunk_id(repo_name: str, source: str, chunk_index: int) -> str:
    raw = f"{repo_name}:{source}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CodeChunk(BaseModel):
    id: str
    text: str
    source: str
    language: str
    repo_name: str
    chunk_index: int
    total_chunks: int
    start_line: int
    symbols: list[str]


class EmbeddedChunk(CodeChunk):
    embedding: list[float]


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    name: str
    source: str
    language: str
    line_number: Optional[int] = None


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: str


class RepoStats(BaseModel):
    repo_name: str
    total_files: int
    total_chunks: int
    languages: dict[str, int]
    total_lines: int
