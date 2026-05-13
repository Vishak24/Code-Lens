import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from codelens.models import CodeChunk, GraphEdge, GraphNode

# Regex patterns per language. Each value is a dict mapping node_type to patterns.
# Capture group 1 must always be the symbol name.
_PATTERNS: dict[str, dict[str, list[str]]] = {
    "python": {
        "function": [r"^def (\w+)\s*\("],
        "class":    [r"^class (\w+)[\(:]"],
        "import":   [r"^import ([\w.]+)", r"^from ([\w.]+) import"],
    },
    "javascript": {
        "function": [
            r"(?:^|[ \t])function (\w+)\s*\(",
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
        ],
        "class":  [r"(?:^|[ \t])class (\w+)"],
        "import": [r"from\s+['\"]([^'\"]+)['\"]", r"require\(['\"]([^'\"]+)['\"]\)"],
    },
    "typescript": {
        "function": [
            r"(?:^|[ \t])function (\w+)\s*\(",
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
        ],
        "class":  [r"(?:^|[ \t])class (\w+)"],
        "import": [r"from\s+['\"]([^'\"]+)['\"]"],
    },
    "tsx": {
        "function": [
            r"(?:^|[ \t])function (\w+)\s*\(",
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
        ],
        "class":  [r"(?:^|[ \t])class (\w+)"],
        "import": [r"from\s+['\"]([^'\"]+)['\"]"],
    },
    "java": {
        "function": [
            r"(?:public|private|protected|static|final|abstract|synchronized|native|\s)+\s+\w+\s+(\w+)\s*\("
        ],
        "class":  [r"(?:class|interface|enum)\s+(\w+)"],
        "import": [r"^import\s+(?:static\s+)?([\w.]+);"],
    },
    "go": {
        "function": [r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\("],
        "class":    [r"^type\s+(\w+)\s+(?:struct|interface)"],
        "import":   [r'"([^"]+)"'],
    },
    "cpp": {
        "function": [r"(?:[\w:*&]+\s+)+(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{"],
        "class":    [r"(?:class|struct)\s+(\w+)"],
        "import":   [r'#include\s+[<"]([^>"]+)[>"]'],
    },
}

# Names that look like declarations but aren't user symbols
_NOISE: frozenset[str] = frozenset({
    "if", "for", "while", "switch", "catch", "return", "else", "do",
    "try", "new", "delete", "sizeof", "typeof", "void", "int", "bool",
    "string", "main", "self", "cls", "true", "false", "null", "None",
    "override", "public", "private", "static", "async", "await",
})


def _file_id(source: str) -> str:
    return f"file:{source}"


def _symbol_id(node_type: str, source: str, name: str) -> str:
    return f"{node_type}:{source}:{name}"


def _parse_chunk(chunk: CodeChunk) -> tuple[list[GraphNode], list[GraphEdge]]:
    lang = chunk.language
    patterns = _PATTERNS.get(lang, {})
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    text = chunk.text

    file_node_id = _file_id(chunk.source)

    for node_type, type_patterns in patterns.items():
        for pattern in type_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                name = match.group(1)
                if not name or name in _NOISE or len(name) < 2:
                    continue
                node_id = _symbol_id(node_type, chunk.source, name)
                line_no = chunk.start_line + text[: match.start()].count("\n")
                nodes.append(
                    GraphNode(
                        node_id=node_id,
                        node_type=node_type,
                        name=name,
                        source=chunk.source,
                        language=chunk.language,
                        line_number=line_no,
                    )
                )
                if node_type == "import":
                    edges.append(GraphEdge(
                        source_id=file_node_id,
                        target_id=node_id,
                        edge_type="IMPORTS",
                    ))
                else:
                    edges.append(GraphEdge(
                        source_id=file_node_id,
                        target_id=node_id,
                        edge_type="CONTAINS",
                    ))

    return nodes, edges


def build_graph(chunks: list[CodeChunk]) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()

    all_nodes: dict[str, GraphNode] = {}
    all_edges: list[GraphEdge] = []

    # Ensure a file node exists for each chunk's source
    for chunk in chunks:
        fid = _file_id(chunk.source)
        if fid not in all_nodes:
            all_nodes[fid] = GraphNode(
                node_id=fid,
                node_type="file",
                name=chunk.source,
                source=chunk.source,
                language=chunk.language,
                line_number=None,
            )

    # First pass: extract declared symbols from every chunk
    # Maps function/class name → node_id for cross-file CALLS detection
    known_functions: dict[str, str] = {}

    for chunk in chunks:
        nodes, edges = _parse_chunk(chunk)
        for node in nodes:
            if node.node_id not in all_nodes:
                all_nodes[node.node_id] = node
                if node.node_type == "function":
                    # Last writer wins when names collide across files
                    known_functions[node.name] = node.node_id
        all_edges.extend(edges)

    # Second pass: detect cross-file CALLS edges
    # For each chunk, find functions declared in that chunk's file,
    # then check if any known function from another file is called in the text.
    for chunk in chunks:
        local_func_ids = [
            nid for nid, node in all_nodes.items()
            if node.source == chunk.source and node.node_type == "function"
        ]
        if not local_func_ids:
            continue

        for callee_name, callee_id in known_functions.items():
            callee_source = all_nodes[callee_id].source
            if callee_source == chunk.source:
                continue
            if re.search(rf"\b{re.escape(callee_name)}\s*\(", chunk.text):
                for caller_id in local_func_ids:
                    all_edges.append(GraphEdge(
                        source_id=caller_id,
                        target_id=callee_id,
                        edge_type="CALLS",
                    ))

    # Populate graph, deduplicating edges
    for node in all_nodes.values():
        G.add_node(node.node_id, **node.model_dump())

    seen: set[tuple[str, str, str]] = set()
    for edge in all_edges:
        key = (edge.source_id, edge.target_id, edge.edge_type)
        if key in seen:
            continue
        seen.add(key)
        if edge.source_id in G and edge.target_id in G:
            G.add_edge(edge.source_id, edge.target_id, edge_type=edge.edge_type)

    return G


def save_graph(graph: nx.DiGraph, path: str) -> None:
    data = nx.node_link_data(graph, edges="links")
    Path(path).write_text(json.dumps(data, indent=2))


def load_graph(path: str) -> nx.DiGraph:
    data = json.loads(Path(path).read_text())
    return nx.node_link_graph(data, directed=True, multigraph=False, edges="links")


def get_file_summary(graph: nx.DiGraph, filepath: str) -> dict[str, Any]:
    fid = _file_id(filepath)
    summary: dict[str, list[str]] = {"functions": [], "classes": [], "imports": []}
    for _, target, _ in graph.out_edges(fid, data=True):
        node = graph.nodes.get(target, {})
        node_type = node.get("node_type", "")
        name = node.get("name", "")
        if node_type == "function":
            summary["functions"].append(name)
        elif node_type == "class":
            summary["classes"].append(name)
        elif node_type == "import":
            summary["imports"].append(name)
    return summary
