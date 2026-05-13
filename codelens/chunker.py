import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from codelens.models import CodeChunk, make_chunk_id

LANGUAGE_ENUM_MAP: dict[str, Language] = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "cpp": Language.CPP,
    "go": Language.GO,
    "tsx": Language.TS,
}

SYMBOL_PATTERNS: dict[str, list[str]] = {
    "python": [r"def (\w+)", r"class (\w+)"],
    "javascript": [r"function (\w+)", r"class (\w+)", r"const (\w+)\s*=", r"export default (\w+)"],
    "typescript": [r"function (\w+)", r"class (\w+)", r"const (\w+)\s*=", r"export default (\w+)"],
    "tsx": [r"function (\w+)", r"class (\w+)", r"const (\w+)\s*=", r"export default (\w+)"],
    "java": [r"class (\w+)", r"interface (\w+)", r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\("],
    "go": [r"func (\w+)", r"type (\w+)"],
    "cpp": [r"(\w+)::\w+", r"\w+\s+(\w+)\s*\("],
}


def _extract_symbols(text: str, language: str) -> list[str]:
    patterns = SYMBOL_PATTERNS.get(language, [])
    symbols: list[str] = []
    for pattern in patterns:
        symbols.extend(re.findall(pattern, text))
    return list(dict.fromkeys(symbols))


def _make_splitter(language: str) -> RecursiveCharacterTextSplitter:
    lang_enum = LANGUAGE_ENUM_MAP.get(language)
    if lang_enum is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=1000,
            chunk_overlap=200,
        )
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def chunk_file(
    filepath: str,
    language: str,
    repo_name: str,
    rel_path: str,
) -> list[CodeChunk]:
    path = Path(filepath)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    splitter = _make_splitter(language)
    raw_chunks = splitter.split_text(text)
    if not raw_chunks:
        return []

    total = len(raw_chunks)
    lines_so_far = 0
    chunks: list[CodeChunk] = []

    for i, chunk_text in enumerate(raw_chunks):
        chunk_id = make_chunk_id(repo_name, rel_path, i)
        symbols = _extract_symbols(chunk_text, language)
        chunks.append(
            CodeChunk(
                id=chunk_id,
                text=chunk_text,
                source=rel_path,
                language=language,
                repo_name=repo_name,
                chunk_index=i,
                total_chunks=total,
                start_line=lines_so_far + 1,
                symbols=symbols,
            )
        )
        lines_so_far += chunk_text.count("\n")

    return chunks


def chunk_repo(
    repo_path: str,
    file_list: list[tuple[str, str]],
) -> list[CodeChunk]:
    root = Path(repo_path)
    all_chunks: list[CodeChunk] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Chunking files..."),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("chunk", total=len(file_list))
        for filepath, language in file_list:
            rel_path = str(Path(filepath).relative_to(root))
            repo_name = root.name
            chunks = chunk_file(filepath, language, repo_name, rel_path)
            all_chunks.extend(chunks)
            progress.advance(task)

    return all_chunks
