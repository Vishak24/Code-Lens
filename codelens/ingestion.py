import os
import tempfile
from pathlib import Path

import git
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from codelens.models import RepoStats

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".go", ".tsx"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build", "vendor"}

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".go": "go",
    ".tsx": "tsx",
}


def get_repo_name(url_or_path: str) -> str:
    if "github.com" in url_or_path:
        return url_or_path.rstrip("/").split("/")[-1].removesuffix(".git")
    return Path(url_or_path).resolve().name


def clone_repo(github_url: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="codelens_")
    git.Repo.clone_from(github_url, tmp_dir)
    return tmp_dir


def accept_input(url_or_path: str) -> tuple[str, str]:
    repo_name = get_repo_name(url_or_path)
    if "github.com" in url_or_path:
        local_path = clone_repo(url_or_path)
    else:
        local_path = str(Path(url_or_path).resolve())
    return local_path, repo_name


def walk_files(repo_path: str) -> tuple[list[tuple[str, str]], RepoStats]:
    root = Path(repo_path)
    files: list[tuple[str, str]] = []
    languages: dict[str, int] = {}
    total_lines = 0

    all_paths = [
        p for p in root.rglob("*")
        if p.is_file()
        and not any(skip in p.parts for skip in SKIP_DIRS)
        and p.suffix in SUPPORTED_EXTENSIONS
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Walking files..."),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("walk", total=len(all_paths))
        for p in all_paths:
            lang = EXTENSION_TO_LANGUAGE[p.suffix]
            files.append((str(p), lang))
            languages[lang] = languages.get(lang, 0) + 1
            try:
                total_lines += sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
            except OSError:
                pass
            progress.advance(task)

    repo_name = get_repo_name(repo_path)
    stats = RepoStats(
        repo_name=repo_name,
        total_files=len(files),
        total_chunks=0,
        languages=languages,
        total_lines=total_lines,
    )
    return files, stats
