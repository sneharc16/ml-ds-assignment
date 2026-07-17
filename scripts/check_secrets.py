"""Fail when high-confidence credential patterns are present in the repository."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 2 * 1024 * 1024
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "data",
    "node_modules",
    "venv",
}
SKIP_SUFFIXES = {
    ".duckdb",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".parquet",
    ".pdf",
    ".png",
    ".pptx",
}

PATTERNS = {
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "PGP private key": re.compile(b"-----BEGIN PGP " + b"PRIVATE KEY BLOCK-----"),
    "AWS access key": re.compile(rb"(?:A3T[A-Z0-9]|AKIA|ASIA)[A-Z0-9]{16}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,255}"),
    "GitHub fine-grained token": re.compile(rb"github_pat_[A-Za-z0-9_]{20,255}"),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "Slack token": re.compile(rb"xox[baprs]-[0-9A-Za-z-]{10,255}"),
}


def repository_files(root: Path) -> list[Path]:
    """Return scannable files while excluding generated and binary-heavy paths."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        files.append(path)
    return files


def find_secrets(root: Path = ROOT) -> list[tuple[Path, str]]:
    """Return file paths and pattern labels without returning secret values."""
    findings: list[tuple[Path, str]] = []
    for path in repository_files(root):
        content = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append((path.relative_to(root), label))
    return findings


def main() -> int:
    findings = find_secrets()
    if not findings:
        print("Secret scan passed.")
        return 0
    print("Secret scan failed. Remove or rotate the detected credential material:")
    for path, label in findings:
        print(f"- {path}: {label}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
