#!/usr/bin/env python3
"""Validate a Rummage evaluation JSONL against the index and disk paths.

Usage:
    python eval/validate_queries.py eval/queries.jsonl \
        D:/tools/rummage-portable-0.1.0-full/index/index.sqlite3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ALLOWED_TYPES = {"keyword", "semantic", "ocr", "filter"}


def normalize(path: str) -> str:
    return path.strip().replace("\\", "/").rstrip("/").lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("index_db", type=Path)
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.index_db}?mode=ro", uri=True)
    try:
        indexed = {normalize(row[0]) for row in conn.execute("select path from files")}
    finally:
        conn.close()

    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    errors: list[str] = []
    rows: list[dict] = []
    type_counts: dict[str, int] = {}
    rel_counts: list[int] = []

    with args.queries.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                errors.append(f"line {lineno}: invalid JSON: {exc}")
                continue
            rows.append(obj)

            qid = obj.get("id")
            if not qid:
                errors.append(f"line {lineno}: missing id")
            elif qid in seen_ids:
                errors.append(f"line {lineno}: duplicate id {qid}")
            else:
                seen_ids.add(qid)

            query = obj.get("query", "")
            if not isinstance(query, str) or not query.strip():
                errors.append(f"line {lineno}: empty query")
            elif query.strip().lower() in seen_queries:
                errors.append(f"line {lineno}: duplicate query text")
            else:
                seen_queries.add(query.strip().lower())

            kind = obj.get("type")
            if kind not in ALLOWED_TYPES:
                errors.append(f"line {lineno}: bad type {kind!r}")
            else:
                type_counts[kind] = type_counts.get(kind, 0) + 1

            relevant = obj.get("relevant", [])
            grades = obj.get("grades", {})
            if not isinstance(relevant, list) or not isinstance(grades, dict):
                errors.append(f"line {lineno}: relevant must be list, grades must be object")
                continue
            paths = set(relevant) | set(grades.keys())
            rel_counts.append(len(paths))
            if not paths:
                errors.append(f"line {lineno}: no relevant paths")
            for path in paths:
                if not isinstance(path, str):
                    errors.append(f"line {lineno}: non-string path {path!r}")
                    continue
                if not Path(path).is_absolute():
                    errors.append(f"line {lineno}: not absolute: {path}")
                if normalize(path) not in indexed:
                    errors.append(f"line {lineno}: not in index: {path}")
                if not Path(path).exists():
                    errors.append(f"line {lineno}: missing on disk: {path}")
            for path, grade in grades.items():
                if grade not in (1, 2, 3):
                    errors.append(f"line {lineno}: grade must be 1/2/3: {path} -> {grade}")

    print(f"queries: {len(rows)}")
    print(f"errors: {len(errors)}")
    print(f"types: {type_counts}")
    if rel_counts:
        print(
            f"relevant per query: min={min(rel_counts)} "
            f"max={max(rel_counts)} avg={sum(rel_counts)/len(rel_counts):.2f}"
        )
    for error in errors:
        print(" -", error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
