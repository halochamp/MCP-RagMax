#!/usr/bin/env python3
"""Caller-assisted canonical ``rag_index.json`` lifecycle for MCP-RagMax.

MCP-RagMax never calls an LLM.  The backend owns deterministic truth
(registry fingerprint, file count, frontmatter tags and source types) while the
LLM in the *calling agent* owns only the semantic topic summary.

The lifecycle is intentionally two phase:

1. ``prepare_index_context()`` returns a bounded snapshot for the caller LLM.
2. ``commit_index(topics, expected_fingerprint)`` validates the caller topics,
   rechecks the registry fingerprint and atomically installs ``rag_index.json``.

If the knowledge registry changes between prepare and commit, commit fails and
the previous good index is preserved.  This keeps the RAG backend LLM-free
without allowing an external model to write filesystem/index truth directly.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import json
import os
from pathlib import Path
import re
from typing import Iterable

import file_registry
from config import RAG_INDEX_PATH, source_path
from pipeline_config import ORIENTATION_POLICY_ID
_MAX_TOPICS = 30
_MAX_TOPIC_CHARS = 96
_MAX_CONTEXT_FILES = 100
_MAX_CONTEXT_TAGS = 50
_TAGS_BLOCK_RE = re.compile(r"tags:[ \t]*\n((?:[ \t]*-[ \t]*.*(?:\n|$))+)")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class IndexCommitConflict(RuntimeError):
    """The KB registry changed after the caller prepared its topic summary."""


def _to_abs(raw: str) -> Path:
    return source_path(raw)


def _source_type(raw: str) -> str:
    return Path(raw).suffix.lower().lstrip(".") or "unknown"


def _frontmatter_tags(raw: str) -> list[str]:
    path = _to_abs(raw)
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
    except OSError:
        return []
    fm = re.match(r"^---\n(.*?)\n---", head, re.S)
    if not fm:
        return []
    block = _TAGS_BLOCK_RE.search(fm.group(1))
    if not block:
        return []
    out: list[str] = []
    for line in block.group(1).splitlines():
        tag = line.strip("- ").strip().casefold()
        if tag:
            out.append(tag)
    return out


def _sorted_counts(counter: dict[str, int] | Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _metadata(paths: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    tags: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    for raw in paths:
        source_types[_source_type(raw)] += 1
        tags.update(_frontmatter_tags(raw))
    return _sorted_counts(tags), _sorted_counts(source_types)


def _sample_names(paths: list[str], limit: int = _MAX_CONTEXT_FILES) -> list[str]:
    """Return a deterministic, corpus-spanning filename sample."""
    ordered = sorted(paths, key=lambda raw: (Path(raw).name.casefold(), raw.casefold()))
    if len(ordered) <= limit:
        return [Path(raw).name for raw in ordered]
    if limit <= 1:
        return [Path(ordered[0]).name]

    # Evenly span the sorted corpus instead of taking only the alphabetical
    # head.  This is deterministic and bounded, and requires no random state.
    indexes = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)]
    seen: set[int] = set()
    names: list[str] = []
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        names.append(Path(ordered[index]).name)
    return names


def _current_snapshot() -> tuple[str, list[str], dict[str, int], dict[str, int]]:
    if hasattr(file_registry, "reload"):
        file_registry.reload()
    fingerprint = file_registry.registry_fingerprint()
    paths = list(file_registry.all_registered())
    if not paths:
        raise ValueError("knowledge base is empty")
    if not _FINGERPRINT_RE.fullmatch(fingerprint):
        raise RuntimeError("knowledge registry fingerprint unavailable")
    tags, source_types = _metadata(paths)
    return fingerprint, paths, tags, source_types


def prepare_index_context() -> dict[str, object]:
    """Return bounded deterministic context for the *calling* LLM.

    The returned ``expected_fingerprint`` must be echoed unchanged to
    ``commit_index``.  The caller should infer only high-level topic labels; it
    must not invent file counts, tags, source types or fingerprints.
    """
    fingerprint, paths, tags, source_types = _current_snapshot()
    filenames = _sample_names(paths)
    top_tags = dict(list(tags.items())[:_MAX_CONTEXT_TAGS])
    return {
        "status": "prepared",
        "expected_fingerprint": fingerprint,
        "total_files": len(paths),
        "sampled_files": len(filenames),
        "filenames": filenames,
        "top_tags": top_tags,
        "source_types": source_types,
        "topic_rules": {
            "min_topics": 1,
            "max_topics": _MAX_TOPICS,
            "max_chars_each": _MAX_TOPIC_CHARS,
            "guidance": (
                "Using only this KB snapshot, produce concise high-level topic labels. "
                "Prefer Thai when natural, preserve important English technical terms, "
                "avoid duplicates, filenames, explanations, and unsupported topics."
            ),
        },
        "llm_used_by_backend": False,
    }


def _normalize_topics(topics: Iterable[str] | str) -> list[str]:
    if isinstance(topics, str):
        raw = topics.strip()
        if not raw:
            candidates: list[str] = []
        elif "|" in raw:
            candidates = raw.split("|")
        elif "\n" in raw:
            candidates = raw.splitlines()
        else:
            candidates = [raw]
    else:
        try:
            candidates = list(topics)
        except TypeError as exc:
            raise ValueError("topics must be an array of strings") from exc

    out: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str):
            raise ValueError("every topic must be a string")
        topic = " ".join(value.split()).strip(" `\"'.,:;|-/")
        if not topic:
            continue
        if len(topic) > _MAX_TOPIC_CHARS:
            raise ValueError(f"each topic must be <= {_MAX_TOPIC_CHARS} characters")
        if any(ord(ch) < 32 for ch in topic):
            raise ValueError("topics must not contain control characters")
        key = topic.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(topic)

    if not out:
        raise ValueError("at least one topic is required")
    if len(out) > _MAX_TOPICS:
        raise ValueError(f"topics may contain at most {_MAX_TOPICS} unique items")
    return out


def _atomic_write_index(target: Path, payload: dict[str, object], expected_fingerprint: str) -> None:
    if file_registry.registry_fingerprint() != expected_fingerprint:
        raise IndexCommitConflict("knowledge registry changed after prepare; run prepare again")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck immediately before replace so a concurrent KB build cannot
        # make a caller-generated orientation snapshot look current by accident.
        if file_registry.registry_fingerprint() != expected_fingerprint:
            raise IndexCommitConflict("knowledge registry changed during commit; run prepare again")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def commit_index(
    topics: Iterable[str] | str,
    expected_fingerprint: str,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate caller-LLM topics and atomically commit canonical metadata."""
    fingerprint = str(expected_fingerprint or "").strip().lower()
    if not _FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError("expected_fingerprint must be a 64-character SHA-256 hex string")

    current_fingerprint, paths, tags, source_types = _current_snapshot()
    if current_fingerprint != fingerprint:
        raise IndexCommitConflict("knowledge registry changed after prepare; run prepare again")

    normalized_topics = _normalize_topics(topics)
    # Metadata is recomputed at commit, never accepted from the caller.
    payload: dict[str, object] = {
        "updated": str(date.today()),
        "total": len(paths),
        "sampled": len(_sample_names(paths)),
        "topics_line": " | ".join(normalized_topics),
        "topics": normalized_topics,
        "tags": tags,
        "source_types": source_types,
        "registry_fingerprint": fingerprint,
        "topic_method": "caller_llm",
        "orientation_policy_id": ORIENTATION_POLICY_ID,
        "caller_llm_used": True,
        "llm_used_by_backend": False,
    }
    target = Path(output_path) if output_path is not None else RAG_INDEX_PATH
    target = target.expanduser().resolve()
    _atomic_write_index(target, payload, fingerprint)
    return payload


def build_index(*_args, **_kwargs):
    """Compatibility tombstone: automatic backend topic generation was removed."""
    raise RuntimeError(
        "automatic rag_index generation was removed; use prepare_index_context() "
        "then commit_index(topics, expected_fingerprint) with topics from the caller LLM"
    )


if __name__ == "__main__":
    print(json.dumps(prepare_index_context(), ensure_ascii=False, indent=2))
