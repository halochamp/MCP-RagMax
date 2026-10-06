#!/usr/bin/env python3
"""Pipe C: a small stdio MCP adapter for MCP-RagMax.

The dependency direction is intentionally one-way::

    MCP client -> Pipe C -> Pipe B (rag_retrieve) -> retriever/index
                         -> shared KB operations -> registry/files/health
                         -> detached build worker -> ingestor/index stores

MCP-RagMax is deterministic end-to-end. The MCP server never imports the
embedding/build stack until a detached worker needs it, and no retrieval,
build, index, health, or UI path calls an LLM. Pipe B remains the single
implementation of semantic retrieval, while ``kb_operations`` provides
read-only KB helpers. Build control is handed to a detached worker through
``build_jobs`` so status/cancellation persist even when an MCP client opens a
new stdio process for every tool call.
"""

from __future__ import annotations

import datetime as _datetime
import json
import threading
from pathlib import Path
from typing import Literal, TypeAlias

from mcp.server.fastmcp import FastMCP

from rag_retrieve import rag_retrieve as _pipe_b_rag_retrieve
import build_jobs
import kb_operations
from pipeline_config import ORIENTATION_POLICY_ID


Query: TypeAlias = str | list[str]
RetrieveMode = Literal["chunks", "files", "source_first"]
FilesAction = Literal["list", "search", "read"]
ManageAction = Literal["build_start", "build_cancel", "index_prepare", "index_commit"]
StatusView = Literal["health", "build"]

_ALLOWED_MODES = frozenset({"chunks", "files", "source_first"})
_MAX_QUERY_VARIANTS = 8
_MAX_QUERY_CHARS = 4_000
_MAX_FILTER_CHARS = 256
_MAX_OUTPUT_CHARS = 50_000
_MAX_FILENAME_CHARS = 512
_MAX_LIST_LIMIT = 200

# Chroma/BM25 clients are shared by Pipe B.  Keep v1 conservative until a
# dedicated concurrent-read test proves that every backend used by retriever.py
# is safe to call concurrently.
_RETRIEVE_LOCK = threading.Lock()


_QUERY_GUIDANCE = (
    "For rag_retrieve, use one query string or multiple variants of the SAME question only. "
    "Variants must preserve the same intent and should differ only in representation, for example: "
    "the original Thai sentence, the same question in English, Thai keywords for that question, "
    "and English keywords for that question. Do not use variants to introduce new subquestions, "
    "narrower angles, extra assumptions, or merely related topics. Each variant is searched "
    "independently with Dense + BM25 and the ranked lists are fused with RRF, so cross-language "
    "and lexical variants improve recall without changing the user's intent."
)


mcp = FastMCP(
    "MCP-RagMax Pipe C",
    instructions=(
        "MCP-RagMax is a local deterministic RAG backend. Retrieval/inspection tools are "
        "read-only. Bounded build tools mutate only derived index/state, never source "
        "documents. Search, build, index generation, health and UI use no LLM. " + _QUERY_GUIDANCE
    ),
)


def _bounded_text(name: str, value: object, *, limit: int, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not allow_empty and not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > limit:
        raise ValueError(f"{name} must be <= {limit} characters")
    return text


def _validate_query(query: object) -> Query:
    if isinstance(query, str):
        return _bounded_text("query", query, limit=_MAX_QUERY_CHARS, allow_empty=False)
    if not isinstance(query, list):
        raise ValueError("query must be a string or an array of strings")
    if not query:
        raise ValueError("query must contain at least one string")
    if len(query) > _MAX_QUERY_VARIANTS:
        raise ValueError(f"query may contain at most {_MAX_QUERY_VARIANTS} variants")
    normalized: list[str] = []
    for index, item in enumerate(query):
        normalized.append(
            _bounded_text(f"query[{index}]", item, limit=_MAX_QUERY_CHARS, allow_empty=False)
        )
    return normalized


def _validate_date(name: str, value: object) -> str:
    text = _bounded_text(name, value, limit=10)
    if not text:
        return ""
    try:
        parsed = _datetime.date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return text


def _validate_request(
    query: object,
    mode: object,
    tags: object,
    filename_contains: object,
    created_after: object,
    created_before: object,
    source_type: object,
) -> dict[str, object]:
    validated_query = _validate_query(query)
    normalized_mode = _bounded_text("mode", mode, limit=32).lower()
    if normalized_mode not in _ALLOWED_MODES:
        allowed = ", ".join(sorted(_ALLOWED_MODES))
        raise ValueError(f"mode must be one of: {allowed}")

    after = _validate_date("created_after", created_after)
    before = _validate_date("created_before", created_before)
    if after and before and after > before:
        raise ValueError("created_after must be <= created_before")

    return {
        "query": validated_query,
        "mode": normalized_mode,
        "tags": _bounded_text("tags", tags, limit=_MAX_FILTER_CHARS),
        "filename_contains": _bounded_text(
            "filename_contains", filename_contains, limit=_MAX_FILTER_CHARS
        ),
        "created_after": after,
        "created_before": before,
        "source_type": _bounded_text("source_type", source_type, limit=_MAX_FILTER_CHARS),
    }


def _cap_output(result: str) -> str:
    cap = _MAX_OUTPUT_CHARS
    if cap <= 0:
        return ""
    if len(result) <= cap:
        return result
    marker = (
        f"\n\n[truncated] Pipe C output is capped at {cap} characters; "
        "use narrower filters or a more specific query."
    )
    if len(marker) >= cap:
        return marker[:cap]
    return result[: cap - len(marker)] + marker


def _invoke_pipe_b(arguments: dict[str, object]) -> object:
    """Call Pipe B's LangChain wrapper in one patchable boundary."""
    return _pipe_b_rag_retrieve.invoke(arguments)


def _call_pipe_b(arguments: dict[str, object]) -> str:
    """Delegate one validated request to Pipe B while preserving its contract."""
    with _RETRIEVE_LOCK:
        try:
            result = _invoke_pipe_b(arguments)
        except Exception as exc:
            # Do not retain the backend exception as a chained cause: MCP hosts
            # may render exception chains/tracebacks, which could expose paths
            # or other backend details even when the public message is safe.
            raise RuntimeError(f"Pipe B rag_retrieve failed ({type(exc).__name__})") from None

    if not isinstance(result, str):
        raise RuntimeError("Pipe B rag_retrieve returned a non-text result")
    if result.startswith("[error]"):
        # Pipe B's user-facing error can contain backend paths or exception
        # details.  Preserve MCP failure semantics without forwarding those
        # details across the tool boundary.
        raise RuntimeError("Pipe B rag_retrieve returned an error")
    return _cap_output(result)


@mcp.tool()
def rag_retrieve(
    query: str | list[str],
    mode: RetrieveMode = "chunks",
    tags: str = "",
    filename_contains: str = "",
    created_after: str = "",
    created_before: str = "",
    source_type: str = "",
) -> str:
    """Retrieve local knowledge through Pipe B.

    Pipe C validates the request, then delegates unchanged retrieval semantics
    to Pipe B's ``rag_retrieve`` implementation. ``query`` is one question string
    or up to eight bounded variants of the SAME question.

    Query-variant rule for agents: preserve one intent. Good variants are the
    original Thai sentence, the same question in English, Thai keywords for that
    question, and English keywords for that question. Do not add a new subquestion,
    narrower angle, extra assumption, or merely related topic as another variant.
    Variants are searched independently with Dense + BM25 and then RRF-fused.

    ``mode`` is ``chunks``, ``files``, or ``source_first``. Date filters use
    ``YYYY-MM-DD``. This is read-only and LLM-free. It does not read arbitrary
    paths or expose shell/Python/memory-write tools.
    """
    arguments = _validate_request(
        query,
        mode,
        tags,
        filename_contains,
        created_after,
        created_before,
        source_type,
    )
    return _call_pipe_b(arguments)


def _validated_limit(limit: object) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    if limit < 1 or limit > _MAX_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIST_LIMIT}")
    return limit


def _validated_offset(offset: object) -> int:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    return offset


def rag_list(limit: int = 100, offset: int = 0) -> str:
    """List files registered in the knowledge base without using an LLM."""
    page_size = _validated_limit(limit)
    start = _validated_offset(offset)
    paths = kb_operations.registered_paths()
    total = len(paths)
    page = paths[start:start + page_size]
    if not page:
        return f"registered_files={total}\noffset={start}\n(no files in this page)"
    lines = [f"- {Path(path).name}" for path in page]
    return _cap_output(
        f"registered_files={total}\noffset={start}\nreturned={len(page)}\n" + "\n".join(lines)
    )


def rag_search_files(query: str, limit: int = 30) -> str:
    """Search registered knowledge-base filenames, case-insensitively, without an LLM."""
    needle = _bounded_text("query", query, limit=_MAX_FILENAME_CHARS, allow_empty=False)
    page_size = _validated_limit(limit)
    matches = kb_operations.search_registered_paths(needle)
    if not matches:
        return f"matches=0\n(no registered filenames contain {needle!r})"
    shown = matches[:page_size]
    lines = [f"- {Path(path).name}" for path in shown]
    return _cap_output(
        f"matches={len(matches)}\nreturned={len(shown)}\n" + "\n".join(lines)
    )


def rag_read_file(filename: str) -> str:
    """Read one registered KB document only; arbitrary filesystem paths are not allowed."""
    requested = _bounded_text(
        "filename", filename, limit=_MAX_FILENAME_CHARS, allow_empty=False
    )
    try:
        path, text = kb_operations.read_registered_file(requested)
    except kb_operations.RegisteredFileNotFound:
        raise ValueError("filename is not registered in the knowledge base") from None
    except kb_operations.RegisteredFileAmbiguous as exc:
        names = ", ".join(Path(path).name for path in exc.matches[:10])
        raise ValueError(f"filename is ambiguous; matching registered files: {names}") from None
    return _cap_output(f"file={path.name}\n{text}")


def build_kb() -> str:
    """Start a background RAG_MAX knowledge-base build and return a persistent job id.

    The source root is fixed by RAG_MAX's ingestion configuration; callers cannot
    supply arbitrary paths. Public MCP callers poll with ``rag_status(view="build")``
    and cancel with ``rag_manage(action="build_cancel")``.
    """
    return _cap_output(build_jobs.start_build())


def build_status(job_id: str) -> str:
    """Return live progress for one background build job."""
    clean = _bounded_text("job_id", job_id, limit=64, allow_empty=False)
    return _cap_output(build_jobs.build_status(clean))


def cancel_build(job_id: str) -> str:
    """Request safe cooperative cancellation of one background build job.

    New-file ingestion can roll back mid-file. A changed file already inside
    its destructive rebuild section is completed first, then cancellation is
    honored before the next file so the source is not left missing from the
    derived index merely because the user cancelled.
    """
    clean = _bounded_text("job_id", job_id, limit=64, allow_empty=False)
    return _cap_output(build_jobs.cancel_build(clean))


def rag_rebuild_index(
    mode: str = "auto",
    topics: list[str] | None = None,
    expected_fingerprint: str = "",
) -> str:
    """Prepare or commit canonical ``rag_index.json`` using the caller's LLM.

    ``mode='auto'`` (default) makes the first call a safe prepare. If both
    ``topics`` and ``expected_fingerprint`` are supplied it commits instead.
    Supplying topics early without a fingerprint is treated as prepare and does
    not write anything, making the contract robust for generic MCP callers that
    have not seen this remote schema yet.

    ``mode='prepare'`` returns a bounded deterministic snapshot of filenames,
    tags, source types, and an ``expected_fingerprint``. The calling agent's own
    LLM should derive 1-30 concise high-level topic labels from that snapshot.

    ``mode='commit'`` accepts those topic labels plus the exact fingerprint from
    prepare. MCP-RagMax validates the topics, recomputes deterministic metadata,
    rechecks the fingerprint, and atomically writes ``rag_index.json``. If the KB
    changed between phases, commit fails and the caller must prepare again.

    MCP-RagMax never calls an LLM in either phase. Prepare/commit are short
    synchronous operations exposed through ``rag_manage`` action choices.
    """
    selected_mode = _bounded_text("mode", mode, limit=16, allow_empty=False).lower()
    if selected_mode not in {"auto", "prepare", "commit"}:
        raise ValueError("mode must be auto, prepare or commit")
    if selected_mode == "auto":
        has_topics = bool(topics)
        has_fingerprint = bool(str(expected_fingerprint or "").strip())
        if has_topics and has_fingerprint:
            selected_mode = "commit"
        elif has_fingerprint and not has_topics:
            raise ValueError("topics are required when expected_fingerprint is supplied")
        else:
            # No fingerprint means there is no write authority. Ignore any
            # premature caller topics and return the authoritative snapshot.
            selected_mode = "prepare"
            topics = None

    active = build_jobs.active_job_snapshot()
    if active is not None:
        return _cap_output(
            "\n".join(
                [
                    "status=busy",
                    f"build_job_id={active.get('job_id') or ''}",
                    f"build_job_status={active.get('status') or 'running'}",
                    "next=wait for rag_status(view=build, job_id=...) to become terminal, then call rag_manage(action=index_prepare) again",
                ]
            )
        )

    import rag_index_builder

    if selected_mode == "prepare":
        if topics:
            raise ValueError("topics are only accepted in commit mode")
        if str(expected_fingerprint or "").strip():
            raise ValueError("expected_fingerprint is only accepted in commit mode")
        prepared = rag_index_builder.prepare_index_context()
        return _cap_output(
            "\n".join(
                [
                    "status=prepared",
                    "mode=prepare",
                    f"expected_fingerprint={prepared['expected_fingerprint']}",
                    f"total_files={prepared['total_files']}",
                    f"sampled_files={prepared['sampled_files']}",
                    "source_types_json=" + json.dumps(prepared["source_types"], ensure_ascii=False, sort_keys=True),
                    "top_tags_json=" + json.dumps(prepared["top_tags"], ensure_ascii=False, sort_keys=True),
                    "filenames_json=" + json.dumps(prepared["filenames"], ensure_ascii=False),
                    "topic_guidance=" + str(prepared["topic_rules"]["guidance"]),
                    "caller_must_continue=true",
                    "caller_must_not_answer_before_commit=true",
                    "next_tool=rag_manage",
                    "next_arguments_json=" + json.dumps(
                        {
                            "action": "index_commit",
                            "topics": ["<1-30 concise topics derived only from this snapshot>"],
                            "expected_fingerprint": prepared["expected_fingerprint"],
                        },
                        ensure_ascii=False,
                    ),
                    "next=Derive topics with your own LLM, replace the placeholder topics in next_arguments_json, call rag_manage with those arguments, and only answer after status=done",
                    f"orientation_policy_id={ORIENTATION_POLICY_ID}",
                    "llm_used_by_backend=false",
                ]
            )
        )

    if topics is None:
        raise ValueError("topics are required in commit mode")
    if not isinstance(topics, list):
        raise ValueError("topics must be an array of strings")
    if len(topics) > 30:
        raise ValueError("topics may contain at most 30 items")
    fingerprint = _bounded_text(
        "expected_fingerprint", expected_fingerprint, limit=64, allow_empty=False
    ).lower()
    try:
        result = rag_index_builder.commit_index(topics, fingerprint)
    except rag_index_builder.IndexCommitConflict:
        raise RuntimeError("rag_index commit conflict; knowledge changed, run prepare again") from None
    return _cap_output(
        "\n".join(
            [
                "status=done",
                "mode=commit",
                f"files={int(result.get('total') or 0)}",
                f"sampled={int(result.get('sampled') or 0)}",
                f"tags={len(result.get('tags') or {})}",
                f"source_types={len(result.get('source_types') or {})}",
                f"topics={result.get('topics_line') or ''}",
                f"registry_fingerprint={result.get('registry_fingerprint') or ''}",
                "topic_method=caller_llm",
                f"orientation_policy_id={ORIENTATION_POLICY_ID}",
                "llm_used_by_backend=false",
                f"output={rag_index_builder.RAG_INDEX_PATH}",
            ]
        )
    )


def rag_health() -> str:
    """Report deterministic store, registry, rag_index freshness, and build-job health."""
    with _RETRIEVE_LOCK:
        snapshot = kb_operations.health_snapshot()
    issues = list(snapshot.get("issues") or [])
    ghosts = list(snapshot.get("ghost_files") or [])
    index_status = str(snapshot.get("index_status") or "missing")
    pipeline_outdated = int(snapshot.get("pipeline_outdated_files") or 0)
    active = build_jobs.active_job_snapshot()
    healthy = not issues and not ghosts and pipeline_outdated == 0 and index_status == "ready"
    lines = [
        f"status={'healthy' if healthy else 'degraded'}",
        f"issues={len(issues)}",
        f"ghost_files={len(ghosts)}",
        f"registered_files={int(snapshot.get('registered_files') or 0)}",
        f"pipeline_outdated_files={pipeline_outdated}",
        f"index_status={index_status}",
        f"index_files={int(snapshot.get('index_files') or 0)}",
        f"build_job_status={str(active.get('status') or 'running') if active else 'idle'}",
    ]
    if active:
        lines.append(f"build_job_id={active.get('job_id') or ''}")
        lines.append(f"build_job_kind={active.get('job_kind') or 'build_kb'}")
    lines.extend(f"issue: {item}" for item in issues)
    lines.extend(f"ghost: {Path(item).name}" for item in ghosts)
    return _cap_output("\n".join(lines))


@mcp.tool()
def rag_files(
    action: FilesAction,
    query: str = "",
    filename: str = "",
    limit: int = 100,
    offset: int = 0,
) -> str:
    """Browse registered KB files through one action-based tool.

    action choices:
    - list: paged registered filenames using limit/offset
    - search: filename search using query/limit
    - read: read one registered KB file using filename
    """
    action = str(action).strip().casefold()
    if action == "list":
        return rag_list(limit=limit, offset=offset)
    if action == "search":
        return rag_search_files(query, limit=limit)
    if action == "read":
        return rag_read_file(filename)
    raise ValueError("unsupported action")


@mcp.tool()
def rag_manage(
    action: ManageAction,
    job_id: str = "",
    topics: list[str] | None = None,
    expected_fingerprint: str = "",
) -> str:
    """Manage deterministic build and caller-assisted index lifecycle.

    action choices:
    - build_start: start the persistent KB build
    - build_cancel: cooperatively cancel one build job
    - index_prepare: return guarded rag_index context for the caller LLM
    - index_commit: validate caller topics and atomically commit rag_index.json
    """
    action = str(action).strip().casefold()
    if action == "build_start":
        return build_kb()
    if action == "build_cancel":
        return cancel_build(job_id)
    if action == "index_prepare":
        return rag_rebuild_index(mode="prepare")
    if action == "index_commit":
        return rag_rebuild_index(
            mode="commit", topics=topics, expected_fingerprint=expected_fingerprint
        )
    raise ValueError("unsupported action")


@mcp.tool()
def rag_status(view: StatusView = "health", job_id: str = "") -> str:
    """Read RAG health or one persistent build-job status through one tool.

    view choices:
    - health: store/registry/index/pipeline/job health
    - build: one build job by job_id
    """
    view = str(view or "health").strip().casefold()
    if view == "health":
        return rag_health()
    if view == "build":
        return build_status(job_id)
    raise ValueError("unsupported view")


if __name__ == "__main__":
    mcp.run(transport="stdio")
