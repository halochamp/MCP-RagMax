from __future__ import annotations
from pathlib import Path
import datetime

from config import source_path
import re

from langchain_core.tools import tool

import retriever

TOP_N = 5
RELATED_N = 5   # extra lower-ranked fused hits surfaced as see-also (filenames only, no body)

def _to_abs(source: str) -> str:
    """Resolve and validate a stored source inside the configured knowledge root."""
    return str(source_path(source))


def _source_type(source: str) -> str:
    suffix = Path(source).suffix.lower().lstrip(".")
    return suffix or "unknown"


def _frontmatter(source: str) -> dict:
    path = Path(_to_abs(source))
    out = {"tags": [], "created": None}
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2500]
    except Exception:
        return out
    fm = re.match(r"^---\n(.*?)\n---", head, re.S)
    if not fm:
        return out
    block = fm.group(1)
    m = re.search(r"^created:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", block, re.M)
    if m:
        try:
            out["created"] = datetime.date.fromisoformat(m.group(1))
        except Exception:
            pass
    tags = re.search(r"^tags:\n((?: *- .*\n?)+)", block, re.M)
    if tags:
        out["tags"] = [line.strip("- ").strip() for line in tags.group(1).splitlines() if line.strip()]
    return out


def _heading_path(source: str, parent_text: str) -> str:
    """Best-effort markdown heading path for the parent chunk."""
    abs_path = Path(_to_abs(source))
    try:
        full = abs_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        full = ""
    needle = parent_text[:300].strip()
    scope = parent_text
    if full and needle:
        idx = full.find(needle)
        if idx >= 0:
            scope = full[:idx] + parent_text
    stack: list[tuple[int, str]] = []
    for line in scope.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        stack = [(lvl, txt) for lvl, txt in stack if lvl < level]
        stack.append((level, title))
    return " > ".join(txt for _, txt in stack[-4:])


def _read_file_hint(source: str, heading: str = "") -> str:
    path = _to_abs(source)
    if heading:
        last = heading.split(" > ")[-1].replace('"', '\\"')
        if _source_type(source) in {"pdf", "docx", "xlsx", "xls"}:
            return f'read_file(path="{path}", contains="{last}", doc_mode="section")'
        return f'read_file(path="{path}", contains="{last}", context_lines=12)'
    return f'read_file(path="{path}")'


def _matches_filters(c: dict, tags: str = "", filename_contains: str = "",
                     created_after: str = "", created_before: str = "",
                     source_type: str = "") -> bool:
    src = c.get("source", "")
    fname = Path(src).name.lower()
    if filename_contains and filename_contains.lower() not in fname:
        return False
    if source_type:
        wanted = {x.strip().lower().lstrip(".") for x in re.split(r"[, ]+", source_type) if x.strip()}
        if wanted and _source_type(src) not in wanted:
            return False
    fm = _frontmatter(src)
    if tags:
        have = {str(t).lower() for t in fm.get("tags", [])}
        wanted_tags = {x.strip().lower() for x in re.split(r"[, ]+", tags) if x.strip()}
        if wanted_tags and not wanted_tags.issubset(have):
            return False
    created = fm.get("created")
    if created_after:
        if created is None or created < datetime.date.fromisoformat(created_after):
            return False
    if created_before:
        if created is None or created > datetime.date.fromisoformat(created_before):
            return False
    return True


def _enrich(c: dict) -> dict:
    item = dict(c)
    fm = _frontmatter(item.get("source", ""))
    heading = _heading_path(item.get("source", ""), item.get("parent_text", ""))
    item["source_type"] = _source_type(item.get("source", ""))
    item["tags"] = fm.get("tags", [])
    item["created"] = fm.get("created")
    item["heading"] = heading
    item["read_file_hint"] = _read_file_hint(item.get("source", ""), heading)
    return item


def _format_meta(c: dict) -> list[str]:
    tags = ", ".join(c.get("tags") or [])
    heading = c.get("heading") or ""
    return [
        f"  meta: type={c.get('source_type', 'unknown')}" + (f" | tags={tags}" if tags else ""),
        f"  heading: {heading or '[N/A]'}",
        f"  read_file_hint: {c.get('read_file_hint', _read_file_hint(c.get('source', '')))}",
    ]


def _format_result(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        c = _enrich(c)
        abs_path = _to_abs(c["source"])
        fname    = Path(c["source"]).name
        header = (
            f"[{i+1}] (source: {fname} | rrf: {c.get('rrf_score', 0):.4f} "
            f"| dense_hits: {c.get('dense_hits', 0)} | bm25_hits: {c.get('bm25_hits', 0)} "
            f"| raw_score: {c.get('score', 0):.3f} | file: {abs_path})"
        )
        parts.append("\n".join([header, *_format_meta(c), c["parent_text"]]))
    return "\n\n".join(parts)


def _format_related(related: list[dict]) -> str:
    """Machine-readable see-also footer: lower-ranked docs, no body."""
    if not related:
        return ""
    lines = ["", "RELATED_DOCS:", "use_when: top_chunks_insufficient_or_need_second_source", "skip_when: top_chunks_answer_the_question"]
    for c in related:
        c = _enrich(c)
        fname = Path(c["source"]).name
        lines.extend([
            f"- file: {fname}",
            f"  path: {_to_abs(c['source'])}",
            f"  rrf: {c.get('rrf_score', 0):.4f}",
            f"  heading: {c.get('heading') or '[N/A]'}",
            f"  read_file_hint: {c.get('read_file_hint')}",
        ])
    return "\n".join(lines)


def _format_files(chunks: list[dict], source_first: bool = False) -> str:
    seen: set[str] = set()
    lines = ["SOURCE_FIRST:" if source_first else "FILES:"]
    for c in chunks:
        if c["source"] in seen:
            continue
        seen.add(c["source"])
        c = _enrich(c)
        lines.extend([
            f"- file: {Path(c['source']).name}",
            f"  path: {_to_abs(c['source'])}",
            f"  rrf: {c.get('rrf_score', 0):.4f}",
            f"  retriever_coverage: dense={c.get('dense_hits', 0)}, bm25={c.get('bm25_hits', 0)}",
            f"  heading: {c.get('heading') or '[N/A]'}",
            f"  read_file_hint: {c.get('read_file_hint')}",
        ])
        if source_first:
            snippet = " ".join(c.get("parent_text", "").split())[:260]
            lines.append(f"  snippet: {snippet}")
    return "\n".join(lines)


@tool
def rag_retrieve(query: str | list[str], mode: str = "chunks", tags: str = "",
                 filename_contains: str = "", created_after: str = "",
                 created_before: str = "", source_type: str = "") -> str:
    """Retrieve top-5 document chunks from the local knowledge base.

    query: one query string, or a list of query variants of the SAME question
    (e.g. Thai sentence, English sentence, Thai keywords, English keywords).
    Every variant is searched separately (dense + BM25 each) and all ranked
    lists are RRF-fused into one ranking — cross-language variants let BM25
    reach documents written in the other language.

    mode: "chunks" returns top chunks; "files" returns matched files only;
          "source_first" prioritizes read_file hints with short snippets.
    filters: tags (all required), filename_contains, created_after/before
             (YYYY-MM-DD), source_type (extension such as md/pdf/docx).

    Returns raw parent chunks with absolute file paths and read_file hints.
    Use when you want to read source material directly.
    To read the full file, pass the 'file:' path to your read_file tool.
    """
    try:
        mode = (mode or "chunks").strip().lower()
        if mode not in {"chunks", "files", "source_first"}:
            return f"[error] rag_retrieve: invalid mode '{mode}' (use chunks/files/source_first)"
        raw = [query] if isinstance(query, str) else list(query)
        seen_q: set[str] = set()
        queries: list[str] = []
        for q in raw:
            q = (q or "").strip()
            if q and q.lower() not in seen_q:
                seen_q.add(q.lower())
                queries.append(q)
        if not queries:
            return "[error] rag_retrieve: empty query"
        fused = retriever.search(queries, top_k=10, top_fused=TOP_N + RELATED_N)
        fused = retriever.fetch_parents(fused)   # dedup only (no I/O); parent_text already present
        if not fused:
            return "[error] rag_retrieve: no candidates from dense/BM25 for query variants: " + " | ".join(queries[:4])
        before_filter = len(fused)
        fused = [
            c for c in fused
            if _matches_filters(c, tags=tags, filename_contains=filename_contains,
                                created_after=created_after, created_before=created_before,
                                source_type=source_type)
        ]
        if not fused:
            return (
                "[error] rag_retrieve: candidates found but filters removed all results "
                f"(candidates={before_filter}, tags={tags or '-'}, filename_contains={filename_contains or '-'}, "
                f"created_after={created_after or '-'}, created_before={created_before or '-'}, source_type={source_type or '-'})"
            )
        main = fused[:TOP_N]
        if mode == "files":
            return _format_files(fused, source_first=False)
        if mode == "source_first":
            return _format_files(fused, source_first=True)
        # related = next-ranked, deduped by source, excluding anything already shown in main
        seen = {c["source"] for c in main}
        related = []
        for c in fused[TOP_N:]:
            if c["source"] in seen:
                continue
            seen.add(c["source"])
            related.append(c)
        return _format_result(main) + _format_related(related)
    except Exception as e:
        return f"[error] {e}"
