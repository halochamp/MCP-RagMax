from __future__ import annotations
import json
from pathlib import Path

from config import MEMORY_PATH, source_path
from langchain_core.tools import tool

import retriever
import kb_operations
from expander import expand

MAX_CHUNKS    = 5   # child chunks sent to reranker
KEEP_CHUNKS   = 3   # parent chunks returned
QUALITY_HIGH  = 0.62
QUALITY_LOW   = 0.35


def _to_abs(s: str) -> str:
    """Resolve and validate a source path inside the configured knowledge root."""
    return str(source_path(s))


def _rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Deterministically keep the top RRF-ranked chunks; no LLM reranker."""
    del query  # ranking is already determined by Dense + BM25 + RRF
    return chunks[:KEEP_CHUNKS]


def _compute_quality(chunks: list[dict]) -> str:
    if not chunks:
        return "low"
    first = chunks[0]
    top = first.get("dense_score")
    if top is None and first.get("retriever") in {"dense", "hybrid"}:
        top = first.get("score", 0.0)
    lexical_hits = int(first.get("bm25_hits", 0))
    n   = len(chunks)
    if top is None:
        return "medium" if lexical_hits else "low"
    if top > 0.78 and n >= 3 and lexical_hits:
        return "high"
    if top < QUALITY_LOW and not lexical_hits:
        return "low"
    if top > QUALITY_HIGH or n >= 2:
        return "medium"
    return "medium"


def _format_result(chunks: list[dict], quality: str) -> str:
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    sources_line = "SOURCES: " + " | ".join(
        _to_abs(s) for s in sources
    )
    body = "\n\n".join(
        f"[{i+1}] (source: {Path(c['source']).name} | {c.get('retriever','?')}={c.get('score', 0):.3f})\n{c['parent_text']}"
        for i, c in enumerate(chunks)
    )
    prefix = "[low_quality]\n" if quality == "low" else ""
    return prefix + sources_line + "\n" + body


@tool
def list_knowledge() -> str:
    """Describe the KB deterministically from canonical metadata; no LLM."""
    try:
        paths = kb_operations.registered_paths()
        if not paths:
            return "ยังไม่มีไฟล์ใน knowledge base ครับ"

        total = len(paths)
        index_path = Path(__file__).resolve().parent / "rag_index.json"
        payload: dict = {}
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            payload = {}

        lines = [f"พบไฟล์ทั้งหมด {total} ไฟล์"]
        topics = str(payload.get("topics_line") or "").strip()
        if topics:
            lines.append(f"หัวข้อ: {topics}")
        tags = payload.get("tags") or {}
        if isinstance(tags, dict) and tags:
            top_tags = list(tags.items())[:20]
            lines.append("tags: " + " | ".join(f"{name}({count})" for name, count in top_tags))
        source_types = payload.get("source_types") or {}
        if isinstance(source_types, dict) and source_types:
            lines.append("ชนิดไฟล์: " + " | ".join(f"{name}({count})" for name, count in source_types.items()))
        if len(lines) == 1:
            names = " | ".join(Path(path).name for path in paths[:20])
            lines.append(f"ตัวอย่างไฟล์: {names}")
        return "\n".join(lines)
    except Exception as e:
        return f"[error] {e}"


_MAX_SHOW_FILES = 30


@tool
def search_files(query: str) -> str:
    """Search for files in the knowledge base whose filename contains the query keyword.

    Use when the user asks to find files by name or topic keyword (e.g. 'หาไฟล์ที่มีคำว่า X').
    Returns up to 30 matching filenames. Shows remaining count if more exist.
    """
    try:
        paths = kb_operations.registered_paths()
        if not paths:
            return "ยังไม่มีไฟล์ใน knowledge base ครับ"
        q = query.lower()
        matched = [p for p in paths if q in Path(p).name.lower()]
        if not matched:
            return f"ไม่พบไฟล์ที่มีคำว่า '{query}' ในชื่อไฟล์"
        total = len(matched)
        lines = "\n".join(f"  - {Path(p).name}" for p in matched[:_MAX_SHOW_FILES])
        suffix = f"\n  ... ยังมีอีก {total - _MAX_SHOW_FILES} ไฟล์" if total > _MAX_SHOW_FILES else ""
        return f"พบ {total} ไฟล์ที่มีคำว่า '{query}':\n{lines}{suffix}"
    except Exception as e:
        return f"[error] {e}"


@tool
def read_file(filename: str) -> str:
    """Read and return the raw content of a file from the knowledge base.

    Use when the user says 'เปิดไฟล์ X', 'อ่านไฟล์ X', or explicitly asks to see file content.
    Accept partial filename, exact filename, or absolute path.
    If multiple files match, ask user to be more specific.
    """
    try:
        try:
            _, text = kb_operations.read_registered_file(filename)
            return text
        except kb_operations.RegisteredFileNotFound:
            return f"[error] ไม่พบไฟล์ '{filename}' ใน knowledge base"
        except kb_operations.RegisteredFileAmbiguous as exc:
            names = "\n".join(f"  - {Path(p).name}" for p in exc.matches[:10])
            more = f"\n  ... และอีก {len(exc.matches)-10} ไฟล์" if len(exc.matches) > 10 else ""
            return f"[error] พบหลายไฟล์ที่ตรงกับ '{filename}':\n{names}{more}\nกรุณาระบุชื่อให้ชัดเจนขึ้น"
    except Exception as e:
        return f"[error] {e}"


@tool
def save_memory(text: str) -> str:
    """Save a note to persistent memory file (data/memory.md).

    Use when user says 'จำไว้', 'บันทึกไว้', 'จำไว้ว่า', or 'remember'.
    IMPORTANT: pass the user's EXACT words as text — do NOT paraphrase, translate, or summarize.
    Memory is loaded automatically on every session start.
    """
    try:
        from datetime import datetime
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"- [{timestamp}] {text.strip()}\n"
        with open(MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"บันทึกแล้วครับ: {text.strip()}"
    except Exception as e:
        return f"[error] {e}"


@tool
def rag_search(query: str) -> str:
    """Search the local knowledge base and return relevant document chunks.

    Use when the user asks about topics that may be in the knowledge base.
    Returns 2-3 parent chunks with source files.
    Returns [low_quality] prefix if results are weak — caller should retry with rephrased query.
    Returns [error] prefix on failure.
    Do NOT use for general knowledge, math, small talk, or coding questions.
    """
    try:
        import _progress as P
        P.report("🔍 ค้นหาใน knowledge base…", "📝 สร้าง query variants แบบ deterministic…")
        q1, q2, q3 = expand(query)
        queries = [q for q in [q1, q2, q3] if q.strip()]

        P.report("🔍 ค้นหาใน knowledge base…", "🔎 Dense search + BM25…")
        chunks = retriever.search(queries, top_k=10, top_fused=MAX_CHUNKS)

        P.report("🔍 ค้นหาใน knowledge base…", "⚖️  RRF fusion…")
        chunks = retriever.fetch_parents(chunks)
        if not chunks:
            return "[low_quality]\nNo results found in knowledge base."

        P.report("🔍 ค้นหาใน knowledge base…", "🎯 เลือกอันดับสูงสุดจาก RRF…")
        selected = _rerank(q1, chunks)
        quality  = _compute_quality(selected)
        return _format_result(selected, quality)
    except Exception as e:
        return f"[error] {e}"
