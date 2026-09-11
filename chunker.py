from __future__ import annotations
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pythainlp import word_tokenize


def _has_thai(text: str) -> bool:
    return any("฀" <= c <= "๿" for c in text)


def _clean(text: str) -> str:
    # Remove transcript timestamps only when bracketed or leading a line.
    # A bare time-like token inside prose may be a ratio, market time, or other
    # meaningful data and must not be stripped silently.
    text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", text)
    text = re.sub(r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?\s*", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pack_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    window_len = 0
    for word in words:
        wlen = len(word)
        if wlen > chunk_size:
            if window:
                chunks.append("".join(window))
                window, window_len = [], 0
            stride = max(1, chunk_size - overlap)
            pieces = [word[i:i + chunk_size] for i in range(0, wlen, stride)]
            chunks.extend(pieces[:-1])
            window = [pieces[-1]]
            window_len = len(pieces[-1])
            continue
        if window_len + wlen > chunk_size and window:
            chunks.append("".join(window))
            tail: list[str] = []
            tail_len = 0
            for w in reversed(window):
                if tail_len + len(w) > overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w)
            while tail and tail_len + wlen > chunk_size:
                removed = tail.pop(0)
                tail_len -= len(removed)
            window, window_len = tail, tail_len
        window.append(word)
        window_len += wlen
    if window:
        chunks.append("".join(window))
    return chunks


def _thai_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split preserving Thai word boundaries.
    Priority: paragraph break → line break → word boundary.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    segments: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
            buf: list[str] = []
            buf_len = 0
            for line in lines:
                llen = len(line) + 1
                if buf_len + llen > chunk_size and buf:
                    segments.append("\n".join(buf))
                    buf, buf_len = [], 0
                buf.append(line)
                buf_len += llen
            if buf:
                segments.append("\n".join(buf))
        else:
            words = word_tokenize(para, engine="newmm", keep_whitespace=True)
            segments.extend(_pack_words(words, chunk_size, overlap))

    # merge small segments + add cross-segment overlap
    chunks: list[str] = []
    window_segs: list[str] = []
    for seg in segments:
        candidate = "\n\n".join([*window_segs, seg])
        if len(candidate) > chunk_size and window_segs:
            chunks.append("\n\n".join(window_segs))
            tail_segs: list[str] = []
            tail_len = 0
            for s in reversed(window_segs):
                if tail_len + len(s) > overlap:
                    break
                tail_segs.insert(0, s)
                tail_len += len(s)
            while tail_segs and len("\n\n".join([*tail_segs, seg])) > chunk_size:
                tail_segs.pop(0)
            window_segs = tail_segs
        window_segs.append(seg)
    if window_segs:
        chunks.append("\n\n".join(window_segs))

    return chunks or [text]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks. Thai path respects word boundaries."""
    text = _clean(text)
    if not text:
        return []
    if _has_thai(text):
        return _thai_split(text, chunk_size, overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    ).split_text(text)


def chunk_document(text: str,
                   parent_size: int = 3000, parent_overlap: int = 400,
                   child_size: int = 400,  child_overlap: int = 80,
                   ) -> list[dict]:
    """Return list of {parent_text, child_text, child_index} dicts."""
    parents = split_text(text, parent_size, parent_overlap)
    result = []
    for p_idx, parent in enumerate(parents):
        children = split_text(parent, child_size, child_overlap)
        for c_idx, child in enumerate(children):
            result.append({
                "parent_text": parent,
                "child_text": child,
                "parent_index": p_idx,
                "child_index": c_idx,
            })
    return result
