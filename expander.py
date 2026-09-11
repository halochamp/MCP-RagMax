from __future__ import annotations
import re

from pythainlp import word_tokenize, pos_tag

CONNECTORS = {"แล้ว", "อีก", "ด้วย", "นั้น", "แบบ", "อันนั้น", "แล้วก็", "ที่ว่า"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "NUM", "ADJ"}
TH_STOPWORDS = {"ที่", "ของ", "และ", "หรือ", "ใน", "มี", "เป็น", "คือ", "ให้", "กับ",
                "จาก", "ได้", "ว่า", "จะ", "นี้", "นั้น", "แต่", "ก็", "แล้ว", "โดย"}
EN_STOPWORDS = {"the","a","an","is","are","was","were","what","how","why","of",
                "in","on","at","to","for","with","by","from","this","that","it"}


def _is_thai(text: str) -> bool:
    return any("฀" <= c <= "๿" for c in text)


def _normalize_query(text: str) -> str:
    """Deterministic whitespace/punctuation normalization; never calls an LLM."""
    compact = re.sub(r"\s+", " ", (text or "").strip())
    return compact


def _extract_keywords(text: str) -> str:
    if _is_thai(text):
        try:
            tagged = pos_tag(
                word_tokenize(text, engine="newmm"), corpus="orchid_ud"
            )
            words = [w for w, pos in tagged
                     if pos in CONTENT_POS and w not in TH_STOPWORDS and len(w) > 1]
        except Exception:
            words = [w for w in word_tokenize(text, engine="newmm")
                     if w not in TH_STOPWORDS and len(w) > 1]
    else:
        words = [w for w in text.lower().split()
                 if w not in EN_STOPWORDS and len(w) > 2]
    return " ".join(words)


def maybe_contextualize(query: str, history: list[dict]) -> str:
    """Prepend recent history for short/connector queries."""
    first_word = query.split()[0] if query.split() else ""
    if len(query) < 50 or first_word in CONNECTORS:
        recent = " ".join(m["content"] for m in history[-4:] if m.get("content"))
        return f"{recent} {query}".strip()
    return query


def expand(query: str, history: list[dict] | None = None) -> tuple[str, str, str]:
    """Return three deterministic query representations without an LLM.

    Q1 is the contextualized original, Q2 is a keyword-focused representation
    for BM25, and Q3 is a normalized lexical form. The multilingual MiniLM
    dense retriever handles cross-language semantics directly; callers such as
    Agent MAX may still pass their own same-intent variants through MCP.
    """
    q1 = _normalize_query(maybe_contextualize(query, history or []))
    q2 = _extract_keywords(q1)
    q3 = _normalize_query(re.sub(r"[^\w\u0E00-\u0E7F]+", " ", q1.casefold()))
    if q3 == q1.casefold():
        q3 = ""
    return q1, q2, q3
