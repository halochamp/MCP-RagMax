from __future__ import annotations
import hashlib
import os
import pickle
import threading
from contextlib import contextmanager
import fcntl
from pathlib import Path

from config import BM25_PATH, CHROMA_DIR as DB_DIR

import chromadb
from rank_bm25 import BM25Okapi
from pythainlp import word_tokenize

_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None
_bm25: BM25Okapi | None = None
_bm25_corpus: list[list[str]] = []
_bm25_ids: list[str] = []
_bm25_sources: list[str] = []
_bm25_mtime: float = 0.0
_bm25_pending: dict[str, tuple[list[str], str]] = {}
_lock = threading.Lock()


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                DB_DIR.mkdir(parents=True, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
                _collection = _chroma_client.get_or_create_collection(
                    name="endeavor_rag",
                    metadata={"hnsw:space": "cosine"},
                )
    return _collection


def _chunk_id(source: str, parent_idx: int, child_idx: int) -> str:
    return hashlib.md5(f"{source}:{parent_idx}:{child_idx}".encode()).hexdigest()


def upsert_chunk(child_text: str, parent_text: str, embedding: list[float],
                 source: str, parent_idx: int, child_idx: int) -> str:
    cid = _chunk_id(source, parent_idx, child_idx)
    _get_collection().upsert(
        ids=[cid],
        embeddings=[embedding],
        documents=[child_text],
        metadatas=[{"source": source, "parent_text": parent_text,
                    "parent_idx": parent_idx, "child_idx": child_idx}],
    )
    return cid


def dense_search(query_vec: list[float], top_k: int = 10) -> list[dict]:
    col = _get_collection()
    n = col.count()
    if n == 0:
        return []
    k = min(top_k, n)
    res = col.query(query_embeddings=[query_vec], n_results=k,
                    include=["documents", "metadatas", "distances"])
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({
            "id": res["ids"][0][len(results)],
            "child_text": doc,
            "parent_text": meta["parent_text"],
            "source": meta["source"],
            "score": float(1.0 - dist),
            "retriever": "dense",
        })
    return results


def has_source(source: str) -> bool:
    """Return True if at least one chunk for this source exists in the vector store."""
    col = _get_collection()
    res = col.get(where={"source": source}, include=[])
    return bool(res["ids"])


def delete_by_source(source: str) -> int:
    col = _get_collection()
    res = col.get(where={"source": source}, include=[])
    ids = res["ids"]
    if ids:
        col.delete(ids=ids)
    return len(ids)


def get_chunk_hash(child_text: str) -> str:
    return hashlib.sha256(child_text.encode()).hexdigest()


# ── BM25 ─────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return word_tokenize(text.lower(), engine="newmm", keep_whitespace=False)


def _load_bm25():
    global _bm25, _bm25_corpus, _bm25_ids, _bm25_sources, _bm25_mtime
    if BM25_PATH.exists():
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25_corpus = data["corpus"]
        _bm25_ids = data["ids"]
        _bm25_sources = data["sources"]
        _bm25 = BM25Okapi(_bm25_corpus) if _bm25_corpus else None
        _bm25_mtime = BM25_PATH.stat().st_mtime
    else:
        _bm25_corpus = []
        _bm25_ids = []
        _bm25_sources = []
        _bm25 = None
        _bm25_mtime = 0.0


def _bm25_on_disk_changed() -> bool:
    """True if bm25.pkl was written by another process since our last load."""
    if not BM25_PATH.exists():
        return _bm25_mtime != 0.0
    return BM25_PATH.stat().st_mtime != _bm25_mtime


@contextmanager
def _bm25_file_lock():
    """Serialize BM25 read-modify-write cycles across processes."""
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = BM25_PATH.with_name(BM25_PATH.name + ".lock")
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _save_bm25():
    global _bm25_mtime
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(BM25_PATH) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"corpus": _bm25_corpus, "ids": _bm25_ids,
                     "sources": _bm25_sources}, f)
    os.replace(tmp, BM25_PATH)
    _bm25_mtime = BM25_PATH.stat().st_mtime


def _rebuild_bm25():
    global _bm25
    _bm25 = BM25Okapi(_bm25_corpus) if _bm25_corpus else None


def _apply_bm25_pending():
    global _bm25_pending
    known = set(_bm25_ids)
    for chunk_id, (tokens, source) in _bm25_pending.items():
        if chunk_id in known:
            continue
        _bm25_corpus.append(tokens)
        _bm25_ids.append(chunk_id)
        _bm25_sources.append(source)
        known.add(chunk_id)
    _bm25_pending = {}


def bm25_add(child_text: str, chunk_id: str, source: str, flush: bool = True):
    global _bm25_pending
    with _lock:
        if not _bm25_corpus:
            _load_bm25()
        if chunk_id in _bm25_ids or chunk_id in _bm25_pending:
            return
        _bm25_pending[chunk_id] = (_tokenize(child_text), source)
        if flush:
            with _bm25_file_lock():
                _load_bm25()
                _apply_bm25_pending()
                _rebuild_bm25()
                _save_bm25()


def bm25_flush():
    """Rebuild index and persist — call once after batch bm25_add(flush=False)."""
    with _lock:
        with _bm25_file_lock():
            _load_bm25()
            _apply_bm25_pending()
            _rebuild_bm25()
            _save_bm25()


def bm25_delete_by_source(source: str):
    global _bm25_corpus, _bm25_ids, _bm25_sources, _bm25_pending
    with _lock:
        with _bm25_file_lock():
            _load_bm25()
            _bm25_pending = {
                cid: value for cid, value in _bm25_pending.items()
                if value[1] != source
            }
            keep = [(c, i, s) for c, i, s in zip(_bm25_corpus, _bm25_ids, _bm25_sources)
                    if s != source]
            if not keep:
                _bm25_corpus, _bm25_ids, _bm25_sources = [], [], []
            else:
                _bm25_corpus, _bm25_ids, _bm25_sources = map(list, zip(*keep))
            _rebuild_bm25()
            _save_bm25()


def health_check() -> list[str]:
    """Whole-KB consistency check: BM25 vs Chroma sync. Returns anomaly descriptions (empty = healthy)."""
    issues: list[str] = []
    col = _get_collection()
    chunks = col.get(include=["metadatas"])
    chroma_ids = set(chunks["ids"])
    valid_metadata = [m for m in chunks["metadatas"] if isinstance(m, dict) and "source" in m]
    malformed_count = len(chunks["metadatas"]) - len(valid_metadata)
    if malformed_count:
        issues.append(f"{malformed_count} Chroma chunks have malformed source metadata")
    chroma_sources = {m["source"] for m in valid_metadata}

    with _lock:
        _load_bm25()
        bm25_ids = set(_bm25_ids)
        bm25_sources = set(_bm25_sources)

    missing_from_bm25 = chroma_ids - bm25_ids
    if missing_from_bm25:
        issues.append(f"{len(missing_from_bm25)} Chroma chunks missing from BM25 index (run backfill_bm25.py)")

    orphaned_in_bm25 = bm25_ids - chroma_ids
    if orphaned_in_bm25:
        issues.append(f"{len(orphaned_in_bm25)} BM25 entries orphaned (no matching Chroma chunk)")

    if chroma_sources != bm25_sources:
        issues.append(
            f"source mismatch — chroma-only: {len(chroma_sources - bm25_sources)}, "
            f"bm25-only: {len(bm25_sources - chroma_sources)}"
        )

    return issues


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    with _lock:
        if not _bm25_corpus or _bm25_on_disk_changed():
            _load_bm25()
        bm25, ids, corpus = _bm25, list(_bm25_ids), list(_bm25_corpus)
    if not bm25:
        return []
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    query_tokens = set(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked = [
        (idx, score) for idx, score in ranked
        if idx < len(corpus) and query_tokens.intersection(corpus[idx])
    ][:top_k]
    id_to_score = {ids[idx]: float(score) for idx, score in ranked if idx < len(ids)}
    if not id_to_score:
        return []
    col = _get_collection()
    res = col.get(ids=list(id_to_score), include=["documents", "metadatas"])
    results = []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        results.append({
            "id": cid,
            "child_text": doc,
            "parent_text": meta["parent_text"],
            "source": meta["source"],
            "score": id_to_score[cid],  # raw BM25 score > 0
            "retriever": "bm25",
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
