from __future__ import annotations

import embedder
import store

RRF_K   = 60
TOP_K   = 10
TOP_FUSED = 5


def _rrf_merge(ranked_lists: list[list[dict]], k: int = RRF_K,
               top_n: int = TOP_FUSED) -> list[dict]:
    """Reciprocal Rank Fusion across multiple result lists."""
    scores: dict[str, float] = {}
    items:  dict[str, dict]  = {}
    retriever_hits: dict[str, dict[str, int]] = {}
    query_hits: dict[str, set[int]] = {}
    dense_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in items:
                items[doc_id] = dict(item)
            retriever = item.get("retriever", "?")
            raw_score = float(item.get("score", 0.0))
            if retriever == "dense":
                dense_scores[doc_id] = max(dense_scores.get(doc_id, float("-inf")), raw_score)
            elif retriever == "bm25":
                bm25_scores[doc_id] = max(bm25_scores.get(doc_id, float("-inf")), raw_score)
            retriever_hits.setdefault(doc_id, {})
            retriever_hits[doc_id][retriever] = retriever_hits[doc_id].get(retriever, 0) + 1
            if "query_variant" in item:
                query_hits.setdefault(doc_id, set()).add(int(item["query_variant"]))

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    out: list[dict] = []
    for doc_id, rrf_score in fused:
        item = dict(items[doc_id])
        hits = retriever_hits.get(doc_id, {})
        item["rrf_score"] = float(rrf_score)
        item["dense_hits"] = int(hits.get("dense", 0))
        item["bm25_hits"] = int(hits.get("bm25", 0))
        item["retriever_hits"] = dict(sorted(hits.items()))
        item["query_variant_hits"] = sorted(query_hits.get(doc_id, set()))
        item["dense_score"] = dense_scores.get(doc_id)
        item["bm25_score"] = bm25_scores.get(doc_id)
        if item["dense_score"] is not None:
            item["score"] = item["dense_score"]
        elif item["bm25_score"] is not None:
            item["score"] = item["bm25_score"]
        if item["dense_hits"] and item["bm25_hits"]:
            item["retriever"] = "hybrid"
        out.append(item)
    return out


def search(queries: list[str], top_k: int = TOP_K,
           top_fused: int = TOP_FUSED) -> list[dict]:
    """Search with multiple queries, merge via RRF.

    Args:
        queries: list of query strings (Q1, Q2, Q3 or just Q1)
        top_k:   candidates per retriever per query
        top_fused: final results after RRF

    Returns: list of chunk dicts with id, child_text, parent_text, source, score
    """
    all_lists: list[list[dict]] = []

    vecs = embedder.encode(queries)
    for q_idx, (query, vec) in enumerate(zip(queries, vecs)):
        dense = store.dense_search(vec, top_k=top_k)
        sparse = store.bm25_search(query, top_k=top_k)
        for item in dense + sparse:
            item["query_variant"] = q_idx
            item["query_text"] = query
        all_lists.extend([dense, sparse])

    # Fuse the complete candidate pool before parent de-duplication. Truncating
    # child hits first can spend the entire budget on one parent and hide the
    # next distinct context just below the cutoff.
    fused = _rrf_merge(all_lists, k=RRF_K, top_n=sum(map(len, all_lists)))
    return fetch_parents(fused)[:top_fused]


def fetch_parents(chunks: list[dict]) -> list[dict]:
    """Deduplicate by parent_text (same parent may have multiple child hits)."""
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for chunk in chunks:
        key = (chunk["source"], chunk.get("parent_text", ""))
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result
