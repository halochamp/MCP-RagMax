"""_test_retriever.py — retriever._rrf_merge / fetch_parents"""
from _runner import Runner

r = Runner("retriever")


def t24_rrf_merge():
    from retriever import _rrf_merge
    list1 = [{"id": "a", "score": 0.9, "retriever": "dense", "query_variant": 0, "child_text": "", "parent_text": "", "source": ""},
             {"id": "b", "score": 0.7, "retriever": "dense", "query_variant": 0, "child_text": "", "parent_text": "", "source": ""}]
    list2 = [{"id": "b", "score": 0.8, "retriever": "bm25", "query_variant": 1, "child_text": "", "parent_text": "", "source": ""},
             {"id": "c", "score": 0.6, "retriever": "bm25", "query_variant": 1, "child_text": "", "parent_text": "", "source": ""}]
    fused = _rrf_merge([list1, list2], k=60, top_n=3)
    ids = [rr["id"] for rr in fused]
    assert "b" in ids   # b appears in both → should rank high
    assert ids[0] == "b"
    assert fused[0]["rrf_score"] > 0
    assert fused[0]["dense_hits"] == 1
    assert fused[0]["bm25_hits"] == 1
    assert fused[0]["query_variant_hits"] == [0, 1]


def t25_fetch_parents_dedup():
    from retriever import fetch_parents
    chunks = [
        {"id": "1", "source": "a.md", "parent_text": "same parent", "child_text": "c1"},
        {"id": "2", "source": "a.md", "parent_text": "same parent", "child_text": "c2"},
        {"id": "3", "source": "b.md", "parent_text": "other parent", "child_text": "c3"},
    ]
    result = fetch_parents(chunks)
    assert len(result) == 2  # dedup same parent

    prefix = "ซ้ำ" * 60
    distinct_long_parents = [
        {"id": "4", "source": "a.md", "parent_text": prefix + "A", "child_text": "c4"},
        {"id": "5", "source": "a.md", "parent_text": prefix + "B", "child_text": "c5"},
    ]
    assert len(fetch_parents(distinct_long_parents)) == 2


def t50_search_fills_unique_parent_budget():
    import retriever
    original_encode = retriever.embedder.encode
    original_dense = retriever.store.dense_search
    original_bm25 = retriever.store.bm25_search
    same = [
        {"id": f"same-{i}", "score": 0.8 - i * 0.01, "retriever": "dense",
         "child_text": f"c{i}", "parent_text": "same parent", "source": "same.md"}
        for i in range(5)
    ]
    unique = {"id": "unique", "score": 0.7, "retriever": "dense",
              "child_text": "u", "parent_text": "unique parent", "source": "unique.md"}
    try:
        retriever.embedder.encode = lambda queries: [[1.0, 0.0] for _ in queries]
        retriever.store.dense_search = lambda vec, top_k=10: same + [unique]
        retriever.store.bm25_search = lambda query, top_k=10: []
        results = retriever.search(["query"], top_k=10, top_fused=2)
        assert len(results) == 2
        assert len(retriever.fetch_parents(results)) == 2
    finally:
        retriever.embedder.encode = original_encode
        retriever.store.dense_search = original_dense
        retriever.store.bm25_search = original_bm25


def t51_fusion_does_not_compare_dense_and_bm25_raw_scales():
    from retriever import _rrf_merge
    from rag_search import _compute_quality
    dense = [
        {"id": f"d{i}", "score": 0.31, "retriever": "dense",
         "child_text": "", "parent_text": f"p{i}", "source": f"s{i}"}
        for i in range(3)
    ]
    sparse = [dict(item, score=8.0 - i, retriever="bm25") for i, item in enumerate(dense)]
    mixed = _rrf_merge([dense, sparse], top_n=3)
    assert mixed[0]["score"] == 0.31
    assert mixed[0]["dense_score"] == 0.31
    assert mixed[0]["bm25_score"] == 8.0
    assert _compute_quality(mixed) != "high"
    assert _compute_quality(_rrf_merge([dense], top_n=3)) == "low"


r.test("T24 RRF ranking", t24_rrf_merge)
r.test("T25 fetch_parents dedup", t25_fetch_parents_dedup)
r.test("T50 search fills unique-parent budget", t50_search_fills_unique_parent_budget)
r.test("T51 fusion keeps raw score scales separate", t51_fusion_does_not_compare_dense_and_bm25_raw_scales)

if __name__ == "__main__":
    r.exit()
