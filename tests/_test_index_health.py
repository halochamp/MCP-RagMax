"""_test_index_health.py — whole-KB consistency checks against the live production index.

Read-only: no writes to data/chroma or data/bm25.pkl. Catches the class of bug
where dense (Chroma) and lexical (BM25) indices silently drift apart — e.g. a
backfill gap after BM25 support was added post-hoc (fixed via backfill_bm25.py).
"""
from _runner import Runner

r = Runner("index_health")


def t36_index_health_check():
    from store import health_check
    issues = health_check()
    assert not issues, "; ".join(issues)


def t38_registry_files_have_chunks():
    """Informational: registry entries with zero chunks (benign dedup ghosts) shouldn't silently grow unbounded."""
    from file_registry import ghost_files
    ghosts = ghost_files()
    assert len(ghosts) <= 10, f"{len(ghosts)} registered files have zero chunks (expected <=10 known dedup ghosts): {ghosts}"


def t39_ghost_files_ignores_malformed_metadata():
    import file_registry
    import store

    class FakeCollection:
        def count(self):
            return 1

        def get(self, include=None):
            return {"metadatas": [{}]}

    original_collection = store._get_collection
    original_registry = file_registry.all_registered
    try:
        store._get_collection = lambda: FakeCollection()
        file_registry.all_registered = lambda: []
        assert file_registry.ghost_files() == []
    finally:
        store._get_collection = original_collection
        file_registry.all_registered = original_registry


def t40_orientation_policy_must_match_caller_llm_contract():
    import kb_operations

    fp = "a" * 64
    legacy = {"total": 2, "registry_fingerprint": fp, "topic_method": "deterministic"}
    assert kb_operations._orientation_status(legacy, fp, 2) == ("stale", 2)

    current = {
        "total": 2,
        "registry_fingerprint": fp,
        "orientation_policy_id": "caller-llm-topics:v1",
        "topic_method": "caller_llm",
    }
    assert kb_operations._orientation_status(current, fp, 2) == ("ready", 2)
    assert kb_operations._orientation_status(current, "b" * 64, 2) == ("stale", 2)
    assert kb_operations._orientation_status(current, fp, 3) == ("stale", 2)


r.test("T36 store.health_check() — bm25/chroma sync", t36_index_health_check)
r.test("T38 registry ghost-file count bounded", t38_registry_files_have_chunks)
r.test("T39 malformed Chroma metadata does not crash ghost check", t39_ghost_files_ignores_malformed_metadata)
r.test("T40 orientation policy must match caller-LLM contract", t40_orientation_policy_must_match_caller_llm_contract)

if __name__ == "__main__":
    r.exit()
