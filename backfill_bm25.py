"""One-time backfill: add BM25 entries for Chroma chunks ingested before
BM25 support existed (file_registry's hash-skip gate never re-ingests
unchanged files, so they never got a chance to backfill on their own).
Re-run safely any time — bm25_add() is a no-op for ids already indexed.
"""
import store


def main():
    store._load_bm25()
    already = set(store._bm25_ids)  # read after _load_bm25() reassigns the module global
    col = store._get_collection()
    all_chunks = col.get(include=["documents", "metadatas"])

    added = 0
    sources_touched = set()
    for cid, doc, meta in zip(all_chunks["ids"], all_chunks["documents"], all_chunks["metadatas"]):
        if cid in already:
            continue
        store.bm25_add(doc, cid, meta["source"], flush=False)
        added += 1
        sources_touched.add(meta["source"])

    if added:
        store.bm25_flush()

    print(f"backfilled {added} chunks across {len(sources_touched)} sources")
    print(f"bm25 total chunks now: {len(already) + added}")


if __name__ == "__main__":
    main()
