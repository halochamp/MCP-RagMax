"""_test_e2e.py — full ingest -> search -> delete round trip"""
import os
import tempfile
from pathlib import Path

from _runner import Runner

r = Runner("e2e")


def t28_e2e():
    from ingestor import ingest_file, delete_file
    from retriever import search, fetch_parents

    content = ("Retrieval Augmented Generation (RAG) combines LLMs with external knowledge.\n"
               "It retrieves relevant documents before generating an answer.\n\n") * 8

    import config
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.KNOWLEDGE_DIR / "_e2e_rag_probe.txt"
    path.write_text(content, encoding="utf-8")

    try:
        n = ingest_file(path)
        assert n > 0, "No chunks stored"

        chunks = search(["RAG retrieval augmented generation"], top_k=5, top_fused=3)
        parents = fetch_parents(chunks)
        assert len(parents) > 0, "No results found after ingest"
        assert any("RAG" in p["parent_text"] or "retrieval" in p["parent_text"].lower()
                   for p in parents)

        delete_file(path)
    finally:
        path.unlink(missing_ok=True)


r.test("T28 ingest txt + search e2e", t28_e2e)

if __name__ == "__main__":
    r.exit()
