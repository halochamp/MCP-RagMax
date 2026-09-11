"""Smoke test — run from anywhere: python _smoke_test.py"""
import sys, os, json, csv, tempfile, traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✅"; FAIL = "❌"
results = []

def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T01 chunker ===")

def t01_thai():
    from chunker import split_text
    chunks = split_text("การเรียนรู้ของเครื่องคือสาขาหนึ่งของปัญญาประดิษฐ์ " * 20, 200, 40)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250, f"chunk too long: {len(c)}"

def t02_english():
    from chunker import split_text
    chunks = split_text("Machine learning is a subset of AI. " * 30, 200, 40)
    assert len(chunks) > 1

def t03_parent_child():
    from chunker import chunk_document
    pairs = chunk_document("hello world " * 100)
    assert len(pairs) > 0
    assert "parent_text" in pairs[0] and "child_text" in pairs[0]
    assert len(pairs[0]["child_text"]) <= 420

def t04_newline_split():
    from chunker import split_text
    text = "บทที่ 1\n\nเนื้อหาแรก\n\nบทที่ 2\n\nเนื้อหาสอง"
    chunks = split_text(text, 20, 5)
    assert len(chunks) >= 2

test("T01 Thai word boundary", t01_thai)
test("T02 English fallback", t02_english)
test("T03 parent/child pairs", t03_parent_child)
test("T04 newline split priority", t04_newline_split)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T05 embedder ===")

def t05_encode_one():
    from embedder import encode_one
    v = encode_one("machine learning")
    assert len(v) == 384
    assert abs(sum(x*x for x in v)**0.5 - 1.0) < 0.01  # normalized

def t06_encode_batch():
    from embedder import encode
    vs = encode(["hello", "world", "สวัสดี"])
    assert len(vs) == 3
    assert all(len(v) == 384 for v in vs)

def t07_empty_encode():
    from embedder import encode
    assert encode([]) == []

test("T05 encode_one normalized", t05_encode_one)
test("T06 encode batch", t06_encode_batch)
test("T07 empty input", t07_empty_encode)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T08 store ===")

def t08_upsert_dense_delete():
    from embedder import encode_one
    from store import upsert_chunk, dense_search, delete_by_source
    vec = encode_one("unit test document about AI")
    upsert_chunk("unit test AI", "parent unit test AI document", vec, "_smoke_.md", 0, 0)
    res = dense_search(encode_one("artificial intelligence"), top_k=5)
    assert any(r["source"] == "_smoke_.md" for r in res)
    n = delete_by_source("_smoke_.md")
    assert n >= 1

def t09_bm25_add_search_delete():
    from embedder import encode_one
    from store import upsert_chunk, bm25_add, bm25_delete_by_source
    vec = encode_one("python programming language")
    cid = upsert_chunk("python programming", "parent python", vec, "_smoke2_.md", 0, 0)
    bm25_add("python programming", cid, "_smoke2_.md")
    from store import bm25_search, delete_by_source
    # add a few more docs so BM25 scores > 0
    for i in range(3):
        v2 = encode_one(f"java c++ golang language {i}")
        cid2 = upsert_chunk(f"java c++ golang {i}", f"parent java {i}", v2, "_smoke2_.md", i+1, 0)
        bm25_add(f"java c++ golang {i}", cid2, "_smoke2_.md")
    res = bm25_search("python programming", top_k=5)
    assert isinstance(res, list)  # may be empty on tiny corpus, just no crash
    delete_by_source("_smoke2_.md")
    bm25_delete_by_source("_smoke2_.md")

def t10_chunk_hash():
    from store import get_chunk_hash
    h1 = get_chunk_hash("hello")
    h2 = get_chunk_hash("hello")
    h3 = get_chunk_hash("world")
    assert h1 == h2
    assert h1 != h3

test("T08 upsert + dense search + delete", t08_upsert_dense_delete)
test("T09 bm25 add + search + delete", t09_bm25_add_search_delete)
test("T10 chunk hash deterministic", t10_chunk_hash)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T11 file_registry ===")

def t11_registry():
    import config
    from file_registry import check, register, deregister
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.KNOWLEDGE_DIR / "_smoke_registry.txt"
    path.write_text("hello", encoding="utf-8")
    assert check(str(path)) == "new"
    register(str(path))
    assert check(str(path)) == "skip"
    path.write_text("changed content", encoding="utf-8")
    assert check(str(path)) == "changed"
    deregister(str(path))
    assert check(str(path)) == "new"
    path.unlink(missing_ok=True)

test("T11 new/skip/changed/deregister", t11_registry)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T12 ingestor file loaders ===")

def t12_txt():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
        f.write("hello world machine learning"); path = Path(f.name)
    text = load_file(path); os.unlink(path)
    assert "machine learning" in text

def t13_md():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w") as f:
        f.write("# Title\n\ncontent here"); path = Path(f.name)
    text = load_file(path); os.unlink(path)
    assert "content" in text

def t14_csv():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as f:
        csv.writer(f).writerows([["name","age"],["Alice","30"],["Bob","25"]]); path = Path(f.name)
    text = load_file(path); os.unlink(path)
    assert "[columns: name, age]" in text
    assert "Alice" in text

def t15_json():
    from ingestor import load_file
    data = {"users": [{"name": "Alice", "age": 30}], "version": 1}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(data, f); path = Path(f.name)
    text = load_file(path); os.unlink(path)
    assert "[json schema:" in text
    assert "Alice" in text
    assert "version" in text

def t16_csv_header_every_chunk():
    from ingestor import _load_csv
    from chunker import chunk_document
    rows = [["col1","col2","col3"]] + [[f"val{i}a", f"val{i}b", f"val{i}c"] for i in range(20)]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as f:
        csv.writer(f).writerows(rows); path = Path(f.name)
    text = _load_csv(path); os.unlink(path)
    chunks = chunk_document(text)
    for c in chunks:
        assert "[columns:" in c["child_text"], f"Missing header in chunk: {c['child_text'][:100]}"

def t17_json_header_every_chunk():
    from ingestor import _load_json
    from chunker import chunk_document
    data = {"items": [{"id": i, "val": f"value_{i}" * 5} for i in range(30)]}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(data, f); path = Path(f.name)
    text = _load_json(path); os.unlink(path)
    chunks = chunk_document(text)
    for c in chunks:
        assert "[json schema:" in c["child_text"], f"Missing header in chunk: {c['child_text'][:100]}"

test("T12 txt loader", t12_txt)
test("T13 md loader", t13_md)
test("T14 csv loader + header", t14_csv)
test("T15 json loader + schema", t15_json)
test("T16 csv header in every child chunk", t16_csv_header_every_chunk)
test("T17 json schema in every child chunk", t17_json_header_every_chunk)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T18 expander ===")

def t18_is_thai():
    from expander import _is_thai
    assert _is_thai("สวัสดี") is True
    assert _is_thai("hello") is False
    assert _is_thai("hello สวัสดี") is True

def t19_keywords_thai():
    from expander import _extract_keywords
    kw = _extract_keywords("การเรียนรู้ของเครื่องและปัญญาประดิษฐ์")
    assert isinstance(kw, str) and len(kw) > 0

def t20_keywords_english():
    from expander import _extract_keywords
    kw = _extract_keywords("what is machine learning and artificial intelligence")
    assert "machine" in kw or "learning" in kw

def t21_contextualize():
    from expander import maybe_contextualize
    history = [{"role":"user","content":"ถามเรื่อง RAG"},
               {"role":"assistant","content":"RAG คือ..."}]
    q = maybe_contextualize("แล้วมีประโยชน์ยังไง", history)
    assert "RAG" in q

def t22_translate():
    from expander import expand
    result = expand("การลงทุนระยะยาว")
    assert isinstance(result, tuple) and len(result) == 3
    assert all(isinstance(item, str) for item in result)
    assert result[0]

def t23_expand_returns_3():
    from expander import expand
    q1, q2, q3 = expand("machine learning คืออะไร")
    assert isinstance(q1, str) and isinstance(q2, str) and isinstance(q3, str)
    assert q1 == "machine learning คืออะไร"

test("T18 _is_thai detection", t18_is_thai)
test("T19 Thai keyword extract", t19_keywords_thai)
test("T20 English keyword extract", t20_keywords_english)
test("T21 contextualize connector word", t21_contextualize)
test("T22 translate no crash", t22_translate)
test("T23 expand returns 3 strings", t23_expand_returns_3)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T24 retriever ===")

def t24_rrf_merge():
    from retriever import _rrf_merge
    list1 = [{"id":"a","score":0.9,"child_text":"","parent_text":"","source":""},
             {"id":"b","score":0.7,"child_text":"","parent_text":"","source":""}]
    list2 = [{"id":"b","score":0.8,"child_text":"","parent_text":"","source":""},
             {"id":"c","score":0.6,"child_text":"","parent_text":"","source":""}]
    fused = _rrf_merge([list1, list2], k=60, top_n=3)
    ids = [r["id"] for r in fused]
    assert "b" in ids   # b appears in both → should rank high
    assert ids[0] == "b"

def t25_fetch_parents_dedup():
    from retriever import fetch_parents
    chunks = [
        {"id":"1","source":"a.md","parent_text":"same parent","child_text":"c1"},
        {"id":"2","source":"a.md","parent_text":"same parent","child_text":"c2"},
        {"id":"3","source":"b.md","parent_text":"other parent","child_text":"c3"},
    ]
    result = fetch_parents(chunks)
    assert len(result) == 2  # dedup same parent

test("T24 RRF ranking", t24_rrf_merge)
test("T25 fetch_parents dedup", t25_fetch_parents_dedup)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T26 rag_search / rag_retrieve (empty DB) ===")

def t26_rag_search_empty():
    from rag_search import rag_search
    result = rag_search.invoke({"query": "machine learning"})
    assert isinstance(result, str)
    assert "[low_quality]" in result or "[error]" in result or "SOURCES:" in result

def t27_rag_retrieve_empty():
    from rag_retrieve import rag_retrieve
    result = rag_retrieve.invoke({"query": "test query"})
    assert isinstance(result, str)
    assert "[error]" in result or "[1]" in result

test("T26 rag_search returns str on empty DB", t26_rag_search_empty)
test("T27 rag_retrieve returns str on empty DB", t27_rag_retrieve_empty)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T28 ingest + search end-to-end ===")

def t28_e2e():
    import config
    from ingestor import ingest_file, delete_file
    from retriever import search, fetch_parents

    content = ("Retrieval Augmented Generation (RAG) combines LLMs with external knowledge.\n"
               "It retrieves relevant documents before generating an answer.\n\n") * 8
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.KNOWLEDGE_DIR / "_smoke_e2e.txt"
    path.write_text(content, encoding="utf-8")
    try:
        n = ingest_file(path)
        assert n > 0, "No chunks stored"
        chunks = search(["RAG retrieval augmented generation"], top_k=5, top_fused=3)
        parents = fetch_parents(chunks)
        assert len(parents) > 0, "No results found after ingest"
        assert any("RAG" in row["parent_text"] or "retrieval" in row["parent_text"].lower() for row in parents)
        delete_file(path)
    finally:
        path.unlink(missing_ok=True)

test("T28 ingest txt + search e2e", t28_e2e)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== T29 tool return contract ===")

def t29_tool_returns_str():
    from rag_search import rag_search
    from rag_retrieve import rag_retrieve
    r1 = rag_search.invoke({"query": "test"})
    r2 = rag_retrieve.invoke({"query": "test"})
    assert isinstance(r1, str), f"rag_search returned {type(r1)}"
    assert isinstance(r2, str), f"rag_retrieve returned {type(r2)}"
    assert r1 is not None and r2 is not None

test("T29 tools return str not None/dict/list", t29_tool_returns_str)

# ─────────────────────────────────────────────────────────────────────────────
print()
passed = sum(1 for r, _ in results if r == PASS)
failed = sum(1 for r, _ in results if r == FAIL)
total  = len(results)
print(f"{'─'*46}")
print(f"  Result: {passed}/{total} passed  {f'({failed} failed)' if failed else '🎉 all passed'}")
print()
if failed:
    print("Failed tests:")
    for r, name in results:
        if r == FAIL:
            print(f"  {FAIL} {name}")
    sys.exit(1)
