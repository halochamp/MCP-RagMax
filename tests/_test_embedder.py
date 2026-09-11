"""_test_embedder.py — embedder.encode / encode_one"""
from _runner import Runner

r = Runner("embedder")


def t05_encode_one():
    from embedder import encode_one
    v = encode_one("machine learning")
    assert len(v) == 384
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 0.01  # normalized


def t06_encode_batch():
    from embedder import encode
    vs = encode(["hello", "world", "สวัสดี"])
    assert len(vs) == 3
    assert all(len(v) == 384 for v in vs)


def t07_empty_encode():
    from embedder import encode
    assert encode([]) == []


def t45_cached_model_load_is_local_first():
    import embedder
    original_model = embedder._model
    original_cls = embedder.SentenceTransformer
    calls = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

    try:
        embedder._model = None
        embedder.SentenceTransformer = FakeModel
        embedder._get_model()
        assert calls == [(embedder.MODEL_NAME, {"local_files_only": True})], calls
    finally:
        embedder._model = original_model
        embedder.SentenceTransformer = original_cls


r.test("T05 encode_one normalized", t05_encode_one)
r.test("T06 encode batch", t06_encode_batch)
r.test("T07 empty input", t07_empty_encode)
r.test("T45 cached embedder loads local-first", t45_cached_model_load_is_local_first)

if __name__ == "__main__":
    r.exit()
