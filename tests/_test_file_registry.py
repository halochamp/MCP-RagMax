"""_test_file_registry.py — file_registry lifecycle + portability (_rel/_abs roundtrip)"""
import os
import tempfile
from pathlib import Path

from _runner import Runner

r = Runner("file_registry")


def t11_registry():
    import config
    from file_registry import check, register, deregister
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.KNOWLEDGE_DIR / "registry-lifecycle.txt"
    path.write_text("hello", encoding="utf-8")
    assert check(str(path)) == "new"
    register(str(path))
    assert check(str(path)) == "skip"
    path.write_text("changed content", encoding="utf-8")
    assert check(str(path)) == "changed"
    deregister(str(path))
    assert check(str(path)) == "new"
    path.unlink(missing_ok=True)


def t30_rel_abs_roundtrip_inside_knowledge():
    """_rel/_abs roundtrip is confined to the configured knowledge root."""
    import config
    from file_registry import _rel, _abs
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    inside = config.KNOWLEDGE_DIR / "_portability_roundtrip_test.tmp"
    inside.write_text("x", encoding="utf-8")
    try:
        key = _rel(str(inside))
        assert key == inside.name
        assert _abs(key) == str(inside.resolve())
    finally:
        inside.unlink(missing_ok=True)


def t31_rel_rejects_outside_knowledge():
    """Public port rejects source paths outside the configured knowledge root."""
    from file_registry import _rel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        outside = f.name
    try:
        try:
            _rel(outside)
        except ValueError:
            pass
        else:
            raise AssertionError("outside source path was accepted")
    finally:
        os.unlink(outside)

def t46_pipeline_change_invalidates_unchanged_file():
    import file_registry
    import pipeline_config

    saved = (
        file_registry.REGISTRY_PATH,
        file_registry._registry,
        file_registry._loaded,
        file_registry._fingerprint,
        pipeline_config.EMBED_MODEL_NAME,
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_registry.REGISTRY_PATH = root / "registry.json"
            file_registry._registry = {}
            file_registry._loaded = False
            file_registry._fingerprint = None
            import config
            path = config.KNOWLEDGE_DIR / "pipeline-change-doc.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unchanged")
            file_registry.register(str(path))
            assert file_registry.check(str(path)) == "skip"
            pipeline_config.EMBED_MODEL_NAME += "-new-vector-space"
            assert file_registry.check(str(path)) == "changed"
            assert file_registry.pipeline_outdated_count() == 1
    finally:
        (
            file_registry.REGISTRY_PATH,
            file_registry._registry,
            file_registry._loaded,
            file_registry._fingerprint,
            pipeline_config.EMBED_MODEL_NAME,
        ) = saved


r.test("T11 new/skip/changed/deregister", t11_registry)
r.test("T30 _rel/_abs roundtrip (inside knowledge)", t30_rel_abs_roundtrip_inside_knowledge)
r.test("T31 _rel rejects outside knowledge", t31_rel_rejects_outside_knowledge)
r.test("T46 pipeline/model change invalidates file", t46_pipeline_change_invalidates_unchanged_file)

if __name__ == "__main__":
    r.exit()
