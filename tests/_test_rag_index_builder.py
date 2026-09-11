"""Caller-assisted canonical rag_index.json tests; no live model required."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest import mock

from _runner import Runner

r = Runner("rag index builder")


def _knowledge_root() -> str:
    import config
    config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return str(config.KNOWLEDGE_DIR)


def _doc(root: Path, name: str, tags: list[str] | None = None) -> Path:
    path = root / name
    if tags:
        block = "\n".join(f"  - {tag}" for tag in tags)
        path.write_text(f"---\ntags:\n{block}\n---\nbody\n", encoding="utf-8")
    else:
        path.write_text("body\n", encoding="utf-8")
    return path


def t_prepare_returns_bounded_deterministic_context():
    import rag_index_builder as builder

    with tempfile.TemporaryDirectory(dir=_knowledge_root()) as td:
        root = Path(td)
        a = _doc(root, "alpha-finance.md", ["Finance", "Macro"])
        b = _doc(root, "beta-risk.pdf")
        with mock.patch.object(builder.file_registry, "reload"), \
             mock.patch.object(builder.file_registry, "registry_fingerprint", return_value="a" * 64), \
             mock.patch.object(builder.file_registry, "all_registered", return_value=[str(b), str(a)]):
            context = builder.prepare_index_context()

    assert context["status"] == "prepared"
    assert context["expected_fingerprint"] == "a" * 64
    assert context["total_files"] == 2
    assert context["sampled_files"] == 2
    assert context["filenames"] == ["alpha-finance.md", "beta-risk.pdf"]
    assert context["top_tags"] == {"finance": 1, "macro": 1}
    assert context["source_types"] == {"md": 1, "pdf": 1}
    assert context["llm_used_by_backend"] is False


def t_commit_uses_caller_topics_and_backend_truth():
    import rag_index_builder as builder

    with tempfile.TemporaryDirectory(dir=_knowledge_root()) as td:
        root = Path(td)
        a = _doc(root, "alpha.md", ["Finance", "Macro"])
        b = _doc(root, "beta.pdf")
        target = root / "rag_index.json"
        with mock.patch.object(builder.file_registry, "reload"), \
             mock.patch.object(builder.file_registry, "registry_fingerprint", return_value="b" * 64), \
             mock.patch.object(builder.file_registry, "all_registered", return_value=[str(a), str(b)]):
            result = builder.commit_index(
                ["การลงทุน", "เศรษฐศาสตร์", "การลงทุน"],
                "b" * 64,
                target,
            )

        disk = json.loads(target.read_text(encoding="utf-8"))
        assert disk == result
        assert result["topics"] == ["การลงทุน", "เศรษฐศาสตร์"]
        assert result["topics_line"] == "การลงทุน | เศรษฐศาสตร์"
        assert result["tags"] == {"finance": 1, "macro": 1}
        assert result["source_types"] == {"md": 1, "pdf": 1}
        assert result["total"] == 2
        assert result["registry_fingerprint"] == "b" * 64
        assert result["topic_method"] == "caller_llm"
        assert result["orientation_policy_id"] == "caller-llm-topics:v1"
        assert result["caller_llm_used"] is True
        assert result["llm_used_by_backend"] is False


def t_topic_validation_is_bounded_and_deduped():
    import rag_index_builder as builder

    assert builder._normalize_topics(" Finance | Macro | finance ") == ["Finance", "Macro"]
    try:
        builder._normalize_topics(["x" * 97])
    except ValueError as exc:
        assert "<= 96" in str(exc)
    else:
        raise AssertionError("overlong topic must be rejected")

    try:
        builder._normalize_topics([f"topic-{i}" for i in range(31)])
    except ValueError as exc:
        assert "at most 30" in str(exc)
    else:
        raise AssertionError("more than 30 topics must be rejected")


def t_stale_prepare_fingerprint_preserves_previous_index():
    import rag_index_builder as builder

    with tempfile.TemporaryDirectory(dir=_knowledge_root()) as td:
        root = Path(td)
        doc = _doc(root, "alpha.md")
        target = root / "rag_index.json"
        sentinel = '{"topics_line":"known-good"}\n'
        target.write_text(sentinel, encoding="utf-8")
        with mock.patch.object(builder.file_registry, "reload"), \
             mock.patch.object(builder.file_registry, "registry_fingerprint", return_value="d" * 64), \
             mock.patch.object(builder.file_registry, "all_registered", return_value=[str(doc)]):
            try:
                builder.commit_index(["การเงิน"], "c" * 64, target)
            except builder.IndexCommitConflict:
                pass
            else:
                raise AssertionError("expected fingerprint conflict")
        assert target.read_text(encoding="utf-8") == sentinel


def t_registry_race_before_replace_preserves_previous_index():
    import rag_index_builder as builder

    with tempfile.TemporaryDirectory(dir=_knowledge_root()) as td:
        root = Path(td)
        doc = _doc(root, "alpha.md")
        target = root / "rag_index.json"
        sentinel = '{"topics_line":"known-good"}\n'
        target.write_text(sentinel, encoding="utf-8")
        fingerprints = iter(["e" * 64, "e" * 64, "f" * 64])
        with mock.patch.object(builder.file_registry, "reload"), \
             mock.patch.object(builder.file_registry, "registry_fingerprint", side_effect=lambda: next(fingerprints)), \
             mock.patch.object(builder.file_registry, "all_registered", return_value=[str(doc)]):
            try:
                builder.commit_index(["การเงิน"], "e" * 64, target)
            except builder.IndexCommitConflict:
                pass
            else:
                raise AssertionError("expected registry-race conflict")
        assert target.read_text(encoding="utf-8") == sentinel
        assert not list(root.glob(".rag_index.json.*.tmp"))


def t_automatic_backend_builder_is_disabled():
    import rag_index_builder as builder

    try:
        builder.build_index()
    except RuntimeError as exc:
        assert "caller LLM" in str(exc)
    else:
        raise AssertionError("backend must not auto-generate caller topics")


def t_source_has_no_llm_dependency():
    import rag_index_builder as builder

    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "import llm_client" not in source
    assert "from llm_client" not in source
    import config
    assert builder.RAG_INDEX_PATH == config.RAG_INDEX_PATH
    assert builder.RAG_INDEX_PATH.parent == config.STATE_DIR


r.test("prepare returns bounded deterministic caller context", t_prepare_returns_bounded_deterministic_context)
r.test("commit combines caller topics with backend truth", t_commit_uses_caller_topics_and_backend_truth)
r.test("topic validation is bounded and deduped", t_topic_validation_is_bounded_and_deduped)
r.test("stale prepare preserves previous index", t_stale_prepare_fingerprint_preserves_previous_index)
r.test("registry race preserves previous index", t_registry_race_before_replace_preserves_previous_index)
r.test("automatic backend builder is disabled", t_automatic_backend_builder_is_disabled)
r.test("builder source has no LLM dependency", t_source_has_no_llm_dependency)

if __name__ == "__main__":
    r.exit()
