from __future__ import annotations

import config
import main


def test_public_paths_are_self_contained():
    assert config.KNOWLEDGE_DIR == config.WORKSPACE_DIR / "knowledge"
    assert config.STATE_DIR == config.WORKSPACE_DIR / ".rag_state"
    assert config.RAG_INDEX_PATH.parent == config.STATE_DIR


def test_ui_is_loopback_only():
    assert config.UI_HOST == "127.0.0.1"
    assert main is not None
