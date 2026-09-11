
"""Run the retained regression modules in isolated temporary state."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEGACY_TESTS = sorted(Path(__file__).parent.glob("_test_*.py"))


@pytest.mark.parametrize("test_file", LEGACY_TESTS, ids=lambda p: p.name)
def test_legacy_module(test_file: Path) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="ragmax-pytest-"))
    env = os.environ.copy()
    env.update({
        "RAGMAX_WORKSPACE": str(temp_root / "workspace"),
        "RAGMAX_KNOWLEDGE_DIR": str(temp_root / "workspace" / "knowledge"),
        "RAGMAX_STATE_DIR": str(temp_root / "workspace" / ".rag_state"),
        "RAGMAX_FAKE_EMBEDDINGS": "1",
    })
    result = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
