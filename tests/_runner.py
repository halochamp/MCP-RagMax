"""_runner.py — shared path setup + test() runner for rag_test_suite.

Adds MCP-RagMax/ to sys.path (so `import chunker`, `import retriever`, etc.
work regardless of the caller's cwd) and provides a small PASS/FAIL collector.
"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path

_RAG_ROOT = str(Path(__file__).resolve().parent.parent)
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)

PASS = "✅"
FAIL = "❌"


class Runner:
    def __init__(self, title: str):
        self.title = title
        self.results: list[tuple[str, str]] = []

    def test(self, name: str, fn) -> None:
        try:
            fn()
            self.results.append((PASS, name))
            print(f"  {PASS}  {name}")
        except Exception:
            self.results.append((FAIL, name))
            print(f"  {FAIL}  {name}")
            traceback.print_exc()

    def summary(self) -> bool:
        passed = sum(1 for r, _ in self.results if r == PASS)
        failed = sum(1 for r, _ in self.results if r == FAIL)
        total = len(self.results)
        print(f"\n{'─'*46}")
        tag = f"({failed} failed)" if failed else "🎉 all passed"
        print(f"  {self.title}: {passed}/{total} passed  {tag}")
        if failed:
            print("\nFailed:")
            for r, name in self.results:
                if r == FAIL:
                    print(f"  {FAIL} {name}")
        return failed == 0

    def exit(self) -> None:
        sys.exit(0 if self.summary() else 1)
