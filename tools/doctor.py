"""Read-only diagnostics for a standalone MCP-RagMax checkout."""
from __future__ import annotations
import argparse
import importlib.util
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config

REQUIRED = ["chromadb", "fastapi", "mcp", "sentence_transformers", "rank_bm25", "pythainlp", "pypdf"]


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    failures = 0
    def report(label: str, ok: bool, detail: str) -> None:
        nonlocal failures
        print(f"{'OK' if ok else 'FAIL':4} {label}: {detail}")
        failures += int(not ok)
    report("platform", platform.system() == "Darwin" and platform.machine() == "arm64",
           f"{platform.system()} {platform.machine()} (Apple Silicon release target)")
    report("python", sys.version_info[:2] == (3, 11), f"{sys.version.split()[0]} (Python 3.11 required)")
    for module in REQUIRED:
        report(f"package {module}", importlib.util.find_spec(module) is not None, module)
    config.ensure_runtime_dirs()
    report("knowledge directory", config.KNOWLEDGE_DIR.is_dir(), str(config.KNOWLEDGE_DIR))
    report("state directory", config.STATE_DIR.is_dir(), str(config.STATE_DIR))
    report("UI binding", config.UI_HOST == "127.0.0.1", f"{config.UI_HOST}:{config.UI_PORT}")
    print("INFO backend: deterministic; no chat model or cloud LLM is started by doctor")
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
