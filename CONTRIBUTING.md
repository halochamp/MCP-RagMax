# Contributing

Use macOS Apple Silicon with Python 3.11. Install with `bash install_library/install.sh`, then run `python -m pytest tests -q` and `python tests/run_all.py` for deterministic verification.

Keep retrieval/build logic deterministic, source reads confined to `workspace/knowledge/`, state under `workspace/.rag_state/`, and the UI loopback-only. Do not commit user documents, generated indexes, caches, logs, credentials, or machine-specific paths.
