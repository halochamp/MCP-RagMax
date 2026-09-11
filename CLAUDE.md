# MCP-RagMax public project rules

- This public repository is self-contained; never depend on a private parent repository or sibling agent.
- Read `AGENT.md` and `AGENT_PROCEDURE.md` before substantial work.
- MCP-RagMax is an LLM-free RAG backend: search, expansion, ranking, build decisions, validation, and writes remain deterministic.
- Preserve the nine-tool stdio MCP surface documented in README; do not add shell/Python/arbitrary-path tools.
- Source reads are confined to the configured `workspace/knowledge/` root. Derived state belongs under `workspace/.rag_state/` and is never committed.
- The HTML UI binds only to `127.0.0.1`; do not expose it to LAN/public networks without a new auth/policy design.
- `rag_rebuild_index` may accept caller-LLM topic labels only through the guarded prepare/commit fingerprint protocol; backend metadata remains deterministic truth.
- Preserve persistent build jobs, cooperative cancellation, Chroma/BM25 consistency, registry atomicity, and pipeline fingerprint checks.
- Add/update deterministic regression coverage for changes. Never commit private documents, indexes, caches, logs, credentials, screenshots, or personal absolute paths.
