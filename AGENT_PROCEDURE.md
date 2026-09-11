# MCP-RagMax — Agent Procedure

1. Read `CLAUDE.md` and `AGENT.md`; inspect Git status and preserve unrelated work.
2. Classify the change: retrieval/ranking, ingestion/store, registry, jobs/cancellation, MCP schema, orientation index, UI, config, packaging, or docs.
3. Preserve the deterministic backend boundary. `rag_rebuild_index` is the only caller-assisted semantic metadata flow and uses prepare/commit fingerprint validation.
4. Keep source paths confined to `config.KNOWLEDGE_DIR`; all Chroma/BM25/registry/jobs/orientation state stays under `config.STATE_DIR`.
5. MCP changes must keep the nine-tool catalog bounded, validate inputs before side effects, cap output, and avoid shell/Python/arbitrary paths.
6. Build changes must preserve rollback/cancellation rules, persistent job state, pipeline-version invalidation, and registry/store consistency.
7. UI changes must keep same-origin mutation checks and loopback-only binding.
8. Run targeted deterministic tests, then `python -m pytest tests -q` and/or `python tests/run_all.py`. Report environment-limited checks separately.
9. Before commit/push, scan for credentials, `/Users/...`, private documents, `workspace/.rag_state`, caches, logs, screenshots, or model/index artifacts.
10. Do not commit or push unless explicitly requested.
