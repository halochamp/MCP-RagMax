# MCP-RagMax — Agent Overview

MCP-RagMax is a standalone deterministic local RAG backend for Thai/English documents. It supersedes the older ENDEAVOR_RAG_LITE architecture in this repository.

Core flow:

```text
MCP client ─stdio─> mcp_server.py ─┐
Browser ─127.0.0.1:8770─> web_ui.py ├─> MiniLM + BM25 + RRF + registry
                                   └─> workspace/.rag_state/
Documents: workspace/knowledge/ (read-only source scope)
```

The MCP catalog is: `rag_retrieve`, `rag_list`, `rag_search_files`, `rag_read_file`, `build_kb`, `build_status`, `cancel_build`, `rag_rebuild_index`, `rag_health`.

Hard invariants: no backend LLM, no arbitrary filesystem read, no network exposure beyond loopback UI, and no generated state in Git. Run deterministic tests before completion; live client integration is separate.
