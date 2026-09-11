"""Compatibility tombstone: MCP-RagMax no longer provides or uses an LLM.

Search is deterministic Dense + BM25 + RRF. Build dedup is exact-hash + cosine,
and rag_index.json topics are derived from tags/filenames. This module remains
only so stale imports fail with an explicit migration message instead of
silently starting a local model server.
"""

_REMOVED = "MCP-RagMax LLM support was removed; use deterministic RAG APIs/MCP tools"


def _removed(*_args, **_kwargs):
    raise RuntimeError(_REMOVED)


chat = _removed
build_llm = _removed
ensure_mlx_server = _removed
MLX_MODEL = "removed"
