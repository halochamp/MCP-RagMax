# Security

MCP-RagMax is intended for trusted local use. The MCP server communicates over stdio and the optional HTML UI binds only to `127.0.0.1`.

Source-file tools can read only files registered from the configured `workspace/knowledge/` tree. The MCP surface exposes no shell, Python execution, arbitrary-path read, credential store, or general memory-write capability. Derived Chroma/BM25/registry/job/orientation state belongs under ignored `workspace/.rag_state/`.

Do not expose the UI or MCP protocol to untrusted networks without a separate authentication/authorization design. Do not include real credentials, private documents, runtime indexes, logs, or personal paths in reports or commits.
