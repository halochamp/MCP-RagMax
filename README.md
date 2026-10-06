# MCP-RagMax

## Local deterministic RAG backend สำหรับเอกสารไทย/อังกฤษ

MCP-RagMax คือ RAG backend ที่ทำงานบนเครื่องและเปิดให้ agent อื่นเรียกผ่าน MCP โดย **ตัว backend ไม่ใช้ LLM** สำหรับ search, query expansion, reranking, build decisions, metadata validation หรือ file writes

โปรเจกต์นี้เข้ามาแทน architecture เดิมของ `ENDEAVOR_RAG_LITE` ใน repository นี้ โดยคงจุดเด่นของ public release คือ clone แล้วทำงานได้เอง ไม่ต้องพึ่ง private monorepo หรือ agent อื่น

### สิ่งที่ได้

- Hybrid retrieval: multilingual MiniLM dense search + Thai-aware BM25 + Reciprocal Rank Fusion (RRF)
- Incremental KB build พร้อม file registry, exact-hash dedup และ deterministic semantic duplicate rejection
- Persistent background build jobs พร้อม progress, ETA และ cooperative cancellation
- MCP stdio 4 tools สำหรับ retrieval, file access ภายใน KB, build, health และ orientation index
- `rag_index.json` lifecycle แบบ prepare/commit: caller LLM ช่วยเสนอได้เฉพาะ topic labels ส่วน counts/tags/source types/fingerprint เป็น backend truth
- Local HTML RAG console ที่ `127.0.0.1:8770`
- Source confinement: อ่านไฟล์ได้เฉพาะใต้ `workspace/knowledge/`
- Derived state ทั้งหมดอยู่ใต้ `workspace/.rag_state/` และถูก ignore จาก Git

## Architecture

```text
MCP client ──stdio──► mcp_server.py ───────────────┐
                                                    │
Browser ──127.0.0.1:8770──► web_ui.py ────────────┤
                                                    ▼
                                  deterministic RAG core
                           MiniLM + BM25 + RRF + registry
                                                    │
                           workspace/.rag_state/ (derived)
                                                    ▲
                           workspace/knowledge/ (source)
```

Backend ไม่มี chat agent และไม่ start local/cloud LLM เอง

## MCP tools

MCP catalog มี 4 tools:

1. `rag_retrieve` — hybrid retrieval (`chunks`, `files`, `source_first`)
2. `rag_files` — `list` (limit/offset), `search` (query/limit), `read` (filename)
3. `rag_manage` — `build_start`, `build_cancel` (job_id), `index_prepare`, `index_commit` (topics/expected_fingerprint)
4. `rag_status` — `health` หรือ `build` (job_id)

อ่านไฟล์ผ่าน `rag_files(action="read", filename=...)`; prepare/commit ใช้
`rag_manage` และ build polling ใช้ `rag_status` ตาม action/view ข้างต้น.

MCP ไม่ expose shell, Python execution หรือ arbitrary filesystem read

## เริ่มใช้งาน

### 1. ติดตั้ง

```bash
cd ENDEAVOR_RAG_LITE
bash install_library/install.sh
source .venv/bin/activate
```

Installer ติดตั้ง dependency เท่านั้น ไม่ scan files, ไม่ build index และไม่ start model

### 2. ใส่เอกสาร

วางไฟล์ไว้ใต้:

```text
workspace/knowledge/
```

รองรับ `.md`, `.txt`, `.pdf`, `.csv`, `.json`

### 3. ตรวจ environment

```bash
python tools/doctor.py
```

### 4. Build KB

แบบ foreground:

```bash
python tools/build_index.py
```

หรือผ่าน MCP ใช้ `rag_manage(action="build_start")` แล้ว poll ด้วย `rag_status(view="build", job_id=...)`

### 5. เปิด HTML UI

```bash
python main.py
```

แล้วเปิด `http://127.0.0.1:8770`

หรือบน macOS double-click `Start MCP-RagMax UI.command`

UI ใช้สำหรับ search, health, file list, build/progress/cancel และ rag-index prepare/commit; ไม่ใช่ chat UI

## MCP launch

ตัวอย่างการเปิด stdio server:

```bash
source .venv/bin/activate
python mcp_server.py
```

MCP host ควร launch process นี้เป็น local stdio child process

## Retrieval

Search pipeline:

```text
query normalization
  → multilingual MiniLM embedding
  → Thai-aware BM25
  → RRF fusion
  → deterministic unique-parent selection
```

Caller สามารถส่ง query variants ได้สูงสุดตาม schema แต่ต้องเป็น **same intent** ไม่ใช่ subquestions ใหม่ Backend จะ retrieve แต่ละ variant แบบ deterministic เหมือนเดิม

## Build และ cancellation

`rag_manage(action="build_start")` สแกน `workspace/knowledge/` แล้ว process new/changed files เท่านั้น Registry, Chroma และ BM25 มี consistency/rollback guards

- new file: cancellation สามารถหยุดกลาง ingest และ rollback partial writes
- changed file: เมื่อเริ่ม replace old rows แล้ว จะ finish file ปัจจุบันก่อน honoring cancellation เพื่อลดช่วงที่ source หายจาก index
- job state อยู่ใน `workspace/.rag_state/build_jobs/` จึง poll/cancel ต่อได้แม้ MCP stdio process เดิมปิดไปแล้ว

## `rag_index.json` และ caller LLM

MCP-RagMax ไม่เรียก LLM เอง แต่ caller agent อาจใช้ LLM ของตัวเองช่วยสร้าง high-level topics ผ่าน protocol ที่จำกัด:

1. `rag_manage(action="index_prepare")` คืน bounded deterministic snapshot + `expected_fingerprint`
2. caller สร้าง topic labels 1–30 รายการจาก snapshot
3. `rag_manage(action="index_commit", topics=..., expected_fingerprint=...)`
4. backend validate topics, recompute metadata และ recheck fingerprint ก่อน atomic install

ถ้า KB เปลี่ยนระหว่าง prepare/commit จะ fail ด้วย conflict และไม่ทับ index ที่ดีอยู่เดิม

Public port เก็บ orientation file ที่ `workspace/.rag_state/rag_index.json` แทนการสร้าง tracked runtime file ใน repository

## Configuration

| Environment variable | Default | ความหมาย |
|---|---|---|
| `RAGMAX_WORKSPACE` | `workspace/` | workspace root |
| `RAGMAX_KNOWLEDGE_DIR` | `workspace/knowledge/` | source document root |
| `RAGMAX_STATE_DIR` | `workspace/.rag_state/` | derived private state |
| `RAGMAX_UI_PORT` | `8770` | local UI port |
| `RAGMAX_FAKE_EMBEDDINGS` | unset | test-only deterministic embedding path |

`RAGMAX_KNOWLEDGE_DIR` สามารถชี้ไป directory อื่นที่ผู้ใช้เลือกเองได้ แต่ทุก source path ที่ backend เปิดต้อง resolve อยู่ภายใน root นี้; symlink/path traversal ที่หนี root จะถูกปฏิเสธ

## Privacy & security

- UI bind ที่ `127.0.0.1` เท่านั้น
- MCP ใช้ stdio สำหรับ trusted local host
- `rag_files(action="read")` อ่านได้เฉพาะ registered file ใน configured knowledge root
- ไม่มี shell/Python/arbitrary path MCP tools
- user documents และ generated state ไม่ถูก commit
- ไม่มี cloud LLM call ใน backend

หากต้องการ expose ผ่าน LAN/Internet ต้องออกแบบ authentication, authorization, request limits และ privacy policy ใหม่ ไม่ควรเปลี่ยน host flag อย่างเดียว

## Testing

Deterministic suite แบบเดียวกับ private MCP-RagMax:

```bash
python tests/run_all.py
```

และ public packaging regression:

```bash
python -m pytest tests -q
```

Test suite ครอบคลุม chunking, dense/BM25/RRF, registry lifecycle, deterministic dedup, dual-store rollback, cancellation, persistent jobs, 4-tool MCP schema/handshake, caller-assisted rag index และ loopback Web UI

## System requirements

- macOS on Apple Silicon
- Python 3.11
- RAM/disk เพียงพอสำหรับ local embedding model และ Chroma index
- internet ครั้งแรกที่ต้อง download embedding model หากยังไม่มี cache

Retrieval model default คือ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; model weights ไม่ได้ถูกเก็บใน repository

## Repository migration note

ชื่อ GitHub repository เดิมคือ `ENDEAVOR_RAG_LITE` แต่ runtime architecture ปัจจุบันคือ **MCP-RagMax** และแทนที่ Pipe A/B/C แบบเก่าที่เคย expose MCP เพียง `rag_retrieve` tool เดียว
