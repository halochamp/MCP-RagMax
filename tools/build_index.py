"""Build/sync the configured knowledge index without starting an LLM."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config


def main() -> int:
    config.ensure_runtime_dirs()
    import ingestor
    import kb_operations
    print(f"Knowledge directory: {config.KNOWLEDGE_DIR}")
    print(f"State directory: {config.STATE_DIR}")
    result = ingestor.sync_knowledge_base()
    print(f"files={result['total_found']} elapsed={result['elapsed']:.2f}s ghosts={result['ghost_count']}")
    health = kb_operations.health_snapshot()
    print(f"issues={len(health['issues'])} registered_files={health['registered_files']} pipeline_outdated_files={health['pipeline_outdated_files']} index_status={health['index_status']}")
    return 1 if result['health_issues'] else 0

if __name__ == "__main__":
    raise SystemExit(main())
