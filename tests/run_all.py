"""run_all.py — run every _test_*.py in rag_test_suite and report a combined result.

Usage:
    python run_all.py            (from rag_test_suite/ or MCP-RagMax/)
    python rag_test_suite/run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_FILES = sorted(HERE.glob("_test_*.py"))

failed_files = []

for f in TEST_FILES:
    print(f"\n{'='*60}\n{f.name}\n{'='*60}")
    proc = subprocess.run([sys.executable, str(f)], cwd=str(HERE))
    if proc.returncode != 0:
        failed_files.append(f.name)

print(f"\n{'#'*60}")
if failed_files:
    print(f"  FAILED FILES ({len(failed_files)}/{len(TEST_FILES)}): {', '.join(failed_files)}")
    sys.exit(1)
else:
    print(f"  ALL {len(TEST_FILES)} TEST FILES PASSED 🎉")
