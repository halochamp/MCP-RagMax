#!/usr/bin/env python3
"""Detached persistent build worker for MCP-RagMax."""
from __future__ import annotations

import argparse
from pathlib import Path

import build_jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--kind", choices=("build_kb",), default="build_kb")
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--cancel-file", type=Path, required=True)
    args = parser.parse_args()
    return build_jobs.run_persistent_job(
        args.job_id,
        args.status_file,
        args.cancel_file,
        args.kind,
    )


if __name__ == "__main__":
    raise SystemExit(main())
