"""Central progress callback — rag_search reports steps → spinner updates."""
from __future__ import annotations
from typing import Callable

_callback: Callable[[str, str], None] | None = None


def set_callback(fn: Callable[[str, str], None] | None):
    """fn(label, sub) — set by main.py when spinner is active."""
    global _callback
    _callback = fn


def report(label: str = "", sub: str = ""):
    if _callback:
        _callback(label, sub)
