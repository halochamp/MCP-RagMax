"""_test_spinner.py — check MCP-RagMax for the V2 B2 spinner bug class.

V2 B2 had two mechanisms:
  H4 (orphan race): a per-tool spinner (start/stop in a callback handler) got
    orphaned under concurrent ToolNode dispatch.
  H1 (blocking-write hang): Spinner.__exit__ join() with no timeout blocks the
    main thread forever if a stdout write is wedged by terminal backpressure.

RAG uses ONE spinner per turn (`with ui.Spinner(...)`), updated from the main
stream loop + _progress.report (fired by tools, possibly concurrent). So:
  - H4 should be STRUCTURALLY ABSENT (no per-tool spinner creation).
  - H1 is LATENT: ui.Spinner.__exit__ uses bare join() (no timeout).

This test confirms both, without needing Ollama.
"""
from __future__ import annotations
import os, sys, time, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spinner_threads():
    out = []
    for t in threading.enumerate():
        qn = getattr(getattr(t, "_target", None), "__qualname__", "") or ""
        if "Spinner._spin" in qn and t.is_alive():
            out.append(t)
    return out


class _MuteOut:
    def write(self, s): return len(s)
    def flush(self): pass


def test_concurrent_progress_no_orphan() -> bool:
    """H4-equivalent: one spinner per turn + many concurrent _progress.report
    calls (like concurrent tools). Must leave zero orphan spinner threads."""
    import ui, _progress

    real = sys.stdout
    sys.stdout = _MuteOut()
    try:
        baseline = len(_spinner_threads())
        with ui.Spinner("🧠 AI กำลังคิด…") as sp:
            _progress.set_callback(lambda lbl, sub: (sp.update(lbl), sp.update_sub(sub)))

            def hammer(n):
                for i in range(50):
                    _progress.report(f"label-{n}", f"sub-{n}-{i}")
                    sp.update(f"main-{n}")  # main-loop style update too
                    time.sleep(0.001)

            ts = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
            for t in ts: t.start()
            for t in ts: t.join()
            _progress.set_callback(None)
        time.sleep(0.3)
        leftover = len(_spinner_threads())
    finally:
        sys.stdout = real

    ok = leftover <= baseline
    print(f"  [H4] concurrent progress → leftover spinner threads={leftover} "
          f"(baseline={baseline})  {'✅ PASS (not vulnerable)' if ok else '❌ FAIL (orphans!)'}")
    return ok


def test_join_has_timeout() -> bool:
    """H1: under a blocking stdout write, __exit__ must not block the main thread
    indefinitely. Measures how long __exit__ takes with a slow stdout."""
    import importlib, ui
    importlib.reload(ui)
    from ui import Spinner

    class SlowOut:
        def __init__(self, d): self.d = d
        def write(self, s): time.sleep(self.d); return len(s)
        def flush(self): time.sleep(self.d)

    real = sys.stdout
    sys.stdout = SlowOut(1.0)   # each write blocks 1s
    try:
        sp = Spinner("x")
        sp.__enter__()
        time.sleep(0.15)        # thread enters a blocking _draw()
        t0 = time.time()
        sp.__exit__(None, None, None)
        exit_dt = time.time() - t0
    finally:
        sys.stdout = real

    # PASS if __exit__ is bounded (~join timeout). A bare join() blocks until the
    # wedged write drains (here ≥ ~2-3s of serial 1s writes) → effectively a hang.
    ok = exit_dt < 2.5
    print(f"  [H1] blocking write → __exit__ took {exit_dt:.2f}s  "
          f"{'✅ PASS (bounded)' if ok else '❌ FAIL (join has no timeout → main thread blocks)'}")
    return ok


if __name__ == "__main__":
    print("RAG spinner bug-class check:")
    r1 = test_concurrent_progress_no_orphan()
    r2 = test_join_has_timeout()
    print(f"\n{sum([r1, r2])}/2 passed")
    sys.exit(0 if (r1 and r2) else 1)
