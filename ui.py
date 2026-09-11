"""ui.py — retained terminal presentation helpers for MCP-RagMax."""
from __future__ import annotations
import sys
import threading
import time
import itertools
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from rich.table import Table

_theme = Theme({
    "markdown.h1":         "bold white",
    "markdown.h2":         "bold white",
    "markdown.h3":         "bold white",
    "markdown.code":       "bold cyan",
    "markdown.code_block": "dim",
})
_console = Console(theme=_theme, highlight=False)

# ── ANSI colors ──────────────────────────────────────────────────────────────
R      = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

C_USER   = "\033[38;5;75m"    # blue  — You
C_AGENT  = "\033[38;5;86m"    # green — RAG
C_TOOL   = "\033[38;5;208m"   # orange — tool / step
C_META   = "\033[38;5;244m"   # gray  — metadata
C_ARROW  = "\033[38;5;240m"   # dark gray
C_DIM    = "\033[38;5;237m"   # very dark — divider
C_HEADER = "\033[38;5;252m"   # near-white
C_ORANGE = "\033[38;5;208m"   # prompt marker
C_REF    = "\033[38;5;69m"    # blue-ish — references
C_WARN   = "\033[38;5;214m"   # amber — low quality / retry
C_GREEN  = "\033[38;5;82m"    # green — ok status

WIDTH = 62

console = _console   # alias for build progress


# ── Spinner ──────────────────────────────────────────────────────────────────
class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Thinking…"):
        self.label   = label
        self.sub     = ""
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._drawn  = False

    def _clear(self):
        sys.stdout.write("\r\033[2K")
        if self._drawn:
            sys.stdout.write("\033[F\033[2K")
        sys.stdout.flush()

    def _draw(self, frame: str):
        self._clear()
        sys.stdout.write(f"   {C_META}{frame} {self.label}{R}")
        if self.sub:
            sys.stdout.write(f"\n      {C_ARROW}{DIM}⎿  {self.sub[:60]}{R}")
            self._drawn = True
        else:
            self._drawn = False
        sys.stdout.flush()

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            self._draw(frame)
            time.sleep(0.08)
        self._clear()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        # timeout so a stdout write wedged by terminal backpressure can't block
        # the main thread indefinitely (thread is daemon + _stop is set, so it
        # terminates on its own once the write drains). Parity with V2 B2 fix.
        self._thread.join(timeout=2.0)

    def update(self, label: str):
        self.label = label

    def update_sub(self, msg: str):
        self.sub = msg or ""


def spinner(label: str = "Searching…") -> Spinner:
    return Spinner(label)


# ── Header ───────────────────────────────────────────────────────────────────
def print_header(model: str = ""):
    print()
    print(f" {BOLD}{C_HEADER}Endeavor{R}  "
          f"{C_META}{model}{R}  "
          f"{C_GREEN}● ready{R}")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print(f"   {C_META}พิมพ์ {C_ORANGE}'mode'{C_META} เพื่อเปิดเมนู  {C_ORANGE}'exit'{C_META} เพื่อออก{R}")
    print()


# ── Mode menu ────────────────────────────────────────────────────────────────
def print_mode_menu():
    print()
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print(f"   {C_ORANGE}[2]{R} Build Knowledge")
    print(f"   {C_ORANGE}[q]{R} Quit")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print()


# ── Divider ──────────────────────────────────────────────────────────────────
def print_divider():
    print(f"\n{C_DIM}{'╌' * (WIDTH // 2)}{R}\n")


# ── User prompt ──────────────────────────────────────────────────────────────
def prompt_user() -> str:
    try:
        indicator = f" {C_ORANGE}{BOLD}>{R}  "
        text = input(indicator).strip()
        if text:
            # erase the " > query" line, reprint as "You  query"
            sys.stdout.write("\033[F\033[2K")
            sys.stdout.flush()
        return text
    except (EOFError, KeyboardInterrupt):
        return "exit"


# ── Tool step ────────────────────────────────────────────────────────────────
def print_rag_step(label: str, detail: str = ""):
    line = f"   {C_TOOL}⎿ {label}{R}"
    if detail:
        line += f"  {C_META}{DIM}{detail[:60]}{R}"
    print(f"\n{line}")


def print_retry(round_num: int):
    print(f"\n   {C_WARN}⎿ Round {round_num} — rephrasing…{R}")


# ── Agent answer ─────────────────────────────────────────────────────────────


def _print_sources(sources_line: str,
                   scores: dict[str, tuple[str, float]] | None = None):
    if not sources_line.startswith("SOURCES:"):
        return
    paths = [p.strip() for p in sources_line[8:].split("|") if p.strip()]
    if not paths:
        return
    print(f"\n   {C_META}📎 แหล่งอ้างอิง{R}")
    for i, p in enumerate(paths, 1):
        name = Path(p).name
        link = f"file://{p}"
        score_str = ""
        if scores and name in scores:
            retriever, score = scores[name]
            score_str = f"  {C_META}{DIM}{retriever}={score:.3f}{R}"
        sys.stdout.write(
            f"   {C_ARROW}{DIM} {i}. "
            f"\033]8;;{link}\033\\{name}\033]8;;\033\\{R}"
            f"{score_str}\n"
        )
    sys.stdout.flush()


def print_answer(answer: str, sources_line: str | None = None,
                 scores: dict[str, float] | None = None):
    print()
    sys.stdout.write(f" {C_AGENT}{BOLD}Endeavor{R}\n")
    sys.stdout.flush()
    _console.print(Markdown(answer), end="\n")
    if sources_line:
        _print_sources(sources_line, scores)


def extract_sources_line(rag_result: str) -> str | None:
    for line in rag_result.splitlines():
        if line.startswith("SOURCES:"):
            return line
    return None


def extract_scores(rag_result: str) -> dict[str, tuple[str, float]]:
    """Parse '[N] (source: file.md | dense=0.847)' → {filename: (retriever, score)}."""
    import re
    scores = {}
    for line in rag_result.splitlines():
        m = re.match(r"\[\d+\] \(source: (.+?) \| (dense|bm25)=([\d.]+)", line)
        if m:
            scores[m.group(1)] = (m.group(2), float(m.group(3)))
    return scores


# ── Build Knowledge UI ───────────────────────────────────────────────────────
def print_build_header(data_dir: str, file_count: int):
    print()
    print(f" {BOLD}{C_HEADER}Build Knowledge{R}")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print(f"   {C_META}folder: {C_ARROW}{data_dir}{R}")
    print(f"   {C_META}found:  {C_HEADER}{file_count} file(s){R}")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print()


def print_health_report(issues: list[str], ghost_count: int = 0):
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    if not issues and ghost_count == 0:
        print(f"   {C_GREEN}✓ self-check: no anomalies{R}")
    else:
        print(f"   {C_WARN}⚠ self-check found anomalies:{R}")
        for issue in issues:
            print(f"     {C_WARN}- {issue}{R}")
        if ghost_count:
            print(f"     {C_META}- {ghost_count} registered file(s) have zero chunks (dedup ghosts, informational){R}")
    print()


class BuildProgress:
    def __init__(self):
        self._rows:      list[tuple] = []
        self._processed = 0
        self._skipped   = 0
        self._failed    = 0
        self._chunks    = 0

    def add_row(self, filename: str, status: str, chunks: int = 0):
        if status == "skip":
            self._skipped += 1
            self._rows.append((filename, "skip", 0))
        elif status == "error":
            self._failed += 1
            self._rows.append((filename, "error", 0))
        else:
            self._processed += 1
            self._chunks += chunks
            self._rows.append((filename, status, chunks))
        self._render_last()

    def _render_last(self):
        filename, status, chunks = self._rows[-1]
        fname = filename[:30].ljust(32)
        if status == "skip":
            tag   = f"{C_META}{DIM}↷ skip{R}"
            extra = f"{C_META}{DIM}(unchanged){R}"
        elif status == "error":
            tag   = f"\033[31m✗ error{R}"
            extra = f"{C_META}(see log){R}"
        elif status == "new":
            tag   = f"{C_GREEN}✓ new{R}"
            extra = f"{C_META}→ {chunks:3d} chunks{R}"
        else:
            tag   = f"{C_WARN}✓ changed{R}"
            extra = f"{C_META}→ {chunks:3d} chunks{R}"
        print(f"   {C_ARROW}{fname}{R}  {tag}  {extra}")

    def print_summary(self):
        print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
        failed_part = f"  \033[31m{self._failed} failed{R}" if self._failed else ""
        print(f"   {BOLD}Done:{R} "
              f"{C_GREEN}{self._processed} processed{R}  "
              f"{C_META}{self._skipped} skipped{R}  "
              f"{C_HEADER}{self._chunks} chunks added{R}"
              f"{failed_part}")
        print()
