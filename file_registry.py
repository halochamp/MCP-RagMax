from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
import fcntl
from pathlib import Path

from config import REGISTRY_PATH, source_key, source_path

_registry: dict[str, object] = {}
_loaded = False
# (st_mtime_ns, st_size, st_ino) of the registry file as of the last successful
# load. Ingestion runs in a different process from the agent, so a
# load-once-per-process registry silently hides every file added or removed
# after the agent booted. st_ino is what makes this reliable: _save() installs
# the new registry with os.replace(), which always yields a fresh inode even
# when two writes land inside the same mtime granularity.
_fingerprint: tuple[int, int, int] | None = None
_LOCK = threading.RLock()


@contextmanager
def _registry_file_lock():
    """Serialize registry read-modify-write cycles across processes."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = REGISTRY_PATH.with_name(REGISTRY_PATH.name + ".lock")
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _rel(path: str) -> str:
    """Convert a knowledge-root path to a portable relative key."""
    return source_key(path)


def _abs(rel: str) -> str:
    """Reconstruct and validate an absolute path from a stored key."""
    return str(source_path(rel))


def _stat_fingerprint() -> tuple[int, int, int] | None:
    try:
        st = REGISTRY_PATH.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _load(force: bool = False):
    """Load the registry, re-reading it whenever the file on disk changed.

    Cheap in the steady state: one stat() per call, a full read only when
    another process replaced the registry. At ~420 entries the reparse cost is
    negligible next to serving a stale file list until the agent restarts.
    """
    global _registry, _loaded, _fingerprint
    with _LOCK:
        fp = _stat_fingerprint()
        if _loaded and not force and fp == _fingerprint:
            return
        if fp is None:
            _registry = {}
            _fingerprint = None
            _loaded = True
            return
        # A reader can land between the writer's open() and its os.replace().
        # Retry briefly; keep the previous good copy rather than surfacing a
        # torn read to callers.
        for _ in range(3):
            try:
                with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                time.sleep(0.005)
                fp = _stat_fingerprint()
                if fp is None:
                    _registry = {}
                    _fingerprint = None
                    _loaded = True
                    return
                continue
            _registry = data if isinstance(data, dict) else {}
            _fingerprint = fp
            _loaded = True
            return


def reload():
    """Force an unconditional re-read. Used before an index rebuild so the
    build always sees the newest registry even if a same-size, same-mtime,
    same-inode in-place rewrite made the fingerprint look unchanged."""
    _load(force=True)


def registry_fingerprint() -> str:
    """SHA-256 of the current atomic registry file, or empty when unavailable."""
    with _LOCK:
        _load()
        try:
            payload = REGISTRY_PATH.read_bytes()
        except OSError:
            return ""
    return hashlib.sha256(payload).hexdigest()


def _save():
    global _loaded, _fingerprint
    with _LOCK:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(REGISTRY_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_registry, f, ensure_ascii=False, indent=2)
        os.replace(tmp, REGISTRY_PATH)
        # Force the next _load() to re-read rather than adopting a fingerprint
        # here. Another process can replace the registry between our
        # os.replace() and a stat() — we would then adopt THEIR inode while
        # holding OUR dict, so every later _load() would skip the re-read and
        # silently shadow their registry for this process's lifetime. That is
        # exactly the staleness class this module was fixed for. One extra read
        # per registration is noise next to that.
        _loaded = False
        _fingerprint = None


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pipeline_fingerprint() -> str:
    """Fingerprint every setting that changes persisted chunk/vector meaning."""
    from pipeline_config import EMBED_MODEL_NAME, PIPELINE_VERSION, SIMILARITY_REJECT

    h = hashlib.sha256()
    h.update((PIPELINE_VERSION + "\0").encode())
    h.update((Path(__file__).with_name("chunker.py")).read_bytes())
    h.update(EMBED_MODEL_NAME.encode())
    h.update(f"{SIMILARITY_REJECT}".encode())
    return h.hexdigest()


def pipeline_outdated_count() -> int:
    """Count registered files built with a different persisted pipeline policy."""
    with _LOCK:
        _load()
        current = _pipeline_fingerprint()
        return sum(
            1
            for value in _registry.values()
            if not isinstance(value, dict) or value.get("pipeline_fingerprint") != current
        )


def check(path: str) -> str:
    """Return 'new' | 'changed' | 'skip'."""
    _load()
    key = _rel(path)
    current = _hash_file(path)
    stored = _registry.get(key)
    if stored is None:
        return "new"
    if not isinstance(stored, dict):
        return "changed"  # legacy registry has no pipeline identity
    if stored.get("file_hash") != current:
        return "changed"
    if stored.get("pipeline_fingerprint") != _pipeline_fingerprint():
        return "changed"
    return "skip"


def register(path: str):
    with _LOCK:
        with _registry_file_lock():
            _load(force=True)
            key = _rel(path)
            _registry[key] = {
                "file_hash": _hash_file(path),
                "pipeline_fingerprint": _pipeline_fingerprint(),
            }
            _save()


def deregister(path: str):
    with _LOCK:
        with _registry_file_lock():
            _load(force=True)
            key = _rel(path)
            if key in _registry:
                del _registry[key]
                _save()


def all_registered() -> list[str]:
    """Return absolute paths (reconstructed from stored relative keys)."""
    with _LOCK:
        _load()
        return [_abs(k) for k in _registry.keys()]


def ghost_files() -> list[str]:
    """Registered files with zero chunks in the vector store (benign dedup ghosts).

    Read-only — single source of truth for the registry/Chroma comparison, shared
    by main.py's build self-check, _test_index_health.py, and any external caller
    by any trusted local caller that wants the same signal.
    """
    import store
    col = store._get_collection()
    if col.count() == 0:
        return []
    # m["source"] may be an old-format absolute path or current-format _BASE-relative
    # string — only absolute ones need _rel() (which itself expects a real fs path,
    # not a value already relative to _BASE).  Malformed metadata must not make a
    # read-only health check crash the build.
    chroma_sources = set()
    for metadata in col.get(include=["metadatas"])["metadatas"]:
        try:
            raw_source = metadata["source"]
            chroma_sources.add(
                _rel(raw_source) if Path(raw_source).is_absolute() else raw_source
            )
        except (KeyError, TypeError, OSError, RuntimeError, ValueError):
            continue
    registered_rel = {_rel(p) for p in all_registered()}
    return sorted(registered_rel - chroma_sources)
