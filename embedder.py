from __future__ import annotations
import logging
import os
import threading
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

from pipeline_config import EMBED_MODEL_NAME

MODEL_NAME = EMBED_MODEL_NAME

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
                except (OSError, ValueError):
                    # First use may legitimately need to populate the cache.
                    _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """Encode texts to dense vectors. Returns list of float lists."""
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    return vecs.tolist()


def encode_one(text: str) -> list[float]:
    return encode([text])[0]
