"""``_embed`` in the three RAG tools is serialized (review on #17).

Every thread that searches (concurrent requests, and the concurrent tool calls of one
chat round) encodes on the module's one ``SentenceTransformer``, whose HF fast tokenizer
is not safe to share across threads. The fake-model tests need nothing downloaded; the last
test runs the real model, and is skipped when it isn't in the local cache.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sme_chatbot.tools import cdo_rag, docsvc_rag, manuals_rag

RAG_MODULES = [cdo_rag, manuals_rag, docsvc_rag]


class _OverlapRecordingModel:
    """Fake embedder that records how many ``encode`` calls ever ran at once."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def encode(self, texts):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)  # wide window for another thread to enter
        with self._lock:
            self.active -= 1
        return np.array([[0.5, 0.5] for _ in texts])


@pytest.mark.parametrize("module", RAG_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_concurrent_embeds_never_overlap(module, monkeypatch):
    model = _OverlapRecordingModel()
    monkeypatch.setattr(module, "_model", model)

    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(module._embed, [f"query {i}" for i in range(8)]))

    assert model.peak == 1
    assert vectors == [[0.5, 0.5]] * 8


@pytest.fixture(scope="module")
def real_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(docsvc_rag.EMBEDDING_MODEL, local_files_only=True)
    except Exception as exc:  # broad on purpose: not cached, or no backend
        pytest.skip(f"embedding model not in the local cache: {exc}")


def test_concurrent_searches_encode_with_the_real_model(real_model, monkeypatch):
    """Concurrent embeds on the real model all succeed, from the cold tokenizer state on."""
    monkeypatch.setattr(docsvc_rag, "_model", real_model)
    # Back to "no truncation or padding set", so the first calls take the tokenizer's write path.
    real_model.tokenizer._tokenizer.no_truncation()
    real_model.tokenizer._tokenizer.no_padding()
    queries = ["pressione aria compressa", "how to replace the capping head clutch " * 10] * 50

    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(docsvc_rag._embed, queries))

    assert len(vectors) == len(queries)
    assert {len(v) for v in vectors} == {384}
