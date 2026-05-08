"""
Semantic search over FAISS index of SHL assessments.
"""

import json
import numpy as np
import faiss
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_index = None
_meta = None


def _load():
    global _model, _index, _meta
    if _model is None:
        _model = TextEmbedding(MODEL_NAME)
        _index = faiss.read_index("faiss.index")
        with open("index_meta.json") as f:
            _meta = json.load(f)


def search(query: str, top_k: int = 10, test_type_filter: list[str] | None = None) -> list[dict]:
    """Return top_k assessments matching query. Optionally filter by test_type codes."""
    _load()

    vec = np.array(list(_model.embed([query])), dtype="float32")
    # Fetch extra results to allow for post-filtering
    fetch_k = min(top_k * 5, len(_meta)) if test_type_filter else top_k
    scores, indices = _index.search(vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = _meta[idx]
        if test_type_filter:
            if not any(t in item["test_types"] for t in test_type_filter):
                continue
        results.append({**item, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def get_by_name(name: str) -> dict | None:
    """Exact or fuzzy lookup by assessment name."""
    _load()
    name_lower = name.lower()
    for item in _meta:
        if item["name"].lower() == name_lower:
            return item
    # Fuzzy: find best substring match
    best = None
    best_score = 0
    for item in _meta:
        n = item["name"].lower()
        if name_lower in n or n in name_lower:
            score = len(set(name_lower.split()) & set(n.split()))
            if score > best_score:
                best_score = score
                best = item
    return best


def get_all_urls() -> set[str]:
    _load()
    return {item["url"] for item in _meta}
