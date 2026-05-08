"""
Build FAISS vector index from catalog.json.
Run once after scraper.py: python build_index.py
Outputs: faiss.index, index_meta.json
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def build_document(item: dict) -> str:
    types_text = ", ".join(
        item.get("test_types", [])
    )
    job_levels_text = ", ".join(item.get("job_levels", []))
    remote = "remote testing supported" if item.get("remote_testing") else ""
    adaptive = "adaptive/IRT" if item.get("adaptive") else ""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        f"Test types: {types_text}" if types_text else "",
        f"Job levels: {job_levels_text}" if job_levels_text else "",
        remote,
        adaptive,
    ]
    return " | ".join(p for p in parts if p)


def main():
    with open("catalog.json") as f:
        catalog = json.load(f)

    print(f"Building index for {len(catalog)} items...")
    model = SentenceTransformer(MODEL_NAME)

    docs = [build_document(item) for item in catalog]
    embeddings = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (normalized vectors)
    index.add(embeddings)

    faiss.write_index(index, "faiss.index")

    meta = [
        {
            "id": i,
            "name": item["name"],
            "url": item["url"],
            "test_types": item.get("test_types", []),
            "remote_testing": item.get("remote_testing", False),
            "adaptive": item.get("adaptive", False),
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "document": docs[i],
        }
        for i, item in enumerate(catalog)
    ]
    with open("index_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved faiss.index ({dim}d) and index_meta.json")


if __name__ == "__main__":
    main()
