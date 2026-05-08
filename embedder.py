"""
Minimal ONNX embedder for all-MiniLM-L6-v2.
No PyTorch. No Rust. Pre-built wheels only (onnxruntime + tokenizers).
"""

import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


class MiniLMEmbedder:
    def __init__(self, model_dir: str = _MODEL_DIR):
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        tok.enable_truncation(max_length=256)
        self._tokenizer = tok
        self._session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        token_embeddings = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })[0]  # (batch, seq_len, 384)

        # Attention-mask-weighted mean pooling
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        embeddings = np.sum(token_embeddings * mask, axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)

        # L2 normalize → compatible with IndexFlatIP cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.clip(norms, 1e-9, None)).astype(np.float32)
