# embedding_infer.py
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

# --- Настройка модели ---
# "intfloat/e5-base-v2", "BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

KNOWN_PIPELINES = [
    "text-generation",
    "text-classification",
    "text-to-image",
    "reinforcement-learning",
    "automatic-speech-recognition",
    "token-classification",
    "image-classification",
    "fill-mask",
    "feature-extraction",
    "question-answering",
    "sentence-similarity",

    "summarization",
    "audio-classification",
    "translation",
    "object-detection"
]

class PipelineTagInferencer:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"🧠 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.pipeline_embeddings = self.model.encode(KNOWN_PIPELINES, normalize_embeddings=True)

    def make_text_repr(self, row: Dict[str, Any]) -> str:
        """
        Собирает текстовое описание модели на основе тегов, имени и конфигурации.
        """
        parts: List[str] = []
        if row.get("model_id"):
            parts.append(row["model_id"])
        if row.get("tags"):
            parts.extend([t for t in row["tags"] if isinstance(t, str)])
        if row.get("config") and isinstance(row["config"], dict):
            for k, v in row["config"].items():
                if isinstance(v, str):
                    parts.append(v)
        return " ".join(parts)

    def infer_pipeline_tag(self, tags: Optional[List[str]], model_id: str, other_fields: Dict[str, Any]) -> Optional[str]:
        """
        Возвращает наиболее вероятный pipeline_tag или None, если уверенность низкая.
        """
        text_repr = self.make_text_repr({"model_id": model_id, "tags": tags, "config": other_fields.get("config")})
        if not text_repr.strip():
            return None

        text_emb = self.model.encode(text_repr, normalize_embeddings=True)
        sim = cosine_similarity([text_emb], self.pipeline_embeddings)[0]
        best_idx = int(np.argmax(sim))
        best_score = float(sim[best_idx])
        predicted_tag = KNOWN_PIPELINES[best_idx]

        # можно логировать
        if best_score >= 0.45:  # эмпирический порог уверенности
            return predicted_tag
        return None
