from __future__ import annotations

from typing import List, Tuple
from app.models import PolicyChunk


class CrossEncoderReranker:
    """
    Reranks retrieved policy chunks against the user query using
    cross-encoder/ms-marco-MiniLM-L-6-v2, with fallback to lexical score combination.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                # Fallback if model cannot be downloaded/loaded
                self._model = False
        return self._model

    def rerank(self, query: str, chunks: List[PolicyChunk], top_k: int = 5) -> List[PolicyChunk]:
        """Reranks candidate chunks based on cross-encoder relevance scores."""
        if not chunks:
            return []

        model = self._get_model()
        if model and model is not False:
            try:
                pairs = [(query, f"{c.heading}\n{c.content}") for c in chunks]
                scores = model.predict(pairs)
                for chunk, score in zip(chunks, scores):
                    chunk.rerank_score = float(score)
                
                sorted_chunks = sorted(chunks, key=lambda x: x.rerank_score if x.rerank_score is not None else 0.0, reverse=True)
                return sorted_chunks[:top_k]
            except Exception:
                pass

        # Lexical score fallback if CrossEncoder is offline
        for chunk in chunks:
            text = f"{chunk.heading} {chunk.content}".lower()
            query_terms = [t for t in query.lower().split() if len(t) > 2]
            match_count = sum(1 for t in query_terms if t in text)
            chunk.rerank_score = float(match_count)

        sorted_chunks = sorted(chunks, key=lambda x: x.rerank_score or 0.0, reverse=True)
        return sorted_chunks[:top_k]
