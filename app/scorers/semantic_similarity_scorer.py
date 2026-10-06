import numpy as np

from app.datasets.schemas import ScoreResult, TestCase
from app.scorers.base_scorer import BaseScorer


class SemanticSimilarityScorer(BaseScorer):
    """Cosine similarity between embeddings of the output and the expected answer.
    Requires: pip install sentence-transformers
    """

    name = "semantic_similarity"
    needs_expected = True

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.75):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, case: TestCase, output: str) -> ScoreResult:
        emb = self._get_model().encode([output, case.expected or ""], normalize_embeddings=True)
        sim = float(np.dot(emb[0], emb[1]))
        return self._result(sim, f"cosine={sim:.3f}", self.threshold)