"""Embedding utilities for semantic similarity analysis."""

import numpy as np
from typing import List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from ..config.prompts import KNOWN_ADVERSARIAL_PROMPTS
from .llm_factory import get_embedding_model


class EmbeddingAnalyzer:
    """Handles embedding generation and similarity computations."""

    def __init__(self, embeddings_model: Optional[Any] = None):
        """Initialize the embedding analyzer.

        Args:
            embeddings_model: Optional embeddings model. Defaults to factory embedding model.
        """
        self.embeddings = embeddings_model or get_embedding_model()
        self._adversarial_embeddings = None

    def _get_adversarial_embeddings(self) -> np.ndarray:
        """Get or compute embeddings for known adversarial prompts.

        Returns:
            Array of embeddings for adversarial prompts
        """
        if self._adversarial_embeddings is None:
            # Compute embeddings for all known adversarial prompts
            embeddings_list = self.embeddings.embed_documents(KNOWN_ADVERSARIAL_PROMPTS)
            self._adversarial_embeddings = np.array(embeddings_list)

        return self._adversarial_embeddings

    def compute_similarity(self, prompt: str) -> float:
        """Compute maximum similarity between prompt and known adversarial prompts.

        Args:
            prompt: User prompt to analyze

        Returns:
            Maximum cosine similarity score (0.0 to 1.0)
        """
        try:
            # Get embedding for user prompt
            prompt_embedding = np.array(self.embeddings.embed_query(prompt)).reshape(1, -1)

            # Get adversarial embeddings
            adversarial_embeddings = self._get_adversarial_embeddings()

            # Compute cosine similarities
            similarities = cosine_similarity(prompt_embedding, adversarial_embeddings)

            # Return maximum similarity
            return float(np.max(similarities))

        except Exception as e:
            print(f"Error computing embeddings: {e}")
            return 0.0

    def find_most_similar(self, prompt: str, top_k: int = 3) -> List[tuple[str, float]]:
        """Find the most similar adversarial prompts.

        Args:
            prompt: User prompt to analyze
            top_k: Number of most similar prompts to return

        Returns:
            List of (adversarial_prompt, similarity_score) tuples
        """
        try:
            # Get embedding for user prompt
            prompt_embedding = np.array(self.embeddings.embed_query(prompt)).reshape(1, -1)

            # Get adversarial embeddings
            adversarial_embeddings = self._get_adversarial_embeddings()

            # Compute cosine similarities
            similarities = cosine_similarity(prompt_embedding, adversarial_embeddings)[0]

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Return prompts and scores
            results = []
            for idx in top_indices:
                results.append((
                    KNOWN_ADVERSARIAL_PROMPTS[idx],
                    float(similarities[idx])
                ))

            return results

        except Exception as e:
            print(f"Error finding similar prompts: {e}")
            return []