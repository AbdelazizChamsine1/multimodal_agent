import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, Tuple
from colorama import Fore, Style


class SemanticCache:
    """In-memory semantic cache using cosine similarity for question matching."""
    model_instance = None

    def __init__(self, similarity_threshold: float = 0.85, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic cache.

        Args:
            similarity_threshold: Minimum cosine similarity (0-1) for cache hit
            model_name: SentenceTransformer model name (small & fast by default)
        """
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, Tuple[np.ndarray, str]] = {}  # {question: (embedding, answer)}
        if not SemanticCache.model_instance:
            SemanticCache.model_instance = SentenceTransformer(model_name)
        self.model = SemanticCache.model_instance

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def get(self, question: str, context_summary: str = "") -> Optional[str]:
        """Retrieve cached answer for similar question.

        Args:
            question: User's question
            context_summary: Optional context summary for better matching

        Returns:
            Cached answer if similar question found, else None
        """
        if not self.cache:
            return None

        # Create search key with context
        search_key = f"{question}|{context_summary}"
        query_embedding = self._get_embedding(search_key)

        # Find most similar cached question
        best_similarity = 0.0
        best_answer = None

        for cached_question, (cached_embedding, cached_answer) in self.cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = cached_answer

        # Return if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold:
            return best_answer
        return None

    def set(self, question: str, answer: str, context_summary: str = ""):
        """Save question-answer pair to cache.

        Args:
            question: User's question
            answer: LLM response
            context_summary: Optional context summary
        """
        # Create cache key with context
        cache_key = f"{question}|{context_summary}"
        embedding = self._get_embedding(cache_key)
        self.cache[cache_key] = (embedding, answer)
        print(f"{Fore.GREEN}[DEBUG] Saved to cache (total entries: {len(self.cache)}){Style.RESET_ALL}")