"""
Reranker — uses a cross-encoder model to re-score retrieved documents.
Input: top-20 candidates from hybrid retrieval.
Output: top-5 most relevant documents.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU — suppresses CUDA driver warning

import asyncio
import logging
from typing import List
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from app.core.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self):
        self.model: CrossEncoder = None

    def load_model(self):
        """Load cross-encoder model (lazy — called on first use)."""
        if self.model is None:
            logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
            self.model = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
            logger.info("Reranker model loaded")

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        Re-score documents using cross-encoder.
        Returns top_k documents sorted by relevance.
        """
        if not documents:
            return []

        self.load_model()

        # Build query-document pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Cross-encoder scoring is CPU-bound → run in executor
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, lambda p=pairs: self.model.predict(p).tolist()
        )

        # Attach scores and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = round(float(score), 4)
            results.append(doc)

        logger.info(
            f"Reranker: {len(documents)} → {len(results)} docs "
            f"(top score: {results[0].metadata['rerank_score'] if results else 'N/A'})"
        )
        return results
