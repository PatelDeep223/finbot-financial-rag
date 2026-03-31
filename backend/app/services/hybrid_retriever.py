"""
Hybrid Retriever — combines BM25 keyword search with FAISS vector search.
Results merged using Reciprocal Rank Fusion (RRF).
"""

import asyncio
import logging
from typing import List, Optional
from rank_bm25 import BM25Okapi
from langchain.schema import Document

logger = logging.getLogger(__name__)

# RRF constant (standard value from the literature)
RRF_K = 60


class HybridRetriever:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []

    def build_bm25_index(self, documents: List[Document]):
        """Build BM25 index from LangChain documents."""
        self.documents = documents
        self._tokenized_corpus = [
            doc.page_content.lower().split() for doc in documents
        ]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built: {len(documents)} documents")

    def add_documents(self, new_docs: List[Document]):
        """Add new documents to BM25 index (rebuild)."""
        self.documents.extend(new_docs)
        self._tokenized_corpus.extend(
            [doc.page_content.lower().split() for doc in new_docs]
        )
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index updated: {len(self.documents)} total documents")

    def _bm25_search(self, query: str, top_k: int) -> List[tuple]:
        """Returns list of (Document, score) sorted by BM25 score."""
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    async def search(
        self,
        query: str,
        vectorstore,
        top_k: int = 20,
    ) -> List[Document]:
        """
        Hybrid search: BM25 + FAISS, merged with Reciprocal Rank Fusion.
        Returns top_k documents sorted by fused score.
        """
        loop = asyncio.get_event_loop()

        # Run BM25 and FAISS in parallel
        bm25_task = loop.run_in_executor(None, self._bm25_search, query, top_k)
        faiss_task = loop.run_in_executor(
            None, lambda q=query, k=top_k: vectorstore.similarity_search_with_score(q, k=k)
        )
        bm25_results, faiss_results = await asyncio.gather(bm25_task, faiss_task)

        logger.info(f"Hybrid search: BM25={len(bm25_results)} hits, FAISS={len(faiss_results)} hits")

        # Reciprocal Rank Fusion
        doc_scores = {}  # content_hash → {"doc": Document, "score": float}

        # Score BM25 results
        for rank, (doc, _score) in enumerate(bm25_results):
            key = doc.page_content[:200]
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 1.0 / (RRF_K + rank + 1)

        # Score FAISS results (faiss returns (doc, distance) — lower distance = better)
        for rank, (doc, _distance) in enumerate(faiss_results):
            key = doc.page_content[:200]
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 1.0 / (RRF_K + rank + 1)

        # Sort by fused score descending
        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

        # Store RRF score in metadata for downstream use
        results = []
        for item in ranked[:top_k]:
            doc = item["doc"]
            doc.metadata["rrf_score"] = round(item["score"], 6)
            results.append(doc)

        logger.info(f"RRF merged: {len(results)} candidates")
        return results
