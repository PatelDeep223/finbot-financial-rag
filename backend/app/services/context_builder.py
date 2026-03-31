"""
Context Builder — assembles the final context for the LLM.
Deduplicates, orders by relevance, and trims to token limit.
"""

import hashlib
import logging
from typing import List
from langchain.schema import Document
from app.core.config import settings

logger = logging.getLogger(__name__)

# Rough approximation: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4


class ContextBuilder:
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or settings.MAX_CONTEXT_TOKENS

    def build(self, documents: List[Document]) -> str:
        """
        Build context string from documents.
        1. Deduplicate by content hash
        2. Order by rerank score (if available), then RRF score
        3. Trim to token limit
        """
        if not documents:
            return ""

        # 1. Deduplicate
        seen = set()
        unique_docs = []
        for doc in documents:
            content_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        # 2. Sort by best available score
        def sort_key(doc):
            return (
                doc.metadata.get("rerank_score", 0),
                doc.metadata.get("rrf_score", 0),
            )
        unique_docs.sort(key=sort_key, reverse=True)

        # 3. Build context with token limit
        max_chars = self.max_tokens * CHARS_PER_TOKEN
        context_parts = []
        total_chars = 0

        for doc in unique_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            header = f"[Source: {source}" + (f", Page {page}]" if page != "" else "]")
            chunk = f"{header}\n{doc.page_content}"

            if total_chars + len(chunk) > max_chars:
                # Add partial if significant
                remaining = max_chars - total_chars
                if remaining > 200:
                    context_parts.append(chunk[:remaining] + "...")
                break

            context_parts.append(chunk)
            total_chars += len(chunk)

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            f"Context built: {len(unique_docs)} docs, "
            f"{len(context)} chars (~{len(context) // CHARS_PER_TOKEN} tokens)"
        )
        return context
