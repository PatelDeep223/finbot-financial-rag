"""
Query Rewriter — expands vague queries for better retrieval.
Only rewrites complex queries; short/clear ones pass through unchanged.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """Rewrite the query to improve retrieval from financial documents.

Rules:
- Make it more specific
- Add financial context if needed
- Keep meaning same

Examples:
"profit?" → "net profit in Q3 2024 earnings report"
"EPS?" → "earnings per share in Q3 2024"

Query:
{query}

Rewritten Query:"""


class QueryRewriter:
    def __init__(self, llm=None):
        self.llm = llm

    async def rewrite(self, query: str, intent: str = "factual") -> str:
        """Rewrite query for better retrieval. Skips short/clear queries."""
        # Short queries, summaries, and risk queries don't benefit from rewriting
        if len(query.split()) <= 6 or intent in ("summary", "risk"):
            return query

        if not self.llm:
            return query

        try:
            loop = asyncio.get_running_loop()
            prompt = REWRITE_PROMPT.format(query=query)
            rewritten = await loop.run_in_executor(
                None, lambda p=prompt: self.llm.invoke(p).content.strip()
            )
            # Reject bad rewrites
            if not rewritten or len(rewritten) <= 5:
                return query
            if len(rewritten) > len(query) * 3:
                return query
            if "company xyz" in rewritten.lower():
                return query

            logger.info(f"Rewriter: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Rewriter failed: {e}")
            return query
