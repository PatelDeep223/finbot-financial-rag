"""
Router Agent — classifies query intent before retrieval.

Intents:
  factual    → exact lookup ("What is the revenue?")
  comparison → compare two things ("Q3 vs Q2")
  summary    → broad overview ("Tell me about NovaTech")
  risk       → risk-related question ("What are the risk factors?")
  off_topic  → unrelated to finance ("What's the weather?")
"""

import asyncio
import logging
from typing import Literal

logger = logging.getLogger(__name__)

QueryIntent = Literal["factual", "comparison", "summary", "risk", "off_topic"]

ROUTER_PROMPT = """Classify the user query into one of these categories:

- factual (specific data like revenue, EPS)
- comparison (compare values across periods)
- summary (summarize document)
- risk (risk-related question)
- off_topic (not related to financial report)

Query:
{query}

Category:"""

# Fast rule-based checks before burning an LLM call
OFF_TOPIC_KEYWORDS = {"weather", "recipe", "joke", "hello", "hi", "hey", "thanks", "thank you"}
COMPARISON_KEYWORDS = {"compare", "vs", "versus", "difference", "between", "against"}
SUMMARY_KEYWORDS = {"summarize", "summary", "overview", "tell me about", "explain", "everything about"}
RISK_KEYWORDS = {"risk", "risks", "risk factors", "threats", "challenges", "uncertainties"}


class QueryRouter:
    def __init__(self, llm=None):
        self.llm = llm

    async def classify(self, query: str) -> QueryIntent:
        """Classify query intent. Uses rules first, LLM fallback."""
        q_lower = query.lower().strip()

        # Rule-based fast path
        words = set(q_lower.split())
        if words & OFF_TOPIC_KEYWORDS and len(words) <= 4:
            logger.info(f"Router [rules]: off_topic — '{query}'")
            return "off_topic"
        if words & COMPARISON_KEYWORDS:
            logger.info(f"Router [rules]: comparison — '{query}'")
            return "comparison"
        if words & RISK_KEYWORDS:
            logger.info(f"Router [rules]: risk — '{query}'")
            return "risk"
        if any(kw in q_lower for kw in SUMMARY_KEYWORDS):
            logger.info(f"Router [rules]: summary — '{query}'")
            return "summary"

        # LLM fallback for ambiguous queries
        if self.llm:
            try:
                loop = asyncio.get_running_loop()
                prompt = ROUTER_PROMPT.format(query=query)
                result = await loop.run_in_executor(
                    None, lambda p=prompt: self.llm.invoke(p).content.strip().lower()
                )
                if result in ("factual", "comparison", "summary", "risk", "off_topic"):
                    logger.info(f"Router [LLM]: {result} — '{query}'")
                    return result
            except Exception as e:
                logger.warning(f"Router LLM failed: {e}")

        # Default to factual
        logger.info(f"Router [default]: factual — '{query}'")
        return "factual"
