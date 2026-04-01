"""
RAGAS-style Evaluation Service for Financial RAG.

Implements 4 core metrics from the RAGAS paper (Es et al., 2023) using
direct OpenAI calls, specifically tuned for financial documents.

Methodology:
  1. Faithfulness     — Claim decomposition → verify each claim against context
  2. Answer Relevancy — Would this answer's question match the original?
  3. Context Precision — Per-chunk relevance scoring with rank weighting
  4. Context Recall    — Sentence-level ground truth coverage check

Each metric returns score (0.0–1.0) + one-line reason.
"""

import asyncio
import json
import logging
import re
from typing import List
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── PRODUCTION EVALUATION PROMPTS (Financial-Tuned) ─────────────────────────

FAITHFULNESS_PROMPT = """You are a financial auditor verifying whether an AI answer is grounded in source documents.

TASK: Evaluate faithfulness using claim-level decomposition (per RAGAS methodology).

Step 1 — Extract every factual claim from the answer. A claim is any statement that
asserts a specific fact: a number ($4.2B), a percentage (14.9%), a comparison (up from Q2),
a date, a named entity, or a causal relationship. Financial figures require EXACT match.

Step 2 — For each claim, verify if it is supported by the context:
  - SUPPORTED: the claim matches information explicitly stated in the context
  - UNSUPPORTED: the claim cannot be verified or contradicts the context
  - Financial precision rule: "$4.2 billion" and "$4,200M" are equivalent;
    "$4.2B" and "$4.3B" are NOT equivalent — even $100M difference is UNSUPPORTED

Step 3 — Calculate: faithfulness = (number of SUPPORTED claims) / (total claims)

Context:
{context}

Answer:
{answer}

Respond in this exact JSON format:
{{"score": <0.0 to 1.0>, "reason": "<one-line explanation citing which claims failed, if any>"}}"""

ANSWER_RELEVANCY_PROMPT = """You are a senior financial analyst evaluating whether an AI response actually answers the user's question.

TASK: Score answer relevancy using reverse question generation (per RAGAS methodology).

Consider:
1. Does the answer directly address what was asked? (not a tangential topic)
2. Is the answer complete? (e.g., if asked "What is the revenue?", just saying "revenue grew" without the number is incomplete)
3. Does the answer contain unnecessary information that dilutes the response?

Financial-specific rules:
- If asked for a specific metric (EPS, revenue, margin), the answer MUST include that exact metric
- If asked for a comparison, the answer MUST include both values being compared
- A refusal ("I don't have enough information") when the context clearly contains the answer scores 0.1
- A refusal when the context truly lacks the answer scores 0.9 (appropriate refusal)

Question:
{question}

Answer:
{answer}

Respond in this exact JSON format:
{{"score": <0.0 to 1.0>, "reason": "<one-line explanation of what was addressed or missing>"}}"""

CONTEXT_PRECISION_PROMPT = """You are a retrieval quality analyst for a financial RAG system.

TASK: Score context precision using per-chunk relevance assessment (per RAGAS methodology).

For each retrieved context chunk, determine:
  - RELEVANT: contains information that directly helps answer the question
  - IRRELEVANT: does not contain useful information for answering the question

Then calculate weighted precision (higher-ranked chunks matter more):
  precision = sum(relevant_i / rank_i) / sum(1 / rank_i) for i in 1..N

Financial-specific rules:
- A chunk mentioning the same metric (e.g., "revenue") but for a DIFFERENT period
  than asked is PARTIALLY relevant (0.5), not fully relevant
- A chunk with the exact data point asked about is RELEVANT (1.0)
- Boilerplate text (disclaimers, legal notices) is IRRELEVANT (0.0)

Question:
{question}

Retrieved Context Chunks (in retrieval order):
{context}

Respond in this exact JSON format:
{{"score": <0.0 to 1.0>, "reason": "<one-line summary: X of Y chunks relevant, note any noise>"}}"""

CONTEXT_RECALL_PROMPT = """You are a financial data completeness auditor.

TASK: Score context recall by checking sentence-level coverage of the ground truth
(per RAGAS methodology).

Step 1 — Decompose the ground truth into individual factual sentences/claims.
Step 2 — For each sentence, check if the retrieved context contains supporting evidence.
Step 3 — Calculate: recall = (sentences with context support) / (total sentences)

Financial-specific rules:
- A ground truth claim like "revenue was $4.2B" requires the context to contain
  that exact figure (or equivalent: "$4,200M", "$4.20 billion")
- Percentage claims (e.g., "14.9% YoY growth") need BOTH the percentage AND
  the comparison basis to be present in context
- If the ground truth mentions a trend ("increased from X to Y"), context must
  contain BOTH X and Y values

Ground Truth:
{ground_truth}

Retrieved Context:
{context}

Respond in this exact JSON format:
{{"score": <0.0 to 1.0>, "reason": "<one-line summary: X of Y ground truth claims covered>"}}"""


# ─── EVALUATOR CLASS ─────────────────────────────────────────────────────────

class RAGEvaluator:
    """Evaluates RAG pipeline quality using 4 RAGAS-style metrics."""

    def __init__(self):
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            self.llm = ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
                temperature=0.0,
            )
        return self.llm

    async def _score(self, prompt: str) -> dict:
        """Send prompt to LLM and parse JSON response with score + reason."""
        try:
            llm = self._get_llm()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda p=prompt: llm.invoke(p).content.strip()
            )

            # Parse JSON response
            # Handle cases where LLM wraps in ```json ... ```
            cleaned = re.sub(r'^```json\s*', '', result)
            cleaned = re.sub(r'\s*```$', '', cleaned)

            parsed = json.loads(cleaned)
            score = float(parsed.get("score", 0))
            reason = str(parsed.get("reason", ""))
            return {
                "score": max(0.0, min(1.0, round(score, 3))),
                "reason": reason,
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Fallback: try to extract just a number
            logger.warning(f"Evaluator JSON parse failed, trying number fallback: {e}")
            try:
                numbers = re.findall(r'(\d+\.?\d*)', result)
                if numbers:
                    score = float(numbers[0])
                    return {"score": max(0.0, min(1.0, score)), "reason": "parse fallback"}
            except Exception:
                pass
            return {"score": 0.0, "reason": f"parse error: {e}"}
        except Exception as e:
            logger.error(f"Evaluator LLM call failed: {e}")
            return {"score": 0.0, "reason": f"LLM error: {e}"}

    async def faithfulness(self, answer: str, context: str) -> dict:
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        return await self._score(prompt)

    async def answer_relevancy(self, question: str, answer: str) -> dict:
        prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        return await self._score(prompt)

    async def context_precision(self, question: str, context: str) -> dict:
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, context=context)
        return await self._score(prompt)

    async def context_recall(self, ground_truth: str, context: str) -> dict:
        prompt = CONTEXT_RECALL_PROMPT.format(ground_truth=ground_truth, context=context)
        return await self._score(prompt)

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> dict:
        """Evaluate a single Q&A pair on all 4 metrics (in parallel)."""
        context_str = "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{c}" for i, c in enumerate(contexts)
        )

        # Run all 4 metrics in parallel
        faith, relevancy, precision, recall = await asyncio.gather(
            self.faithfulness(answer, context_str),
            self.answer_relevancy(question, answer),
            self.context_precision(question, context_str),
            self.context_recall(ground_truth, context_str),
        )

        return {
            "question": question,
            "faithfulness": faith["score"],
            "faithfulness_reason": faith["reason"],
            "answer_relevancy": relevancy["score"],
            "answer_relevancy_reason": relevancy["reason"],
            "context_precision": precision["score"],
            "context_precision_reason": precision["reason"],
            "context_recall": recall["score"],
            "context_recall_reason": recall["reason"],
        }

    async def evaluate_batch(self, samples: List[dict]) -> dict:
        """
        Evaluate a batch of Q&A samples.
        Returns per-sample scores with reasons + aggregated averages.
        """
        results = []
        for sample in samples:
            result = await self.evaluate_single(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample["contexts"],
                ground_truth=sample["ground_truth"],
            )
            results.append(result)

        n = len(results)
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        avg = {m: round(sum(r[m] for r in results) / n, 3) for m in metrics}

        return {
            "num_samples": n,
            "average_scores": avg,
            "per_sample": results,
        }


# Singleton
evaluator = RAGEvaluator()
