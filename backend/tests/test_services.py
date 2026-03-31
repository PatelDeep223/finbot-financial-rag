"""Tests for the new RAG service modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document


# ─── Router Tests ────────────────────────────────────────────────────────────

class TestQueryRouter:
    def setup_method(self):
        from app.services.router import QueryRouter
        self.router = QueryRouter(llm=None)  # rule-based only

    @pytest.mark.asyncio
    async def test_factual_query(self):
        intent = await self.router.classify("What is the Q3 2024 revenue?")
        assert intent == "factual"

    @pytest.mark.asyncio
    async def test_comparison_query(self):
        intent = await self.router.classify("Compare Q3 vs Q2 revenue")
        assert intent == "comparison"

    @pytest.mark.asyncio
    async def test_summary_query(self):
        intent = await self.router.classify("Give me a summary of the report")
        assert intent == "summary"

    @pytest.mark.asyncio
    async def test_tell_me_about(self):
        intent = await self.router.classify("Tell me about NovaTech's performance")
        assert intent == "summary"

    @pytest.mark.asyncio
    async def test_off_topic_greeting(self):
        intent = await self.router.classify("hello")
        assert intent == "off_topic"

    @pytest.mark.asyncio
    async def test_off_topic_thanks(self):
        intent = await self.router.classify("thanks")
        assert intent == "off_topic"

    @pytest.mark.asyncio
    async def test_risk_query(self):
        intent = await self.router.classify("What are the risk factors?")
        assert intent == "risk"

    @pytest.mark.asyncio
    async def test_risk_keyword(self):
        intent = await self.router.classify("List the main risks and challenges")
        assert intent == "risk"

    @pytest.mark.asyncio
    async def test_default_to_factual(self):
        intent = await self.router.classify("operating margin percentage last quarter")
        assert intent == "factual"


# ─── Rewriter Tests ──────────────────────────────────────────────────────────

class TestQueryRewriter:
    def setup_method(self):
        from app.services.rewriter import QueryRewriter
        self.rewriter = QueryRewriter(llm=None)

    @pytest.mark.asyncio
    async def test_short_query_unchanged(self):
        result = await self.rewriter.rewrite("What is revenue?")
        assert result == "What is revenue?"

    @pytest.mark.asyncio
    async def test_summary_intent_skipped(self):
        result = await self.rewriter.rewrite(
            "Give me a detailed overview of the entire financial report",
            intent="summary"
        )
        assert result == "Give me a detailed overview of the entire financial report"

    @pytest.mark.asyncio
    async def test_no_llm_returns_original(self):
        result = await self.rewriter.rewrite(
            "What was the total consolidated revenue for the third quarter of fiscal year 2024?",
            intent="factual"
        )
        assert result == "What was the total consolidated revenue for the third quarter of fiscal year 2024?"


# ─── Hybrid Retriever Tests ─────────────────────────────────────────────────

class TestHybridRetriever:
    def setup_method(self):
        from app.services.hybrid_retriever import HybridRetriever
        self.retriever = HybridRetriever()

    def test_build_bm25_index(self):
        docs = [
            Document(page_content="Revenue was $4.2 billion in Q3 2024"),
            Document(page_content="Net income increased 15% year over year"),
            Document(page_content="Free cash flow was $380 million"),
        ]
        self.retriever.build_bm25_index(docs)
        assert self.retriever.bm25 is not None
        assert len(self.retriever.documents) == 3

    def test_bm25_search(self):
        docs = [
            Document(page_content="Revenue was $4.2 billion in Q3 2024"),
            Document(page_content="The weather is nice today"),
            Document(page_content="Free cash flow was $380 million"),
        ]
        self.retriever.build_bm25_index(docs)
        results = self.retriever._bm25_search("revenue Q3", top_k=2)
        assert len(results) > 0
        # The revenue doc should rank highest
        assert "revenue" in results[0][0].page_content.lower()

    def test_bm25_empty_index(self):
        results = self.retriever._bm25_search("revenue", top_k=5)
        assert results == []

    def test_add_documents(self):
        docs1 = [Document(page_content="Revenue was $4.2B")]
        docs2 = [Document(page_content="Net income was $520M")]
        self.retriever.build_bm25_index(docs1)
        self.retriever.add_documents(docs2)
        assert len(self.retriever.documents) == 2
        assert self.retriever.bm25 is not None

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        docs = [
            Document(page_content="Revenue was $4.2 billion", metadata={}),
            Document(page_content="Net income $520 million", metadata={}),
        ]
        self.retriever.build_bm25_index(docs)

        # Mock FAISS vectorstore
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="Revenue was $4.2 billion", metadata={}), 0.1),
            (Document(page_content="Operating margin 18.5%", metadata={}), 0.3),
        ]

        results = await self.retriever.search("revenue", mock_vs, top_k=5)
        assert len(results) > 0
        # Revenue doc should be top (appears in both BM25 and FAISS)
        assert "revenue" in results[0].page_content.lower()
        assert "rrf_score" in results[0].metadata


# ─── Reranker Tests ──────────────────────────────────────────────────────────

class TestReranker:
    def setup_method(self):
        from app.services.reranker import Reranker
        self.reranker = Reranker()

    @pytest.mark.asyncio
    async def test_rerank_empty(self):
        results = await self.reranker.rerank("test", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_with_mock_model(self):
        import numpy as np
        self.reranker.model = MagicMock()
        self.reranker.model.predict.return_value = np.array([0.9, 0.3, 0.7])

        docs = [
            Document(page_content="Revenue $4.2B", metadata={}),
            Document(page_content="Weather is nice", metadata={}),
            Document(page_content="Net income $520M", metadata={}),
        ]
        results = await self.reranker.rerank("revenue", docs, top_k=2)
        assert len(results) == 2
        # Highest score (0.9) should be first
        assert results[0].page_content == "Revenue $4.2B"
        assert results[0].metadata["rerank_score"] == 0.9


# ─── Context Builder Tests ──────────────────────────────────────────────────

class TestContextBuilder:
    def setup_method(self):
        from app.services.context_builder import ContextBuilder
        self.builder = ContextBuilder(max_tokens=500)

    def test_build_empty(self):
        result = self.builder.build([])
        assert result == ""

    def test_build_with_docs(self):
        docs = [
            Document(page_content="Revenue was $4.2B", metadata={"source": "report.pdf", "page": 1, "rerank_score": 0.9}),
            Document(page_content="Net income $520M", metadata={"source": "report.pdf", "page": 2, "rerank_score": 0.7}),
        ]
        result = self.builder.build(docs)
        assert "Revenue was $4.2B" in result
        assert "Net income $520M" in result
        assert "[Source: report.pdf" in result

    def test_deduplication(self):
        doc = Document(page_content="Revenue was $4.2B", metadata={"source": "report.pdf", "rerank_score": 0.9})
        result = self.builder.build([doc, doc, doc])
        assert result.count("Revenue was $4.2B") == 1

    def test_token_limit(self):
        from app.services.context_builder import ContextBuilder as CB
        builder = CB(max_tokens=20)  # ~80 chars
        docs = [
            Document(page_content="A" * 200, metadata={"source": "a.pdf", "rerank_score": 0.9}),
            Document(page_content="B" * 200, metadata={"source": "b.pdf", "rerank_score": 0.5}),
        ]
        result = builder.build(docs)
        # Should not contain both full docs
        assert len(result) < 400

    def test_sorted_by_rerank_score(self):
        docs = [
            Document(page_content="Low score", metadata={"source": "a.pdf", "rerank_score": 0.2}),
            Document(page_content="High score", metadata={"source": "b.pdf", "rerank_score": 0.9}),
        ]
        result = self.builder.build(docs)
        high_pos = result.find("High score")
        low_pos = result.find("Low score")
        assert high_pos < low_pos  # High score appears first
