import time
import pytest
from app.rag.pipeline import HallucinationDetector


class TestHallucinationDetector:
    def setup_method(self):
        self.detector = HallucinationDetector()

    def test_confident_answer(self):
        """Clean answer with sources should score high."""
        answer = "According to the Q3 report, revenue was $42 billion."
        sources = [{"content": "Q3 revenue...", "source": "report.pdf"}]
        is_confident, score = self.detector.analyze(answer, sources)
        assert is_confident is True
        assert score >= 0.8

    def test_uncertain_phrases_reduce_score(self):
        """Uncertainty phrases should lower confidence."""
        answer = "I think the revenue might be around $42 billion, probably."
        sources = [{"content": "...", "source": "report.pdf"}]
        is_confident, score = self.detector.analyze(answer, sources)
        # "I think" (-0.2) + "might be" (-0.2) + "probably" (-0.2) = 0.4
        assert score <= 0.5
        assert is_confident is False

    def test_single_uncertain_phrase(self):
        """One uncertainty phrase should still be fairly confident."""
        answer = "I believe the revenue was $42 billion based on the report."
        sources = [{"content": "...", "source": "report.pdf"}]
        is_confident, score = self.detector.analyze(answer, sources)
        # 1.0 - 0.2 + 0.1 = 0.9
        assert score >= 0.7

    def test_refusal_is_high_confidence(self):
        """Honest refusal should get 0.95 confidence."""
        answer = "I don't have enough information in the provided documents to answer."
        sources = []
        is_confident, score = self.detector.analyze(answer, sources)
        assert is_confident is True
        assert score == 0.95

    def test_no_sources_penalty(self):
        """No sources should reduce confidence by 0.4."""
        answer = "The revenue was $42 billion."
        sources = []
        is_confident, score = self.detector.analyze(answer, sources)
        # 1.0 - 0.4 = 0.6
        assert score == 0.6

    def test_short_answer_no_bonus(self):
        """Short answer (<=50 chars) should not get the +0.1 bonus."""
        answer = "Revenue was $42B."  # 17 chars
        sources = [{"content": "...", "source": "report.pdf"}]
        is_confident, score = self.detector.analyze(answer, sources)
        assert score == 1.0  # no bonus, no penalty

    def test_long_answer_with_sources_bonus(self):
        """Long answer with sources gets +0.1 bonus (capped at 1.0)."""
        answer = "According to the Q3 report, total revenue was $42 billion, a 15% increase."
        sources = [{"content": "...", "source": "report.pdf"}]
        is_confident, score = self.detector.analyze(answer, sources)
        assert score == 1.0  # 1.0 + 0.1 capped to 1.0

    def test_score_never_below_zero(self):
        """Even with many penalties, score should not go below 0."""
        answer = "I think it might be probably something I'm not sure about, could be anything."
        sources = []
        _, score = self.detector.analyze(answer, sources)
        assert score >= 0.0

    def test_score_never_above_one(self):
        """Score should not exceed 1.0."""
        answer = "Revenue is $42B according to the annual report, section 3.2, page 15."
        sources = [{"content": "...", "source": "f.pdf"}]
        _, score = self.detector.analyze(answer, sources)
        assert score <= 1.0


class TestDemoResponse:
    def test_demo_response_structure(self):
        from app.rag.pipeline import pipeline

        start = time.time()
        resp = pipeline._demo_response("What is revenue?", start)

        assert "answer" in resp
        assert "upload" in resp["answer"].lower() or "no documents" in resp["answer"].lower()
        assert resp["sources"] == []
        assert resp["confident"] is True
        assert resp["confidence_score"] == 1.0
        assert resp["from_cache"] is False
        assert resp["query_rewritten"] is None
        assert "response_time_ms" in resp
        assert "timestamp" in resp
