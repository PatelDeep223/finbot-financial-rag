"""
MLflow Experiment Tracking for FinBot RAG Pipeline.

Tests combinations of chunk_size, top_k_retrieval, and embedding_model,
runs RAGAS evaluation on each, and logs all params + metrics to MLflow.

Usage:
    cd backend
    source venv/bin/activate
    python experiments/rag_experiments.py

View results:
    mlflow ui --port 5000
    # Open http://localhost:5000
"""

import os
import sys
import json
import time
import asyncio
import logging
import itertools
import tempfile
import shutil

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── EXPERIMENT CONFIG ───────────────────────────────────────────────────────

EXPERIMENT_NAME = "finbot-rag-hyperparams"

PARAM_GRID = {
    "chunk_size": [200, 300, 500],
    "top_k_retrieval": [10, 15, 20],
    "embedding_model": ["text-embedding-ada-002", "text-embedding-3-small"],
}

# Fixed params (not varied in this experiment)
FIXED_PARAMS = {
    "chunk_overlap_ratio": 0.1,  # overlap = chunk_size * ratio
    "top_k_rerank": 5,
    "llm_model": "gpt-3.5-turbo",
    "temperature": 0.0,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

# Document to use for experiments
DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "documents", "NovaTech_Q3_2024_Earnings_Report.pdf")
EVAL_TEST_SET = os.path.join(os.path.dirname(__file__), "..", "data", "eval_test_set.json")

# MLflow tracking URI (local by default)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


# ─── MINI PIPELINE (isolated per experiment run) ─────────────────────────────

class ExperimentPipeline:
    """
    Lightweight RAG pipeline for experiment runs.
    Each instance gets its own vectorstore + BM25 index — fully isolated.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        top_k_retrieval: int,
        top_k_rerank: int,
        embedding_model: str,
        llm_model: str,
        temperature: float,
    ):
        from app.core.config import settings

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=llm_model,
            temperature=temperature,
            max_tokens=1000,
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=embedding_model,
        )

        self.vectorstore = None
        self.bm25 = None
        self.bm25_docs = []
        self.num_chunks = 0

    def ingest(self, file_path: str):
        """Load and chunk the document, build FAISS + BM25 indexes."""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        )
        chunks = splitter.split_documents(documents)
        self.num_chunks = len(chunks)

        # Build FAISS
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Build BM25
        self.bm25_docs = chunks
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info(f"  Ingested: {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")

    def retrieve(self, query: str) -> list:
        """Hybrid retrieval: BM25 + FAISS → RRF merge."""
        k = self.top_k_retrieval
        RRF_K = 60

        # BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        bm25_results = [(self.bm25_docs[i], bm25_scores[i]) for i in bm25_ranked if bm25_scores[i] > 0]

        # FAISS
        faiss_results = self.vectorstore.similarity_search_with_score(query, k=k)

        # RRF fusion
        doc_scores = {}
        for rank, (doc, _) in enumerate(bm25_results):
            key = doc.page_content[:200]
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 1.0 / (RRF_K + rank + 1)

        for rank, (doc, _) in enumerate(faiss_results):
            key = doc.page_content[:200]
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 1.0 / (RRF_K + rank + 1)

        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[:self.top_k_rerank]]

    def generate(self, question: str, docs: list) -> tuple:
        """Generate answer from docs. Returns (answer, contexts)."""
        from app.rag.pipeline import FACTUAL_PROMPT

        contexts = [doc.page_content for doc in docs]
        context_str = "\n\n---\n\n".join(contexts)
        filled = FACTUAL_PROMPT.format(context=context_str, question=question)
        answer = self.llm.invoke(filled).content
        return answer, contexts

    def query(self, question: str) -> dict:
        """Full pipeline: retrieve → generate → return answer + contexts."""
        docs = self.retrieve(question)
        answer, contexts = self.generate(question, docs)
        return {"answer": answer, "contexts": contexts}


# ─── EVALUATOR (reuse existing) ─────────────────────────────────────────────

async def evaluate_pipeline(pipe: ExperimentPipeline, test_samples: list) -> dict:
    """Run all test questions through the pipeline, then evaluate with RAGAS metrics."""
    from app.services.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()
    eval_samples = []

    for sample in test_samples:
        result = pipe.query(sample["question"])
        eval_samples.append({
            "question": sample["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": sample["ground_truth"],
        })

    results = await evaluator.evaluate_batch(eval_samples)
    return results


# ─── MAIN EXPERIMENT RUNNER ──────────────────────────────────────────────────

def run_experiments():
    """Run grid search over hyperparameters, evaluate each, log to MLflow."""

    # Load test set
    with open(EVAL_TEST_SET) as f:
        test_data = json.load(f)
    test_samples = test_data["samples"]
    logger.info(f"Loaded {len(test_samples)} evaluation samples")

    # Verify document exists
    doc_path = os.path.abspath(DOCUMENT_PATH)
    if not os.path.exists(doc_path):
        logger.error(f"Document not found: {doc_path}")
        logger.error("Upload a document first via the FinBot API, then re-run.")
        sys.exit(1)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")
    logger.info(f"MLflow tracking: {MLFLOW_TRACKING_URI}")

    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    logger.info(f"Running {total} experiment combinations...")
    logger.info(f"  chunk_size:      {PARAM_GRID['chunk_size']}")
    logger.info(f"  top_k_retrieval: {PARAM_GRID['top_k_retrieval']}")
    logger.info(f"  embedding_model: {PARAM_GRID['embedding_model']}")

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        chunk_size = params["chunk_size"]
        top_k_retrieval = params["top_k_retrieval"]
        embedding_model = params["embedding_model"]
        chunk_overlap = int(chunk_size * FIXED_PARAMS["chunk_overlap_ratio"])

        run_name = f"cs{chunk_size}_k{top_k_retrieval}_{embedding_model.split('-')[-1]}"
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx}/{total}] {run_name}")
        logger.info(f"  chunk_size={chunk_size}, overlap={chunk_overlap}, "
                     f"top_k={top_k_retrieval}, embed={embedding_model}")

        with mlflow.start_run(run_name=run_name):
            try:
                run_start = time.time()

                # ── Log parameters ──
                mlflow.log_param("chunk_size", chunk_size)
                mlflow.log_param("chunk_overlap", chunk_overlap)
                mlflow.log_param("top_k_retrieval", top_k_retrieval)
                mlflow.log_param("embedding_model", embedding_model)
                for k, v in FIXED_PARAMS.items():
                    mlflow.log_param(k, v)

                # ── Build pipeline ──
                ingest_start = time.time()
                pipe = ExperimentPipeline(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k_retrieval=top_k_retrieval,
                    top_k_rerank=FIXED_PARAMS["top_k_rerank"],
                    embedding_model=embedding_model,
                    llm_model=FIXED_PARAMS["llm_model"],
                    temperature=FIXED_PARAMS["temperature"],
                )
                pipe.ingest(doc_path)
                ingest_time = time.time() - ingest_start

                mlflow.log_metric("num_chunks", pipe.num_chunks)
                mlflow.log_metric("ingest_time_s", round(ingest_time, 2))

                # ── Run evaluation ──
                eval_start = time.time()
                results = asyncio.run(evaluate_pipeline(pipe, test_samples))
                eval_time = time.time() - eval_start

                # ── Log RAGAS metrics ──
                avg = results["average_scores"]
                mlflow.log_metric("faithfulness", avg["faithfulness"])
                mlflow.log_metric("answer_relevancy", avg["answer_relevancy"])
                mlflow.log_metric("context_precision", avg["context_precision"])
                mlflow.log_metric("context_recall", avg["context_recall"])

                # Composite score (average of all 4)
                composite = round(sum(avg.values()) / len(avg), 3)
                mlflow.log_metric("ragas_composite", composite)

                mlflow.log_metric("eval_time_s", round(eval_time, 2))
                mlflow.log_metric("total_time_s", round(time.time() - run_start, 2))

                # ── Log per-sample details as artifact ──
                artifact_path = os.path.join(tempfile.mkdtemp(), "eval_results.json")
                with open(artifact_path, "w") as f:
                    json.dump(results, f, indent=2)
                mlflow.log_artifact(artifact_path)

                logger.info(f"  Results: faith={avg['faithfulness']:.3f} "
                           f"relev={avg['answer_relevancy']:.3f} "
                           f"prec={avg['context_precision']:.3f} "
                           f"recall={avg['context_recall']:.3f} "
                           f"composite={composite:.3f}")
                logger.info(f"  Time: ingest={ingest_time:.1f}s eval={eval_time:.1f}s")

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e)[:250])

    logger.info(f"\n{'='*60}")
    logger.info(f"All {total} experiments complete!")
    logger.info(f"View results: mlflow ui --port 5000")
    logger.info(f"Then open: http://localhost:5000")


if __name__ == "__main__":
    run_experiments()
