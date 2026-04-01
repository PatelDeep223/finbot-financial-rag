import os
import json
import time
import hashlib
import asyncio
import logging
from typing import Optional, List, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import redis
from app.core.config import settings
from app.models.schemas import SourceDocument
from app.services.router import QueryRouter
from app.services.rewriter import QueryRewriter
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import Reranker
from app.services.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

# ─── PROMPTS ─────────────────────────────────────────────────────────────────

# ─── INTENT-SPECIFIC PROMPTS ─────────────────────────────────────────────────

# Response format appended to all prompts
RESPONSE_FORMAT = """
Format your response exactly like this:

Answer:
<your main answer here>

Key Insight:
<one-line interpretation or takeaway>

Source:
<document name + page number>"""

FACTUAL_PROMPT = PromptTemplate(
    template="""You are a financial AI assistant.

Your task is to answer the user's question using ONLY the provided context.

Rules:
- Do NOT make up information.
- If the answer is not in the context, say: "I don't have enough information to answer this."
- Always include key numbers (revenue, EPS, etc.) clearly.
- Keep answers concise, professional, and analyst-style.

Style Guidelines:
- Start with: "According to the report..."
- Use complete sentences
- Highlight key metrics (numbers, percentages)
- Avoid raw data dumping
""" + RESPONSE_FORMAT + """

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"]
)

SUMMARY_PROMPT = PromptTemplate(
    template="""You are a financial analyst.

Summarize the report in a professional and concise way.

Focus on:
- Overall performance
- Key financial metrics (revenue, profit, growth)
- Major business drivers (segments, AI, cloud)
- Final outlook

Avoid:
- Listing too many raw numbers
- Repeating same data
""" + RESPONSE_FORMAT + """

Context:
{context}

Summary:""",
    input_variables=["context"]
)

COMPARISON_PROMPT = PromptTemplate(
    template="""You are a financial analyst.

Compare the values across different periods clearly.

Rules:
- Mention both values
- Highlight increase/decrease
- Give a short interpretation
""" + RESPONSE_FORMAT + """

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"]
)

RISK_PROMPT = PromptTemplate(
    template="""You are a financial analyst.

Identify and summarize key risks from the report.

Focus on:
- Economic risks
- Competitive risks
- Operational risks
- Regulatory risks

Keep it structured and clear.
""" + RESPONSE_FORMAT + """

Context:
{context}

Answer:""",
    input_variables=["context"]
)

# Map intent → prompt
INTENT_PROMPTS = {
    "factual": FACTUAL_PROMPT,
    "comparison": COMPARISON_PROMPT,
    "summary": SUMMARY_PROMPT,
    "risk": RISK_PROMPT,
}

OFF_TOPIC_RESPONSE = (
    "I'm FinBot, a financial document analyst. I can only answer questions "
    "about the uploaded financial documents. Please ask a finance-related question."
)


# ─── CACHE ───────────────────────────────────────────────────────────────────

class SemanticCache:
    def __init__(self):
        try:
            self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.redis.ping()
            self.enabled = True
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, using local cache: {e}")
            self.enabled = False
            self._local_cache = {}

    def make_cache_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return f"finbot:query:{hashlib.md5(normalized.encode()).hexdigest()}"

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return self._local_cache.get(key)
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    def set(self, key: str, value: dict, ttl: int = None):
        ttl = ttl or settings.CACHE_TTL
        if not self.enabled:
            self._local_cache[key] = value
            return
        try:
            self.redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            pass

    def get_stats(self) -> dict:
        if not self.enabled:
            return {"keys": len(self._local_cache), "status": "local"}
        try:
            return {"keys": len(self.redis.keys("finbot:*")), "status": "redis"}
        except Exception:
            return {"keys": 0, "status": "error"}


# ─── HALLUCINATION DETECTOR ─────────────────────────────────────────────────

class HallucinationDetector:
    UNCERTAINTY_PHRASES = [
        "i think", "i believe", "probably", "might be", "could be",
        "i'm not sure", "i'm uncertain", "as far as i know",
        "to my knowledge", "i don't know for certain"
    ]
    REFUSAL_PHRASES = [
        "i don't have enough information",
        "not in the provided documents",
        "cannot find this information",
        "the context doesn't mention"
    ]

    def analyze(self, answer: str, sources: list) -> Tuple[bool, float]:
        if not answer:
            return False, 0.0
        answer_lower = answer.lower()
        score = 1.0
        for phrase in self.UNCERTAINTY_PHRASES:
            if phrase in answer_lower:
                score -= 0.2
        for phrase in self.REFUSAL_PHRASES:
            if phrase in answer_lower:
                return True, 0.95
        if not sources:
            score -= 0.4
        if sources and len(answer) > 50:
            score += 0.1
        score = max(0.0, min(1.0, score))
        return score >= 0.6, round(score, 2)


# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────

class FinancialRAGPipeline:
    """
    Advanced RAG Pipeline:
    1. Semantic Cache (Redis)
    2. Router Agent (intent classification)
    3. Query Rewriter (for complex queries)
    4. Hybrid Retrieval (BM25 + FAISS + RRF)
    5. Reranker (cross-encoder)
    6. Context Builder (dedup, sort, trim)
    7. LLM Generation (OpenAI, temp=0)
    8. Hallucination Detection
    9. Cache + DB logging
    """

    def __init__(self):
        logger.info("Initializing FinancialRAGPipeline...")

        # Core LLM + Embeddings
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=1000,
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )

        # Services
        self.router = QueryRouter(llm=self.llm)
        self.rewriter = QueryRewriter(llm=self.llm)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.context_builder = ContextBuilder()

        # Existing components
        self.cache = SemanticCache()
        self.hallucination_detector = HallucinationDetector()
        self.vectorstore: Optional[FAISS] = None
        self.conversation_history: dict = {}
        self.stats = {"total_queries": 0, "cache_hits": 0, "start_time": time.time()}

        self._load_vectorstore()
        logger.info("Pipeline ready!")

    # ─── VECTORSTORE MANAGEMENT ──────────────────────────────────────────

    def _load_vectorstore(self):
        vs_path = settings.VECTOR_STORE_PATH
        if os.path.exists(os.path.join(vs_path, "index.faiss")):
            try:
                self.vectorstore = FAISS.load_local(
                    vs_path, self.embeddings, allow_dangerous_deserialization=True
                )
                # Rebuild BM25 index from FAISS docstore
                self._rebuild_bm25_from_vectorstore()
                logger.info(f"Loaded vectorstore from {vs_path}")
            except Exception as e:
                logger.warning(f"Corrupted vectorstore, resetting: {e}")
                self.vectorstore = None

    def _rebuild_bm25_from_vectorstore(self):
        """Extract all docs from FAISS and build BM25 index."""
        if not self.vectorstore:
            return
        try:
            from langchain.schema import Document
            docstore = self.vectorstore.docstore
            all_docs = []
            for doc_id in docstore._dict:
                doc = docstore._dict[doc_id]
                if isinstance(doc, Document):
                    all_docs.append(doc)
            if all_docs:
                self.hybrid_retriever.build_bm25_index(all_docs)
                logger.info(f"BM25 index rebuilt: {len(all_docs)} docs")
        except Exception as e:
            logger.warning(f"BM25 rebuild failed (vector-only mode): {e}")

    def _save_vectorstore(self):
        if self.vectorstore:
            os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
            self.vectorstore.save_local(settings.VECTOR_STORE_PATH)

    # ─── DOCUMENT INGESTION ──────────────────────────────────────────────

    async def ingest_document(self, file_path: str, filename: str) -> int:
        loop = asyncio.get_running_loop()
        loader = (
            PyPDFLoader(file_path)
            if filename.lower().endswith(".pdf")
            else TextLoader(file_path, encoding="utf-8")
        )
        documents = await loop.run_in_executor(None, loader.load)

        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["ingested_at"] = datetime.now().isoformat()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError(f"No content extracted from {filename}")

        if self.vectorstore is None:
            self.vectorstore = await loop.run_in_executor(
                None, lambda c=chunks: FAISS.from_documents(c, self.embeddings)
            )
        else:
            await loop.run_in_executor(
                None, lambda c=chunks: self.vectorstore.add_documents(c)
            )

        # Update BM25 index with new chunks
        self.hybrid_retriever.add_documents(chunks)

        self._save_vectorstore()
        asyncio.create_task(self._safe_save_document_to_db(file_path, filename, len(chunks)))
        logger.info(f"Ingested '{filename}': {len(chunks)} chunks")
        return len(chunks)

    # ─── FORMAT SOURCES ──────────────────────────────────────────────────

    def _format_sources(self, docs) -> List[SourceDocument]:
        sources, seen = [], set()
        for doc in docs:
            key = hashlib.md5(doc.page_content[:100].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                sources.append(SourceDocument(
                    content=doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page"),
                    score=doc.metadata.get("rerank_score", doc.metadata.get("rrf_score")),
                ))
        return sources[:3]

    # ─── MAIN QUERY PIPELINE ─────────────────────────────────────────────

    async def query(
        self,
        question: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
    ) -> dict:
        start_time = time.time()
        self.stats["total_queries"] += 1

        if self.vectorstore is None:
            return self._demo_response(question, start_time)

        # ── Step 1: Semantic Cache ────────────────────────────────────
        cache_key = self.cache.make_cache_key(question)
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            cached["from_cache"] = True
            cached["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            return cached

        # ── Step 2: Router Agent ──────────────────────────────────────
        intent = await self.router.classify(question)
        if intent == "off_topic":
            return self._off_topic_response(question, start_time)

        # ── Step 3: Query Rewriter ────────────────────────────────────
        rewritten = await self.rewriter.rewrite(question, intent=intent)

        # ── Step 4: Hybrid Retrieval (BM25 + FAISS + RRF) ────────────
        candidates = await self.hybrid_retriever.search(
            query=rewritten,
            vectorstore=self.vectorstore,
            top_k=settings.TOP_K_RETRIEVAL,
        )

        if not candidates:
            return self._no_docs_response(question, start_time)

        # ── Step 5: Reranker (cross-encoder) ──────────────────────────
        top_docs = await self.reranker.rerank(
            query=question,  # rerank with ORIGINAL question for accuracy
            documents=candidates,
            top_k=settings.TOP_K_RESULTS,
        )

        if not top_docs:
            return self._no_docs_response(question, start_time)

        # ── Step 6: Context Builder ───────────────────────────────────
        context = self.context_builder.build(top_docs)

        # ── Step 7: LLM Generation (intent-specific prompt) ─────────
        loop = asyncio.get_running_loop()
        prompt_template = INTENT_PROMPTS.get(intent, FACTUAL_PROMPT)

        # Summary and risk prompts don't take a question variable
        if intent in ("summary", "risk"):
            filled_prompt = prompt_template.format(context=context)
        else:
            filled_prompt = prompt_template.format(context=context, question=question)

        answer = await loop.run_in_executor(
            None, lambda p=filled_prompt: self.llm.invoke(p).content
        )

        # ── Step 8: Hallucination Detection ───────────────────────────
        sources = self._format_sources(top_docs)
        is_confident, confidence_score = self.hallucination_detector.analyze(answer, sources)

        # ── Build Response ────────────────────────────────────────────
        response_time = round((time.time() - start_time) * 1000, 2)
        response = {
            "answer": answer,
            "sources": [s.dict() for s in sources],
            "confident": is_confident,
            "confidence_score": confidence_score,
            "from_cache": False,
            "query_rewritten": rewritten if rewritten != question else None,
            "intent": intent,
            "response_time_ms": response_time,
            "timestamp": datetime.now().isoformat(),
        }

        # ── Step 9: Cache Result ──────────────────────────────────────
        self.cache.set(cache_key, response)

        # ── Conversation History ──────────────────────────────────────
        if session_id:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            self.conversation_history[session_id].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])

        # ── DB Logging (fire-and-forget) ──────────────────────────────
        asyncio.create_task(self._safe_log_query_to_db(
            question=question,
            rewritten_query=rewritten if rewritten != question else None,
            answer=answer,
            confidence_score=confidence_score,
            confident=is_confident,
            from_cache=False,
            response_time_ms=response_time,
            user_id=user_id,
            session_id=session_id,
        ))

        return response

    # ─── STREAMING QUERY ──────────────────────────────────────────────────

    async def query_stream(
        self,
        question: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
    ):
        """
        Streaming version of query(). Yields SSE events:
          event: meta     → {intent, query_rewritten}
          event: token    → {token: "..."}
          event: sources  → {sources, confident, confidence_score, response_time_ms}
          event: done     → [DONE]
        """
        import json as _json
        start_time = time.time()
        self.stats["total_queries"] += 1

        # No vectorstore → demo
        if self.vectorstore is None:
            demo = self._demo_response(question, start_time)
            yield f"event: token\ndata: {_json.dumps({'token': demo['answer']})}\n\n"
            yield f"event: sources\ndata: {_json.dumps({'sources': [], 'confident': True, 'confidence_score': 1.0, 'response_time_ms': demo['response_time_ms']})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        # Step 1: Cache check
        cache_key = self.cache.make_cache_key(question)
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            yield f"event: meta\ndata: {_json.dumps({'intent': cached.get('intent'), 'query_rewritten': cached.get('query_rewritten'), 'from_cache': True})}\n\n"
            yield f"event: token\ndata: {_json.dumps({'token': cached['answer']})}\n\n"
            yield f"event: sources\ndata: {_json.dumps({'sources': cached.get('sources', []), 'confident': cached.get('confident', True), 'confidence_score': cached.get('confidence_score', 0), 'response_time_ms': round((time.time() - start_time) * 1000, 2)})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        # Step 2-6: Router → Rewriter → Retrieval → Reranker → Context (same as query())
        intent = await self.router.classify(question)
        if intent == "off_topic":
            resp = self._off_topic_response(question, start_time)
            yield f"event: token\ndata: {_json.dumps({'token': resp['answer']})}\n\n"
            yield f"event: sources\ndata: {_json.dumps({'sources': [], 'confident': True, 'confidence_score': 1.0, 'response_time_ms': resp['response_time_ms'], 'intent': 'off_topic'})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        rewritten = await self.rewriter.rewrite(question, intent=intent)

        # Send meta event (intent + rewrite info)
        yield f"event: meta\ndata: {_json.dumps({'intent': intent, 'query_rewritten': rewritten if rewritten != question else None, 'from_cache': False})}\n\n"

        candidates = await self.hybrid_retriever.search(
            query=rewritten, vectorstore=self.vectorstore, top_k=settings.TOP_K_RETRIEVAL,
        )
        if not candidates:
            resp = self._no_docs_response(question, start_time)
            yield f"event: token\ndata: {_json.dumps({'token': resp['answer']})}\n\n"
            yield f"event: sources\ndata: {_json.dumps({'sources': [], 'confident': True, 'confidence_score': 0.95, 'response_time_ms': resp['response_time_ms']})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        top_docs = await self.reranker.rerank(query=question, documents=candidates, top_k=settings.TOP_K_RESULTS)
        if not top_docs:
            resp = self._no_docs_response(question, start_time)
            yield f"event: token\ndata: {_json.dumps({'token': resp['answer']})}\n\n"
            yield f"event: sources\ndata: {_json.dumps({'sources': [], 'confident': True, 'confidence_score': 0.95, 'response_time_ms': resp['response_time_ms']})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        context = self.context_builder.build(top_docs)

        # Step 7: Build prompt
        prompt_template = INTENT_PROMPTS.get(intent, FACTUAL_PROMPT)
        if intent in ("summary", "risk"):
            filled_prompt = prompt_template.format(context=context)
        else:
            filled_prompt = prompt_template.format(context=context, question=question)

        # Step 7b: STREAM LLM tokens
        from langchain_openai import ChatOpenAI
        streaming_llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=1000,
            streaming=True,
        )

        full_answer = ""
        for chunk in streaming_llm.stream(filled_prompt):
            token = chunk.content
            if token:
                full_answer += token
                yield f"event: token\ndata: {_json.dumps({'token': token})}\n\n"

        # Step 8: Hallucination detection + sources
        sources = self._format_sources(top_docs)
        is_confident, confidence_score = self.hallucination_detector.analyze(full_answer, sources)
        response_time = round((time.time() - start_time) * 1000, 2)

        # Send final sources event
        yield f"event: sources\ndata: {_json.dumps({'sources': [s.dict() for s in sources], 'confident': is_confident, 'confidence_score': confidence_score, 'response_time_ms': response_time})}\n\n"

        # Cache + history + DB log (same as non-streaming)
        response = {
            "answer": full_answer,
            "sources": [s.dict() for s in sources],
            "confident": is_confident,
            "confidence_score": confidence_score,
            "from_cache": False,
            "query_rewritten": rewritten if rewritten != question else None,
            "intent": intent,
            "response_time_ms": response_time,
            "timestamp": datetime.now().isoformat(),
        }
        self.cache.set(cache_key, response)

        if session_id:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            self.conversation_history[session_id].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer},
            ])

        asyncio.create_task(self._safe_log_query_to_db(
            question=question, rewritten_query=rewritten if rewritten != question else None,
            answer=full_answer, confidence_score=confidence_score, confident=is_confident,
            from_cache=False, response_time_ms=response_time, user_id=user_id, session_id=session_id,
        ))

        yield "event: done\ndata: [DONE]\n\n"

    # ─── RESPONSE BUILDERS ───────────────────────────────────────────────

    def _demo_response(self, question: str, start_time: float) -> dict:
        return {
            "answer": (
                "📂 No documents loaded yet! Please upload a financial PDF or TXT file.\n\n"
                f'Once uploaded, I can answer: "{question}"'
            ),
            "sources": [], "confident": True, "confidence_score": 1.0,
            "from_cache": False, "query_rewritten": None, "intent": None,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
        }

    def _no_docs_response(self, question: str, start_time: float) -> dict:
        return {
            "answer": "I don't have enough information in the provided documents to answer this question.",
            "sources": [], "confident": True, "confidence_score": 0.95,
            "from_cache": False, "query_rewritten": None, "intent": None,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
        }

    def _off_topic_response(self, question: str, start_time: float) -> dict:
        return {
            "answer": OFF_TOPIC_RESPONSE,
            "sources": [], "confident": True, "confidence_score": 1.0,
            "from_cache": False, "query_rewritten": None, "intent": "off_topic",
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
        }

    # ─── DB HELPERS ──────────────────────────────────────────────────────

    async def _safe_save_document_to_db(self, file_path, filename, chunks):
        try:
            from app.db import service as db_service
            await db_service.save_document_record(
                filename=filename,
                file_size_bytes=os.path.getsize(file_path),
                chunks_created=chunks,
            )
        except Exception as e:
            logger.warning(f"DB log skipped (document): {e}")

    async def _safe_log_query_to_db(self, **kwargs):
        try:
            from app.db import service as db_service
            await db_service.log_query(**kwargs)
        except Exception as e:
            logger.warning(f"DB log skipped (query): {e}")

    # ─── STATS ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = self.stats["total_queries"]
        hits = self.stats["cache_hits"]
        return {
            "total_queries": total,
            "cache_hit_rate": round((hits / total * 100) if total > 0 else 0, 1),
            "cache_stats": self.cache.get_stats(),
            "vectorstore_loaded": self.vectorstore is not None,
            "bm25_loaded": self.hybrid_retriever.bm25 is not None,
            "uptime_seconds": round(time.time() - self.stats["start_time"], 0),
        }


# Global pipeline instance
pipeline = FinancialRAGPipeline()
