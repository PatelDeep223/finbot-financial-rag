import os
import json
import time
import hashlib
import asyncio
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


FINANCIAL_RAG_PROMPT = PromptTemplate(
    template="""You are FinBot, an expert financial analyst AI assistant.

INSTRUCTIONS:
1. Answer using ONLY data found in the context documents below
2. Quote exact numbers, percentages, and dollar amounts as they appear
3. Cite the document name where you found the data
4. If context has ANY relevant info, you MUST answer — do NOT refuse
5. Only say "I don't have enough information" when context has NOTHING related
6. Never invent financial data not in the context

Context Documents:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

QUERY_REWRITE_PROMPT = """Rewrite this financial query to improve document retrieval.
RULES: Keep original meaning. Add financial terminology. NEVER invent company names or dates.
Return ONLY the rewritten query.

Original: {query}
Rewritten:"""


class SemanticCache:
    def __init__(self):
        try:
            self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.redis.ping()
            self.enabled = True
            print("✅ Redis cache connected")
        except Exception as e:
            print(f"⚠️  Redis unavailable, using local cache: {e}")
            self.enabled = False
            self._local_cache = {}

    # FIX 1: Simple string hash — no API calls, no blocking
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


class FinancialRAGPipeline:
    def __init__(self):
        print("🚀 Initializing FinancialRAGPipeline...")
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=1000
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        self.cache = SemanticCache()
        self.hallucination_detector = HallucinationDetector()
        self.vectorstore: Optional[FAISS] = None
        self.conversation_history: dict = {}
        self.stats = {"total_queries": 0, "cache_hits": 0, "start_time": time.time()}
        self._load_vectorstore()
        print("✅ Pipeline ready!")

    def _load_vectorstore(self):
        vs_path = settings.VECTOR_STORE_PATH
        if os.path.exists(os.path.join(vs_path, "index.faiss")):
            try:
                self.vectorstore = FAISS.load_local(
                    vs_path, self.embeddings, allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded vectorstore from {vs_path}")
            except Exception as e:
                # FIX 6: Corrupted index — reset gracefully
                print(f"⚠️  Corrupted vectorstore, resetting: {e}")
                self.vectorstore = None

    def _save_vectorstore(self):
        if self.vectorstore:
            os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
            self.vectorstore.save_local(settings.VECTOR_STORE_PATH)

    async def ingest_document(self, file_path: str, filename: str) -> int:
        # FIX: get_running_loop() is correct inside async context (get_event_loop deprecated in 3.10+)
        loop = asyncio.get_running_loop()
        loader = PyPDFLoader(file_path) if filename.lower().endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
        documents = await loop.run_in_executor(None, loader.load)
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["ingested_at"] = datetime.now().isoformat()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
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
        self._save_vectorstore()
        # FIX 4+5: create_task + safe wrapper
        asyncio.create_task(self._safe_save_document_to_db(file_path, filename, len(chunks)))
        print(f"✅ Ingested '{filename}': {len(chunks)} chunks")
        return len(chunks)

    # FIX 2: Async rewrite — never blocks event loop
    async def _rewrite_query(self, query: str) -> str:
        if len(query.split()) <= 6:
            return query
        try:
            loop = asyncio.get_running_loop()  # FIX: use get_running_loop
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            response = await loop.run_in_executor(
                None, lambda p=prompt: self.llm.invoke(p).content.strip()
            )
            if response and 5 < len(response) < len(query) * 3:
                return response
        except Exception as e:
            print(f"⚠️  Query rewrite failed: {e}")
        return query

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
                    score=doc.metadata.get("score")
                ))
        return sources[:3]

    async def query(self, question: str, user_id: str = "anonymous", session_id: Optional[str] = None) -> dict:
        start_time = time.time()
        self.stats["total_queries"] += 1

        if self.vectorstore is None:
            return self._demo_response(question, start_time)

        # 1. Cache check
        cache_key = self.cache.make_cache_key(question)
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            cached["from_cache"] = True
            cached["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            return cached

        # 2. Rewrite query (async, non-blocking)
        rewritten = await self._rewrite_query(question)

        # 3. FAISS retrieval (CPU-bound → executor)
        loop = asyncio.get_running_loop()  # FIX: use get_running_loop
        docs = await loop.run_in_executor(
            None,
            lambda rq=rewritten: self.vectorstore.similarity_search(rq, k=settings.TOP_K_RESULTS)
        )

        if not docs:
            return self._no_docs_response(question, start_time)

        # 4. Build context
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        # 5. LLM generation (IO-bound → executor) — FIX 9: default arg capture
        filled = FINANCIAL_RAG_PROMPT.format(context=context, question=question)
        answer = await loop.run_in_executor(
            None, lambda p=filled: self.llm.invoke(p).content
        )

        # 6. Sources + hallucination check
        sources = self._format_sources(docs)
        is_confident, confidence_score = self.hallucination_detector.analyze(answer, sources)

        # 7. Build response
        response_time = round((time.time() - start_time) * 1000, 2)
        response = {
            "answer": answer,
            "sources": [s.dict() for s in sources],
            "confident": is_confident,
            "confidence_score": confidence_score,
            "from_cache": False,
            "query_rewritten": rewritten if rewritten != question else None,
            "response_time_ms": response_time,
            "timestamp": datetime.now().isoformat()
        }

        # 8. Cache
        self.cache.set(cache_key, response)

        # 9. Conversation history
        if session_id:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            self.conversation_history[session_id].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])

        # 10. DB logging (fire-and-forget, never crashes pipeline)
        asyncio.create_task(self._safe_log_query_to_db(
            question=question,
            rewritten_query=rewritten if rewritten != question else None,
            answer=answer, confidence_score=confidence_score,
            confident=is_confident, from_cache=False,
            response_time_ms=response_time, user_id=user_id, session_id=session_id,
        ))

        return response

    def _demo_response(self, question: str, start_time: float) -> dict:
        return {
            "answer": "📂 No documents loaded yet! Please upload a financial PDF or TXT file using the 'Choose File' button.\n\nOnce uploaded, I can answer: \"" + question + "\"",
            "sources": [], "confident": True, "confidence_score": 1.0,
            "from_cache": False, "query_rewritten": None,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }

    def _no_docs_response(self, question: str, start_time: float) -> dict:
        return {
            "answer": "I don't have enough information in the provided documents to answer this question.",
            "sources": [], "confident": True, "confidence_score": 0.95,
            "from_cache": False, "query_rewritten": None,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }

    # FIX 5: All DB calls wrapped — pipeline never crashes if DB is missing
    async def _safe_save_document_to_db(self, file_path, filename, chunks):
        try:
            from app.db import service as db_service
            await db_service.save_document_record(
                filename=filename,
                file_size_bytes=os.path.getsize(file_path),
                chunks_created=chunks,
            )
        except Exception as e:
            print(f"⚠️  DB log skipped (document): {e}")

    async def _safe_log_query_to_db(self, **kwargs):
        try:
            from app.db import service as db_service
            await db_service.log_query(**kwargs)
        except Exception as e:
            print(f"⚠️  DB log skipped (query): {e}")

    def get_stats(self) -> dict:
        total = self.stats["total_queries"]
        hits = self.stats["cache_hits"]
        return {
            "total_queries": total,
            "cache_hit_rate": round((hits / total * 100) if total > 0 else 0, 1),
            "cache_stats": self.cache.get_stats(),
            "vectorstore_loaded": self.vectorstore is not None,
            "uptime_seconds": round(time.time() - self.stats["start_time"], 0)
        }


pipeline = FinancialRAGPipeline()