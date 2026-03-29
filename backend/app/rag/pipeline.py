import os
import json
import time
import hashlib
import asyncio
from typing import Optional, List, Tuple
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

import redis
import numpy as np
from app.core.config import settings
from app.models.schemas import SourceDocument


# ─── FINANCIAL RAG PROMPT ──────────────────────────────────────────────────
FINANCIAL_RAG_PROMPT = PromptTemplate(
    template="""You are FinBot, an expert financial analyst AI assistant.

INSTRUCTIONS:
1. Read the context documents below carefully
2. Answer the user's question using ONLY data found in the context
3. Quote exact numbers, percentages, and dollar amounts as they appear
4. Cite the document name and page/section where you found the data
5. If the context contains ANY relevant information, you MUST answer — do NOT refuse
6. Only say "I don't have enough information in the provided documents to answer this question." when the context has absolutely nothing related to the question
7. Never invent financial data that is not in the context

Context Documents:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# ─── QUERY REWRITE PROMPT ──────────────────────────────────────────────────
QUERY_REWRITE_PROMPT = """Rewrite this financial query to improve document retrieval accuracy.

RULES:
- Keep the original meaning and intent exactly the same
- Add relevant financial terminology (e.g., "revenue" → "total revenue", "debt" → "debt-to-equity ratio")
- NEVER invent or assume company names, ticker symbols, or specific years/dates
- NEVER add information that was not in the original query
- If the query is already clear and specific, return it unchanged
- Return ONLY the rewritten query, nothing else

Original query: {query}
Rewritten query:"""


class SemanticCache:
    """Redis-based semantic cache for RAG queries"""
    
    def __init__(self):
        try:
            self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.redis.ping()
            self.enabled = True
            print("✅ Redis semantic cache connected")
        except Exception as e:
            print(f"⚠️ Redis not available, cache disabled: {e}")
            self.enabled = False
            self._local_cache = {}
    
    def _make_cache_key(self, query: str, embeddings) -> str:
        """Create cache key from query embedding"""
        try:
            embedding = embeddings.embed_query(query.lower().strip())
            # Use first 20 dimensions as cache key (fast approximation)
            key_data = str([round(x, 3) for x in embedding[:20]])
            return f"finbot:query:{hashlib.md5(key_data.encode()).hexdigest()}"
        except:
            return f"finbot:query:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return self._local_cache.get(key)
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except:
            return None
    
    def set(self, key: str, value: dict, ttl: int = None):
        ttl = ttl or settings.CACHE_TTL
        if not self.enabled:
            self._local_cache[key] = value
            return
        try:
            self.redis.setex(key, ttl, json.dumps(value, default=str))
        except:
            pass
    
    def get_stats(self) -> dict:
        if not self.enabled:
            return {"keys": len(self._local_cache), "status": "local"}
        try:
            keys = self.redis.keys("finbot:query:*")
            return {"keys": len(keys), "status": "redis"}
        except:
            return {"keys": 0, "status": "error"}


class HallucinationDetector:
    """Detects potential hallucinations using multiple strategies"""
    
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
        """
        Returns (is_confident, confidence_score)
        confidence_score: 0.0 to 1.0
        """
        answer_lower = answer.lower()
        score = 1.0
        
        # Check for uncertainty phrases (reduce confidence)
        for phrase in self.UNCERTAINTY_PHRASES:
            if phrase in answer_lower:
                score -= 0.2
        
        # Refusal phrases = honest response (high confidence in refusal)
        for phrase in self.REFUSAL_PHRASES:
            if phrase in answer_lower:
                return True, 0.95  # Confident refusal
        
        # No sources = potential hallucination
        if not sources:
            score -= 0.4
        
        # Short answer with sources = likely good
        if sources and len(answer) > 50:
            score += 0.1
        
        # Cap score between 0 and 1
        score = max(0.0, min(1.0, score))
        is_confident = score >= 0.6
        
        return is_confident, round(score, 2)


class FinancialRAGPipeline:
    """
    Production-grade Financial RAG Pipeline
    
    Features:
    - Semantic caching (Redis)
    - Query rewriting
    - Anti-hallucination via temperature=0 + strict prompts
    - Confidence scoring
    - Source citation
    - Conversation memory
    """
    
    def __init__(self):
        print("🚀 Initializing FinancialRAGPipeline...")
        
        # Core LLM - temperature=0 for anti-hallucination
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=1000
        )
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        
        # Components
        self.cache = SemanticCache()
        self.hallucination_detector = HallucinationDetector()
        self.vectorstore = None
        self.conversation_history = {}  # session_id -> messages
        
        # Stats tracking
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "start_time": time.time()
        }
        
        # Load existing vectorstore if available
        self._load_vectorstore()
        print("✅ FinancialRAGPipeline ready!")
    
    def _load_vectorstore(self):
        """Load existing FAISS vectorstore"""
        vs_path = settings.VECTOR_STORE_PATH
        if os.path.exists(f"{vs_path}/index.faiss"):
            try:
                self.vectorstore = FAISS.load_local(
                    vs_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded existing vectorstore from {vs_path}")
            except Exception as e:
                print(f"⚠️ Could not load vectorstore: {e}")
    
    def _save_vectorstore(self):
        """Save FAISS vectorstore to disk"""
        if self.vectorstore:
            os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
            self.vectorstore.save_local(settings.VECTOR_STORE_PATH)
    
    async def ingest_document(self, file_path: str, filename: str) -> int:
        """Ingest a document into the vectorstore"""
        # Load document
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["ingested_at"] = datetime.now().isoformat()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        chunks = splitter.split_documents(documents)
        
        # Add to vectorstore
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
        
        # Persist
        self._save_vectorstore()

        # Save document record to DB (fire-and-forget)
        asyncio.ensure_future(self._save_document_to_db(file_path, filename, len(chunks)))

        print(f"✅ Ingested {filename}: {len(chunks)} chunks created")
        return len(chunks)

    async def _save_document_to_db(self, file_path: str, filename: str, chunks_created: int):
        from app.db import service as db_service
        await db_service.save_document_record(
            filename=filename,
            file_size_bytes=os.path.getsize(file_path),
            chunks_created=chunks_created,
        )
    
    def _rewrite_query(self, query: str) -> str:
        """Rewrite query for better retrieval. Skip for short/clear queries."""
        # Short, clear queries don't need rewriting — it often makes them worse
        if len(query.split()) <= 8:
            return query
        try:
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            rewritten = self.llm.invoke(prompt).content.strip()
            # Reject rewrites that introduce hallucinated content
            if rewritten and len(rewritten) > 5 and "company xyz" not in rewritten.lower():
                return rewritten
        except:
            pass
        return query
    
    def _format_sources(self, docs) -> List[SourceDocument]:
        """Format source documents for response"""
        sources = []
        seen = set()
        for doc in docs:
            content_hash = hashlib.md5(doc.page_content[:100].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                sources.append(SourceDocument(
                    content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page", None),
                    score=doc.metadata.get("score", None)
                ))
        return sources[:3]  # Top 3 sources
    
    async def query(
        self,
        question: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None
    ) -> dict:
        """
        Main RAG query pipeline:
        1. Check semantic cache
        2. Rewrite query
        3. Retrieve documents
        4. Generate answer
        5. Detect hallucinations
        6. Cache result
        7. Return with sources + confidence
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # No vectorstore = demo mode
        if self.vectorstore is None:
            return self._demo_response(question, start_time)
        
        # 1. Check semantic cache
        cache_key = self.cache._make_cache_key(question, self.embeddings)
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            cached["from_cache"] = True
            cached["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            # Log cached query to DB
            asyncio.ensure_future(self._log_query_to_db(
                question=question,
                rewritten_query=None,
                answer=cached.get("answer", ""),
                confidence_score=cached.get("confidence_score", 0),
                confident=cached.get("confident", True),
                from_cache=True,
                response_time_ms=cached["response_time_ms"],
                user_id=user_id,
                session_id=session_id,
            ))
            return cached
        
        # 2. Rewrite query for better retrieval
        rewritten_query = self._rewrite_query(question)
        
        # 3. Retrieve relevant documents using REWRITTEN query (better semantic match)
        docs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vectorstore.similarity_search(rewritten_query, k=settings.TOP_K_RESULTS)
        )

        # 4. Build context from retrieved documents
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # 5. Generate answer using ORIGINAL question (avoids hallucinated rewrites confusing the LLM)
        filled_prompt = FINANCIAL_RAG_PROMPT.format(context=context, question=question)
        answer = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(filled_prompt)
        )

        source_docs = docs
        sources = self._format_sources(source_docs)
        
        # 6. Hallucination detection
        is_confident, confidence_score = self.hallucination_detector.analyze(
            answer, sources
        )
        
        # 7. Build response
        response = {
            "answer": answer,
            "sources": [s.dict() for s in sources],
            "confident": is_confident,
            "confidence_score": confidence_score,
            "from_cache": False,
            "query_rewritten": rewritten_query if rewritten_query != question else None,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        # 8. Cache the result
        self.cache.set(cache_key, response)

        # 9. Log query to DB (fire-and-forget)
        asyncio.ensure_future(self._log_query_to_db(
            question=question,
            rewritten_query=rewritten_query if rewritten_query != question else None,
            answer=answer,
            confidence_score=confidence_score,
            confident=is_confident,
            from_cache=False,
            response_time_ms=response["response_time_ms"],
            user_id=user_id,
            session_id=session_id,
        ))

        # 10. Store in conversation history
        if session_id:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            self.conversation_history[session_id].append({
                "role": "user", "content": question
            })
            self.conversation_history[session_id].append({
                "role": "assistant", "content": answer
            })
            # Persist to DB (fire-and-forget)
            asyncio.ensure_future(self._save_conversation_to_db(
                session_id=session_id,
                question=question,
                answer=answer,
                is_confident=is_confident,
                confidence_score=confidence_score,
                sources=sources,
            ))

        return response
    
    def _demo_response(self, question: str, start_time: float) -> dict:
        """Demo response when no documents are loaded"""
        return {
            "answer": f"📂 No documents loaded yet! Please upload financial documents first using the 'Upload Document' button. Once uploaded, I can answer questions like: '{question}'",
            "sources": [],
            "confident": True,
            "confidence_score": 1.0,
            "from_cache": False,
            "query_rewritten": None,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _log_query_to_db(self, **kwargs):
        from app.db import service as db_service
        await db_service.log_query(**kwargs)

    async def _save_conversation_to_db(
        self, session_id, question, answer, is_confident, confidence_score, sources
    ):
        from app.db import service as db_service
        await db_service.save_conversation_message(
            session_id=session_id, role="user", content=question
        )
        await db_service.save_conversation_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            confident=is_confident,
            confidence_score=confidence_score,
            sources=[s.dict() for s in sources] if sources else None,
        )

    def get_stats(self) -> dict:
        cache_hit_rate = (
            self.stats["cache_hits"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0 else 0
        )
        return {
            "total_queries": self.stats["total_queries"],
            "cache_hit_rate": round(cache_hit_rate * 100, 1),
            "cache_stats": self.cache.get_stats(),
            "vectorstore_loaded": self.vectorstore is not None,
            "uptime_seconds": round(time.time() - self.stats["start_time"], 0)
        }


# Global pipeline instance
pipeline = FinancialRAGPipeline()
