# FinBot — Advanced Financial RAG System

> Built by **Deep Patel** | Python Backend & GenAI Engineer

Production-grade Financial Document Q&A with **10-stage advanced RAG pipeline** — hybrid retrieval, cross-encoder reranking, anti-hallucination, semantic caching, and confidence scoring.

---

## Architecture

```
User Question
      │
      ▼
┌──────────────────┐
│ 1. Semantic Cache │──── Redis / local fallback
│    (Redis)        │     Cache HIT → instant response (~15ms)
└────────┬─────────┘
         │ Cache MISS
         ▼
┌──────────────────┐
│ 2. Router Agent   │──── Intent classification
│    (Rules + LLM)  │     factual | comparison | summary | off_topic
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Query Rewriter │──── LLM rewrites complex queries
│    (GPT-3.5)      │     Skips short/clear queries (≤6 words)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Hybrid Search  │──── BM25 (keyword) + FAISS (vector)
│    BM25 ∥ FAISS   │     Run in parallel, merge with RRF
│    + RRF Fusion   │     → Top-20 candidates
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Reranker       │──── Cross-encoder (ms-marco-MiniLM-L-6-v2)
│    (Cross-Encoder)│     Rescores 20 → Top-5 most relevant
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. Context Builder│──── Deduplicate + sort by score
│                   │     Trim to 3000 token limit
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 7. LLM Generation │──── GPT-3.5 (temperature=0)
│    (OpenAI)       │     Strict financial prompt
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 8. Hallucination  │──── Confidence scoring (0.0-1.0)
│    Detector       │     Uncertainty phrase detection
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 9. Cache + DB Log │──── Redis cache + PostgreSQL logging
│                   │     Fire-and-forget, never blocks response
└────────┬─────────┘
         │
         ▼
   Answer + Sources + Confidence + Intent
```

---

## Key Features

| Feature | Implementation | Impact |
|---------|---------------|--------|
| **Hybrid Retrieval** | BM25 + FAISS + Reciprocal Rank Fusion | Catches both keyword and semantic matches |
| **Cross-Encoder Reranking** | ms-marco-MiniLM-L-6-v2 | 20 candidates → 5 most relevant docs |
| **Router Agent** | Rule-based + LLM fallback | Routes off-topic queries, adapts strategy per intent |
| **Anti-Hallucination** | temperature=0 + strict prompts + confidence scoring | Never fabricates financial data |
| **Semantic Caching** | Redis with MD5 hash keys | ~15ms cached responses vs ~3s uncached |
| **Query Rewriting** | LLM pre-processing (complex queries only) | Better retrieval without over-rewriting |
| **Context Builder** | Dedup + score sort + token trimming | Clean, relevant context for LLM |
| **PostgreSQL Logging** | Async fire-and-forget to Railway DB | Query logs, conversations, document metadata |
| **Graceful Fallbacks** | Redis → local dict, DB → silent skip | App never crashes from infra failures |
| **Source Citation** | Document + page metadata tracking | Every answer is verifiable |

---

## Project Structure

```
finbot-financial-rag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # 7 API endpoints
│   │   ├── core/
│   │   │   └── config.py              # Pydantic settings (env-based)
│   │   ├── db/
│   │   │   ├── __init__.py            # Async SQLAlchemy engine + session
│   │   │   └── service.py             # DB operations (graceful fallback)
│   │   ├── models/
│   │   │   ├── schemas.py             # Pydantic request/response models
│   │   │   └── database.py            # SQLAlchemy ORM (Document, Conversation, QueryLog)
│   │   ├── rag/
│   │   │   └── pipeline.py            # Main 10-stage RAG orchestrator
│   │   └── services/                  # Modular RAG components
│   │       ├── router.py              # Intent classification
│   │       ├── rewriter.py            # Query expansion
│   │       ├── hybrid_retriever.py    # BM25 + FAISS + RRF fusion
│   │       ├── reranker.py            # Cross-encoder re-scoring
│   │       └── context_builder.py     # Dedup, sort, trim
│   ├── tests/                         # 80 tests (pytest)
│   │   ├── conftest.py                # Fixtures + mocks (no real API/DB needed)
│   │   ├── test_services.py           # Router, rewriter, retriever, reranker, context
│   │   ├── test_query.py              # Query endpoint (11 tests)
│   │   ├── test_pipeline_unit.py      # HallucinationDetector + demo responses
│   │   ├── test_db_service.py         # DB graceful fallback tests
│   │   └── ...                        # Upload, stats, documents, history, schemas, health
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   └── index.html                     # Single-page app (vanilla JS, dark theme)
├── docker/
│   └── nginx.conf                     # Reverse proxy config
├── docker-compose.yml                 # 4-service orchestration
├── setup.sh                           # One-command Docker setup
└── .gitignore
```

---

## Tech Stack

```
Backend:      Python 3.12 · FastAPI · Async/Await
AI/ML:        LangChain · OpenAI GPT-3.5 · OpenAI Embeddings
Retrieval:    FAISS (vector) + BM25 (keyword) + Reciprocal Rank Fusion
Reranking:    sentence-transformers · cross-encoder/ms-marco-MiniLM-L-6-v2
Cache:        Redis (semantic caching with local fallback)
Database:     PostgreSQL (Railway) · SQLAlchemy 2.0 async
Frontend:     Vanilla HTML/CSS/JS (zero dependencies)
Deploy:       Docker · Docker Compose · Nginx
Testing:      pytest (80 tests, fully mocked)
```

---

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/PatelDeep223/finbot-financial-rag
cd finbot-financial-rag
chmod +x setup.sh
./setup.sh YOUR_OPENAI_API_KEY
```
Open http://localhost:3000

### Manual Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Edit with your OpenAI API key + DB URL

# Start (Redis optional — falls back to local cache)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend — open frontend/index.html in browser
```

### Run Tests
```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v    # 80 tests, no API keys needed
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Ask a question (main RAG pipeline) |
| `POST` | `/api/v1/upload` | Upload financial document (PDF/TXT) |
| `GET` | `/api/v1/stats` | System statistics + DB totals |
| `GET` | `/api/v1/documents` | List ingested documents |
| `GET` | `/api/v1/history/{session_id}` | Conversation history |
| `DELETE` | `/api/v1/history/{session_id}` | Clear conversation |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### Example Request
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Q3 revenue?", "user_id": "analyst_1", "session_id": "sess_001"}'
```

### Example Response
```json
{
  "answer": "According to the Q3 2024 Earnings Report, total revenue was $4.20 billion, representing 14.9% year-over-year growth.",
  "sources": [
    {
      "content": "Q3 2024 revenue reached $4.2 billion...",
      "source": "NovaTech_Q3_2024_Earnings.pdf",
      "page": 1,
      "score": 0.8934
    }
  ],
  "confident": true,
  "confidence_score": 0.92,
  "from_cache": false,
  "query_rewritten": null,
  "intent": "factual",
  "response_time_ms": 2341.5
}
```

---

## Advanced RAG Techniques

### 1. Hybrid Retrieval (BM25 + FAISS + RRF)
```
Query: "EPS $2.15"

BM25 (keyword):   Finds exact "$2.15" string match
FAISS (vector):   Finds semantically similar "earnings per share" chunks
RRF Fusion:       Merges both ranked lists → best of both worlds
                   score = Σ 1/(k + rank)  where k=60
```

### 2. Cross-Encoder Reranking
```
Input:  20 candidates from hybrid search
Model:  ms-marco-MiniLM-L-6-v2 (fast, accurate)
Output: Top-5 documents re-scored by true relevance
Why:    FAISS is fast but approximate; reranker is slow but precise
```

### 3. Router Agent (Intent Detection)
```
"What is revenue?"          → factual   (direct lookup)
"Compare Q3 vs Q2"         → comparison (multi-doc analysis)
"Summarize the report"     → summary   (broad overview)
"Hello"                    → off_topic (instant rejection)
```

### 4. Anti-Hallucination (4 Layers)
```
Layer 1: temperature=0              → deterministic output
Layer 2: Strict prompt              → "ONLY use context, never fabricate"
Layer 3: Uncertainty phrase detection → "I think", "probably" → lower score
Layer 4: Source verification         → no sources = -0.4 confidence
```

### 5. Context Builder
```
Input:  5 reranked documents (may have overlapping content)
Step 1: Deduplicate by content hash
Step 2: Sort by rerank_score → rrf_score
Step 3: Trim to 3000 token limit
Output: Clean, relevant context string for LLM
```

---

## Database Schema (PostgreSQL)

### documents
| Column | Type | Description |
|--------|------|-------------|
| id | int | Primary key |
| filename | varchar(500) | Indexed |
| file_size_bytes | bigint | |
| chunks_created | int | |
| uploaded_at | timestamp | Auto-set |

### conversations
| Column | Type | Description |
|--------|------|-------------|
| id | int | Primary key |
| session_id | varchar(255) | Indexed |
| role | varchar(20) | "user" / "assistant" |
| content | text | |
| confident | bool | Nullable |
| confidence_score | float | Nullable |
| sources | json | Nullable |
| created_at | timestamp | Indexed |

### query_logs
| Column | Type | Description |
|--------|------|-------------|
| id | int | Primary key |
| question | text | |
| rewritten_query | text | Nullable |
| answer | text | |
| confidence_score | float | |
| confident | bool | |
| from_cache | bool | |
| response_time_ms | float | |
| user_id | varchar(255) | Indexed |
| session_id | varchar(255) | Nullable, indexed |
| created_at | timestamp | |

---

## Performance

| Metric | Value |
|--------|-------|
| Response time (cached) | ~15ms |
| Response time (full pipeline) | ~3-5s |
| Cache hit rate (typical) | 40-60% |
| Reranker latency | ~200ms for 20 docs |
| Test suite | 80 tests in ~1s |
| Supported document size | Up to 500 pages |
| Concurrent requests | 10+ (fully async) |

---

## Environment Variables

```env
# Required
OPENAI_API_KEY=your-key-here

# Optional (all have defaults)
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
TOP_K_RETRIEVAL=20
TEMPERATURE=0.0
MAX_CONTEXT_TOKENS=3000
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Author

**Deep Patel** — Python Backend & GenAI Engineer
- [GitHub](https://github.com/PatelDeep223)
