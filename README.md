<h1 align="center">FinBot — Production Financial RAG System</h1>

<p align="center">
  <strong>10-stage RAG pipeline with hybrid retrieval, cross-encoder reranking, and 4-layer anti-hallucination — built for real financial documents.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--3.5-412991?logo=openai" />
  <img src="https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql" />
  <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis" />
  <img src="https://img.shields.io/badge/Tests-90%20passed-brightgreen" />
</p>

<p align="center">
  <em><!-- Replace with your demo GIF --></em><br/>
  <code>[ Demo GIF: Upload PDF → Ask question → Watch streaming answer with sources ]</code>
</p>

---

## Performance at a Glance

| Metric | Value |
|--------|-------|
| Cached response | **~15ms** |
| Full pipeline (uncached) | **3-5 seconds** |
| Cache hit rate | **40-60%** |
| Reranker latency | **~200ms** (20 → 5 docs) |
| Concurrent users | **10+** (fully async) |
| Document support | **Up to 500 pages** |
| Test suite | **90 tests**, 2 skipped, <2s runtime |
| Pipeline stages | **10** (hybrid retrieval + reranking + anti-hallucination) |

---

## Why Naive RAG Fails for Finance

Standard RAG pipelines (embed → retrieve → generate) **fail on financial documents** because:

| Problem | What Happens | FinBot's Solution |
|---------|-------------|-------------------|
| **Exact number misses** | Vector search finds "revenue grew" but misses the exact "$4.2B" | **Hybrid BM25 + FAISS** — keyword search catches exact figures, vector search catches meaning |
| **Irrelevant chunks ranked high** | FAISS returns 5 chunks, 2 are noise (disclaimers, headers) | **Cross-encoder reranker** — rescores 20 candidates, keeps only the 5 most relevant |
| **Hallucinated figures** | LLM invents "$4.3B" when the real number is "$4.2B" | **4-layer anti-hallucination** — temperature=0 + strict prompts + phrase detection + source verification |
| **Wrong query interpretation** | "EPS?" retrieves irrelevant earnings paragraphs | **Intent router + query rewriter** — classifies intent, rewrites vague queries for better retrieval |
| **Slow repeat queries** | Same question hits the full pipeline every time | **Redis semantic cache** — 15ms for repeated queries vs 3-5s uncached |
| **No accountability** | "Where did this answer come from?" | **Source citation** — every answer includes document name, page number, and relevance score |

---

## Architecture

```
User Question
      │
      ▼
┌──────────────────────┐
│  1. Semantic Cache    │──── Redis (hit? → 15ms response)
│     (Redis)           │     local dict fallback if Redis down
└──────────┬───────────┘
           │ Cache MISS
           ▼
┌──────────────────────┐
│  2. Router Agent      │──── Intent: factual | comparison | summary | risk | off_topic
│     (Rules + LLM)     │     off_topic → instant rejection
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. Query Rewriter    │──── LLM rewrites complex queries (skips short ones)
│     (GPT-3.5)         │     "profit?" → "net profit in Q3 2024 earnings report"
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. Hybrid Search     │──── BM25 (keyword) ∥ FAISS (vector) — run in PARALLEL
│     BM25 ∥ FAISS      │     Reciprocal Rank Fusion (k=60) → top-20 candidates
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  5. Cross-Encoder     │──── ms-marco-MiniLM-L-6-v2 rescores relevance
│     Reranker          │     20 candidates → top-5 most relevant
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  6. Context Builder   │──── Deduplicate by hash, sort by score, trim to 3000 tokens
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  7. LLM Generation    │──── GPT-3.5 (temp=0) with intent-specific prompts
│     + Streaming SSE   │     Real-time token streaming via Server-Sent Events
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  8. Hallucination     │──── 4 layers: temp=0 + strict prompt + phrase detection
│     Detection         │     + source verification → confidence score (0.0-1.0)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  9. Cache + DB Log    │──── Redis cache (1hr TTL) + PostgreSQL query logging
│     (fire-and-forget) │     Non-blocking — never slows down the response
└──────────┬───────────┘
           │
           ▼
   Answer + Sources + Confidence + Intent
   (streamed token-by-token to the browser)
```

---

## Tech Stack

```
Backend       Python 3.12 · FastAPI · Uvicorn · Pydantic 2.5
LLM           LangChain · OpenAI GPT-3.5-turbo · text-embedding-3-small
Retrieval     FAISS (vector) + BM25 (keyword) + Reciprocal Rank Fusion
Reranking     sentence-transformers · cross-encoder/ms-marco-MiniLM-L-6-v2
Cache         Redis 7 (semantic cache, 1hr TTL, local dict fallback)
Database      PostgreSQL 15 · SQLAlchemy 2.0 async · asyncpg
Auth          JWT (PyJWT) · bcrypt · rate limiting (slowapi)
Evaluation    Custom RAGAS (4 metrics) · MLflow experiment tracking
Observability LangSmith (optional) · structured logging
Frontend      Vanilla HTML/CSS/JS · SSE streaming · dark theme
Deployment    Docker · Docker Compose · Nginx reverse proxy
Testing       pytest (90 tests, fully mocked, <2s)
```

---

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/PatelDeep223/finbot-financial-rag
cd finbot-financial-rag
chmod +x setup.sh
./setup.sh YOUR_OPENAI_API_KEY
# Open http://localhost:3000
```

### Manual Setup
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI API key

# Start server (Redis optional — falls back to local cache)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests (no API keys needed)
python -m pytest tests/ -v
```

### First Steps
```bash
# 1. Sign up
curl -X POST localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"deep","email":"deep@test.com","password":"mypass123"}'

# 2. Upload a financial PDF
curl -X POST localhost:8000/api/v1/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@earnings_report.pdf"

# 3. Ask a question
curl -X POST localhost:8000/api/v1/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the Q3 revenue?"}'
```

---

## API Endpoints

| Method | Endpoint | Auth | Rate Limit | Description |
|--------|----------|------|------------|-------------|
| `POST` | `/api/v1/auth/signup` | No | - | Register (returns JWT) |
| `POST` | `/api/v1/auth/login` | No | - | Login (returns JWT) |
| `GET` | `/api/v1/auth/me` | JWT | 30/min | Current user info |
| `POST` | `/api/v1/query` | JWT | 10/min | Ask a question (full response) |
| `POST` | `/api/v1/query/stream` | JWT | 10/min | Ask a question (streaming SSE) |
| `POST` | `/api/v1/upload` | JWT | 5/min | Upload PDF or TXT |
| `GET` | `/api/v1/stats` | JWT | 30/min | System statistics |
| `GET` | `/api/v1/documents` | JWT | 30/min | List uploaded documents |
| `GET` | `/api/v1/history/{id}` | JWT | 30/min | Conversation history |
| `DELETE` | `/api/v1/history/{id}` | JWT | 30/min | Clear conversation |
| `POST` | `/api/v1/evaluate` | JWT | 3/min | RAGAS evaluation (4 metrics) |
| `GET` | `/health` | No | - | Health check |

**Auth:** JWT auto-disabled until first user signs up (dev mode).

---

## How the Advanced RAG Works

### Hybrid Retrieval (BM25 + FAISS + RRF)
```
Query: "EPS $2.15"

BM25 (keyword):  Finds exact "$2.15" string match       → ranked list A
FAISS (vector):  Finds "earnings per share" semantically → ranked list B

Reciprocal Rank Fusion:
  score(doc) = Σ 1/(k + rank) across both lists, k=60
  → Documents appearing in BOTH lists score highest
  → Output: top-20 candidates (best of both worlds)
```

### Cross-Encoder Reranking
```
Input:  20 candidates from hybrid search
Model:  ms-marco-MiniLM-L-6-v2 (86ms/pair on CPU)
Method: Score each (query, document) pair independently
Output: Top-5 documents sorted by true relevance

Why: FAISS retrieval is fast but approximate.
     Reranker is slower but far more precise.
```

### Anti-Hallucination (4 Layers)
```
Layer 1: temperature=0           → deterministic, no creative output
Layer 2: Strict financial prompts → "ONLY use provided context, never fabricate"
Layer 3: Uncertainty detection    → "I think", "probably" → lower confidence
Layer 4: Source verification      → no sources cited = -0.4 confidence penalty

Result: confidence_score (0.0-1.0) on every response
```

### Intent-Specific Prompts
```
"What is revenue?"              → factual   → strict data extraction prompt
"Compare Q3 vs Q2"             → comparison → side-by-side analysis prompt
"Summarize the report"         → summary    → key metrics + outlook prompt
"What are the risk factors?"   → risk       → structured risk analysis prompt
"Hello"                        → off_topic  → instant rejection (no LLM call)
```

---

## Embedding Model Comparison

| Metric | text-embedding-ada-002 | text-embedding-3-small | text-embedding-3-large |
|--------|----------------------|----------------------|----------------------|
| **Cost** | $0.10 / 1M tokens | $0.02 / 1M tokens | $0.13 / 1M tokens |
| **Cost vs ada-002** | baseline | **5x cheaper** | 1.3x more |
| **MTEB Score** | 61.0 | 62.3 | 64.6 |
| **MIRACL Avg** | 31.4 | 44.0 | 54.9 |
| **Dimensions** | 1536 (fixed) | 1536 (adjustable) | 3072 (adjustable) |
| **Best For** | Legacy | **Cost-optimized RAG** | Max accuracy |

FinBot uses **text-embedding-3-small** — 5x cheaper with better quality.

---

## Database Schema

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `users` | Authentication | username, email, hashed_password, is_active |
| `documents` | Upload tracking | filename, file_size_bytes, chunks_created |
| `conversations` | Chat history | session_id, role, content, confidence, sources (JSON) |
| `query_logs` | Analytics | question, answer, confidence, from_cache, response_time_ms |

---

## Project Structure

```
finbot-financial-rag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py            # 8 endpoints (query, stream, upload, stats, docs, history, evaluate)
│   │   │   └── auth.py              # 3 endpoints (signup, login, me)
│   │   ├── core/
│   │   │   ├── config.py            # 27 settings via Pydantic
│   │   │   └── security.py          # JWT + bcrypt + rate limiting
│   │   ├── db/
│   │   │   ├── __init__.py          # Async SQLAlchemy engine
│   │   │   └── service.py           # 12 DB operations with graceful fallback
│   │   ├── models/
│   │   │   ├── schemas.py           # 14 Pydantic request/response models
│   │   │   └── database.py          # 4 ORM models (User, Document, Conversation, QueryLog)
│   │   ├── rag/
│   │   │   └── pipeline.py          # 10-stage RAG pipeline + streaming
│   │   └── services/
│   │       ├── router.py            # Intent classification (5 types)
│   │       ├── rewriter.py          # Query expansion
│   │       ├── hybrid_retriever.py  # BM25 + FAISS + RRF
│   │       ├── reranker.py          # Cross-encoder (ms-marco)
│   │       ├── context_builder.py   # Dedup + sort + trim
│   │       └── evaluator.py         # RAGAS evaluation (4 metrics)
│   ├── tests/                       # 90 tests (fully mocked)
│   ├── experiments/                 # MLflow hyperparameter tuning
│   └── requirements.txt
├── frontend/
│   └── index.html                   # Dark-themed SPA with SSE streaming
├── docker-compose.yml               # 4 services: backend + redis + postgres + nginx
└── setup.sh                         # One-command Docker setup
```

---

## Environment Variables

```env
# Required
OPENAI_API_KEY=your-key-here

# Optional (all have sensible defaults)
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
JWT_SECRET=your-secret-here
CHUNK_SIZE=500
TOP_K_RESULTS=5
TOP_K_RETRIEVAL=20
MAX_CONTEXT_TOKENS=3000
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LANGCHAIN_TRACING_V2=false
```

---

## Running Tests

```bash
cd backend && source venv/bin/activate
python -m pytest tests/ -v    # 90 tests, no API keys needed, <2s
```

Tests cover: all API endpoints, pipeline components, hallucination detector, DB graceful fallback, auth (signup/login/JWT), Pydantic schemas, all 5 RAG services, and RAGAS evaluation.

---

## Connect

**Deep Patel** — Python Backend & GenAI Engineer

[![GitHub](https://img.shields.io/badge/GitHub-PatelDeep223-181717?logo=github)](https://github.com/PatelDeep223)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/deep-patel-a14848251/)

---

<p align="center">
  <sub>Built with FastAPI, LangChain, and a lot of financial document analysis.</sub>
</p>
