#!/bin/bash

# FinBot — One Command Setup Script
# Usage: ./setup.sh YOUR_OPENAI_API_KEY

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════╗"
echo "║   💹 FinBot — Financial RAG System       ║"
echo "║   Built by Deep Patel                    ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# Check API key
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: ./setup.sh YOUR_OPENAI_API_KEY${NC}"
    echo ""
    echo "Get your key at: https://platform.openai.com/api-keys"
    exit 1
fi

OPENAI_KEY=$1

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Install from https://docker.com${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker found${NC}"

# Create .env file
echo -e "${BLUE}⚙️  Creating environment config...${NC}"
cat > .env << EOF
OPENAI_API_KEY=${OPENAI_KEY}
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://finbot:finbot123@db:5432/finbot
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
TEMPERATURE=0.0
EOF

# Copy .env to backend
cp .env backend/.env
echo -e "${GREEN}✅ Environment configured${NC}"

# Create data directories
mkdir -p backend/data/documents backend/data/vectorstore
echo -e "${GREEN}✅ Data directories created${NC}"

# Build and start
echo -e "${BLUE}🐳 Building Docker containers...${NC}"
docker-compose down --remove-orphans 2>/dev/null || true

if docker compose version &> /dev/null 2>&1; then
    docker compose up --build -d
else
    docker-compose up --build -d
fi

# Wait for backend
echo -e "${BLUE}⏳ Waiting for services to start...${NC}"
sleep 8

# Health check
HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "")
if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Backend API is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Backend may still be starting...${NC}"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   🚀 FinBot is RUNNING!                  ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                          ║${NC}"
echo -e "${GREEN}║  🌐 Frontend:  http://localhost:3000     ║${NC}"
echo -e "${GREEN}║  ⚡ API:       http://localhost:8000     ║${NC}"
echo -e "${GREEN}║  📚 API Docs:  http://localhost:8000/docs║${NC}"
echo -e "${GREEN}║                                          ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Next steps:                             ║${NC}"
echo -e "${GREEN}║  1. Open http://localhost:3000           ║${NC}"
echo -e "${GREEN}║  2. Upload a financial PDF               ║${NC}"
echo -e "${GREEN}║  3. Start asking questions!              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Logs: docker-compose logs -f backend${NC}"
