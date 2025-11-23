# 🚀 Quick Deployment Guide

Get AEGIS LLM Chain running in production in 5 minutes.

## Prerequisites

- Docker & Docker Compose installed
- HuggingFace account with token
- PostgreSQL & Redis (included in Docker setup)

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/aegis-llm-chain.git
cd aegis-llm-chain
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` and set **at minimum**:

```bash
HF_TOKEN=your_huggingface_token_here
SECRET_KEY=$(openssl rand -hex 32)
```

Get your HF token: https://huggingface.co/settings/tokens (Read permission only)

### 3. Deploy

**Linux/Mac:**
```bash
chmod +x scripts/deploy/production.sh
./scripts/deploy/production.sh
```

**Windows:**
```powershell
.\scripts\deploy\production.ps1
```

**Manual Docker:**
```bash
export HF_TOKEN="your_token_here"
cd infrastructure/docker
docker-compose up -d
```

### 4. Verify

```bash
# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs

# Check services
docker-compose ps
```

## Available Services

| Service | URL | Description |
|---------|-----|-------------|
| Backend API | http://localhost:8000 | REST API endpoints |
| API Docs | http://localhost:8000/docs | Interactive Swagger UI |
| Frontend | http://localhost:3000 | Next.js web interface |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache & queue |
| pgAdmin | http://localhost:5050 | Database admin |

## GitHub Deployment (CI/CD)

### 1. Add GitHub Secrets

Navigate to: **Settings → Secrets and variables → Actions**

Required secrets:
- `HF_TOKEN` - Your HuggingFace token
- `SECRET_KEY` - Generate with `openssl rand -hex 32`
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub token

### 2. Trigger Deployment

```bash
# Push to main → deploys to staging
git push origin main

# Push to production → deploys to production
git push origin production

# Manual trigger
gh workflow run deploy.yml
```

## Common Commands

```bash
# View logs
docker-compose logs -f backend

# Restart service
docker-compose restart backend

# Stop all
docker-compose down

# Update & restart
docker-compose pull
docker-compose up -d

# Check status
docker-compose ps
```

## Models Available

13 pre-configured models accessible via API:

**Chat Models (10):**
- cogito-671b, llama2-70b, llama3-8b, mistral-7b, mixtral-8x7b
- phi-3-mini, phi-3-medium, codellama-34b
- zephyr-7b-local, mistral-7b-local

**Embedding Models (3):**
- bge-large, gte-large, e5-large

## Quick API Test

```bash
# List models
curl http://localhost:8000/api/v1/models/

# Chat completion
curl -X POST http://localhost:8000/api/v1/models/cogito-671b/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

## Troubleshooting

**Token not detected:**
```bash
export HF_TOKEN="your_token"
docker-compose restart backend
```

**Port already in use:**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

**Container won't start:**
```bash
docker-compose logs backend
docker-compose down
docker-compose up -d
```

## Full Documentation

- **Complete Deployment Guide**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Model Organization**: [docs/MODEL_ORGANIZATION_GUIDE.md](docs/MODEL_ORGANIZATION_GUIDE.md)
- **Architecture**: [docs/architecture/README.md](docs/architecture/README.md)
- **Security**: [docs/security/README.md](docs/security/README.md)

## Need Help?

1. Check logs: `docker-compose logs -f`
2. Review health: `curl http://localhost:8000/health`
3. Open issue: https://github.com/yourusername/aegis-llm-chain/issues

---

**Ready to deploy?** Run `./scripts/deploy/production.sh` and you're good to go! 🚀
