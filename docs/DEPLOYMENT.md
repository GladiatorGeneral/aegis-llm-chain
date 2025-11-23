# AEGIS LLM Chain - Production Deployment Guide

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [GitHub Secrets Configuration](#github-secrets-configuration)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Required Services

- **Docker** (v24.0+) and **Docker Compose** (v2.0+)
- **PostgreSQL** (v15+)
- **Redis** (v7+)
- **Node.js** (v18+) for frontend
- **Python** (v3.11+) for backend

### Required Tokens & Keys

1. **HuggingFace Token** (Read permission)
   - Get from: https://huggingface.co/settings/tokens
   - Required for accessing inference API

2. **GitHub Personal Access Token** (for CI/CD)
   - Scopes: `repo`, `workflow`, `write:packages`

3. **Docker Hub Account** (for image hosting)

---

## 🌍 Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/aegis-llm-chain.git
cd aegis-llm-chain
```

### 2. Create Environment File

```bash
cp .env.example .env
```

### 3. Configure Environment Variables

Edit `.env` with your actual values:

```bash
# REQUIRED: HuggingFace Token
HF_TOKEN=hf_your_token_here

# Security (generate secure random strings)
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Database
DATABASE_URL=postgresql://postgres:secure_password@db:5432/agi_platform
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_URL=redis://redis:6379/0

# Environment
ENVIRONMENT=production
LOG_LEVEL=info

# Frontend
FRONTEND_URL=https://your-domain.com
NEXT_PUBLIC_API_URL=https://api.your-domain.com
```

---

## 🐳 Docker Deployment

### Production Deployment

#### 1. Build Images

```bash
cd infrastructure/docker
docker-compose build
```

#### 2. Start Services

```bash
# Set HF_TOKEN first
export HF_TOKEN="your_token_here"

# Start all services
docker-compose up -d
```

#### 3. Verify Deployment

```bash
# Check running containers
docker-compose ps

# Check backend health
curl http://localhost:8000/health

# Check backend logs
docker-compose logs -f backend

# Check all service logs
docker-compose logs -f
```

#### 4. Access Services

- **Backend API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **pgAdmin**: http://localhost:5050

### Service Management

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (CAUTION: data loss)
docker-compose down -v

# Restart specific service
docker-compose restart backend

# View logs
docker-compose logs -f backend

# Scale backend workers
docker-compose up -d --scale backend=3

# Update services (pull latest images)
docker-compose pull
docker-compose up -d
```

---

## 🔐 GitHub Secrets Configuration

### Required Secrets

Navigate to your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

#### Core Secrets

| Secret Name | Description | Example |
|------------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | `hf_xxxxxxxxxxxxx` |
| `SECRET_KEY` | Application secret key | `openssl rand -hex 32` |
| `DOCKER_USERNAME` | Docker Hub username | `your_username` |
| `DOCKER_PASSWORD` | Docker Hub password/token | `your_token` |

#### Database Secrets

| Secret Name | Description |
|------------|-------------|
| `STAGING_DATABASE_URL` | Staging database URL |
| `PRODUCTION_DATABASE_URL` | Production database URL |
| `STAGING_REDIS_URL` | Staging Redis URL |
| `PRODUCTION_REDIS_URL` | Production Redis URL |

#### Deployment Secrets

| Secret Name | Description |
|------------|-------------|
| `STAGING_HOST` | Staging server hostname |
| `STAGING_USERNAME` | SSH username for staging |
| `STAGING_SSH_KEY` | SSH private key for staging |
| `PRODUCTION_HOST` | Production server hostname |
| `PRODUCTION_USERNAME` | SSH username for production |
| `PRODUCTION_SSH_KEY` | SSH private key for production |

#### Monitoring Secrets (Optional)

| Secret Name | Description |
|------------|-------------|
| `SENTRY_DSN` | Sentry error tracking DSN |
| `SLACK_WEBHOOK` | Slack webhook for notifications |

### Setting Secrets Example

```bash
# Using GitHub CLI
gh secret set HF_TOKEN --body "hf_your_token_here"
gh secret set SECRET_KEY --body "$(openssl rand -hex 32)"
gh secret set DOCKER_USERNAME --body "your_username"
gh secret set DOCKER_PASSWORD --body "your_token"
```

---

## 🚀 CI/CD Pipeline

### Workflow Triggers

The deployment pipeline triggers on:

- **Push to `main`**: Deploys to staging
- **Push to `production`**: Deploys to production
- **Pull requests**: Runs tests only
- **Manual trigger**: Via GitHub Actions UI

### Pipeline Stages

#### 1. **Test Backend** (`test-backend`)
- Sets up Python 3.11
- Installs dependencies
- Runs pytest with coverage
- Uploads coverage to Codecov

#### 2. **Test Frontend** (`test-frontend`)
- Sets up Node.js 18
- Installs npm dependencies
- Runs linter
- Builds production bundle

#### 3. **Security Scan** (`security-scan`)
- Runs Trivy vulnerability scanner
- Uploads results to GitHub Security tab

#### 4. **Build and Push** (`build-and-push`)
- Builds Docker images for backend and frontend
- Pushes to Docker Hub with tags:
  - `latest` (for main branch)
  - `{branch}-{sha}` (for all branches)

#### 5. **Deploy Staging** (`deploy-staging`)
- SSHs into staging server
- Pulls latest code
- Updates Docker containers
- Runs health checks

#### 6. **Deploy Production** (`deploy-production`)
- Requires manual approval (GitHub environment protection)
- SSHs into production server
- Blue-green deployment strategy
- Health checks and rollback on failure
- Slack notification

### Manual Deployment

```bash
# Trigger workflow manually
gh workflow run deploy.yml

# Deploy specific branch
gh workflow run deploy.yml --ref production
```

---

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Check model availability
curl http://localhost:8000/api/v1/models/

# Test chat completion
curl -X POST http://localhost:8000/api/v1/models/cogito-671b/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### Logs Management

```bash
# View real-time logs
docker-compose logs -f backend

# Save logs to file
docker-compose logs --no-color > logs.txt

# View last 100 lines
docker-compose logs --tail=100 backend

# Filter by timestamp
docker-compose logs --since="2024-01-01" backend
```

### Performance Monitoring

#### Using Docker Stats

```bash
# Monitor resource usage
docker stats

# Monitor specific container
docker stats aegis-backend
```

#### Using Prometheus (if enabled)

```bash
# Access Prometheus metrics
curl http://localhost:9090/metrics
```

### Database Backups

```bash
# Backup PostgreSQL
docker-compose exec db pg_dump -U postgres agi_platform > backup.sql

# Restore from backup
cat backup.sql | docker-compose exec -T db psql -U postgres agi_platform
```

### Updates & Maintenance

```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Rebuild specific service
docker-compose build backend
docker-compose up -d backend

# Clean up old images
docker system prune -a
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **HF_TOKEN Not Detected**

**Symptom**: Warning about missing HuggingFace token

**Solution**:
```bash
# Check if token is set
echo $HF_TOKEN

# Set token before starting
export HF_TOKEN="your_token_here"
docker-compose up -d

# Or add to .env file
echo "HF_TOKEN=your_token_here" >> .env
```

#### 2. **Container Won't Start**

**Symptoms**: Container exits immediately

**Debug**:
```bash
# Check logs
docker-compose logs backend

# Check if port is already in use
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Inspect container
docker-compose ps
docker inspect aegis-backend
```

#### 3. **Database Connection Failed**

**Symptoms**: `psycopg2.OperationalError`

**Solution**:
```bash
# Check if database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Verify connection
docker-compose exec db psql -U postgres -c "SELECT 1"

# Reset database
docker-compose down
docker volume rm aegis-llm-chain_postgres-data
docker-compose up -d
```

#### 4. **Model Inference Failed**

**Symptoms**: 400/500 errors on `/api/v1/models/` endpoints

**Solution**:
```bash
# Verify HF_TOKEN is valid
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami

# Check model availability
curl http://localhost:8000/api/v1/models/

# Check backend logs for errors
docker-compose logs -f backend
```

#### 5. **High Memory Usage**

**Solution**:
```bash
# Limit container memory
docker-compose up -d --scale backend=1

# Check resource usage
docker stats

# Restart services
docker-compose restart backend
```

### Debug Mode

Enable debug logging:

```bash
# Update .env
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart backend

# View detailed logs
docker-compose logs -f backend
```

### Getting Help

1. **Check logs**: `docker-compose logs -f`
2. **Review health**: `curl http://localhost:8000/health`
3. **Test models**: Visit http://localhost:8000/docs
4. **GitHub Issues**: Report bugs with logs and reproduction steps

---

## 🔒 Security Best Practices

### 1. **Environment Variables**

- ✅ Never commit `.env` files to Git
- ✅ Use GitHub Secrets for CI/CD
- ✅ Rotate secrets regularly
- ✅ Use strong random keys (`openssl rand -hex 32`)

### 2. **Container Security**

- ✅ Run containers as non-root user (already configured)
- ✅ Use health checks (already configured)
- ✅ Keep images updated
- ✅ Scan for vulnerabilities (Trivy in CI/CD)

### 3. **Network Security**

- ✅ Use HTTPS in production (configure reverse proxy)
- ✅ Restrict CORS origins
- ✅ Enable rate limiting
- ✅ Use firewall rules

### 4. **Database Security**

- ✅ Use strong passwords
- ✅ Enable SSL connections
- ✅ Regular backups
- ✅ Restrict network access

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Model Registry Guide**: [docs/MODEL_ORGANIZATION_GUIDE.md](../MODEL_ORGANIZATION_GUIDE.md)
- **Architecture Overview**: [docs/architecture/README.md](../architecture/README.md)
- **Security Guide**: [docs/security/README.md](../security/README.md)

---

## 🆘 Quick Reference

### Start Everything
```bash
export HF_TOKEN="your_token"
cd infrastructure/docker
docker-compose up -d
```

### Check Status
```bash
docker-compose ps
curl http://localhost:8000/health
```

### View Logs
```bash
docker-compose logs -f backend
```

### Stop Everything
```bash
docker-compose down
```

### Update & Restart
```bash
docker-compose pull
docker-compose up -d
```

---

**Need help?** Open an issue on GitHub or check the troubleshooting section above.
