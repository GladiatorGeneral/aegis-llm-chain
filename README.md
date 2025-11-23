# Main README for AGI Platform

# AGI Platform

A comprehensive, security-first AI platform with unified cognitive capabilities, model management, and workflow orchestration.

## Features

- **Universal Cognitive Engine**: Unified interface for generation, analysis, and reasoning
- **Model Management**: Secure deployment and orchestration of AI models
- **Workflow Orchestration**: Build and execute complex multi-step AI workflows
- **Security-First Architecture**: Multi-layered security with input validation, output filtering, and rate limiting
- **Scalable Infrastructure**: Docker and Kubernetes support for production deployment

## Architecture

```
Frontend (Next.js) <--> Backend API (FastAPI) <--> Models & Engines
                              |
                    +---------+---------+
                    |                   |
              PostgreSQL            Redis Cache
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- Git

### Setup

1. **Clone and setup:**
```powershell
git clone <repository-url>
cd aegis-llm-chain
.\scripts\dev\setup-environment.ps1
```

2. **Start services:**
```powershell
.\scripts\dev\start-services.ps1
```

3. **Start backend:**
```powershell
cd backend
uvicorn src.main:app --reload
```

4. **Start frontend:**
```powershell
cd frontend
npm run dev
```

5. **Open browser:**
```
http://localhost:3000
```

## Project Structure

```
aegis-llm-chain/
├── .vscode/              # VS Code configuration
├── .devcontainer/        # Development container
├── backend/              # FastAPI backend
│   ├── src/             # Source code
│   │   ├── api/         # API endpoints
│   │   ├── core/        # Core logic
│   │   ├── engines/     # AI engines
│   │   ├── models/      # Model management
│   │   └── workflows/   # Workflow orchestration
│   ├── tests/           # Tests
│   └── requirements/    # Python dependencies
├── frontend/            # Next.js frontend
│   └── src/            # Source code
├── infrastructure/      # Docker, K8s, Terraform
├── config/             # Configuration files
├── docs/               # Documentation
├── experiments/        # Jupyter notebooks
└── scripts/           # Utility scripts
```

## Documentation

- [Architecture](docs/architecture/README.md)
- [API Documentation](docs/api/README.md)
- [Security](docs/security/README.md)
- [Development Guide](docs/development/README.md)

## Key Components

### Backend (FastAPI)

- **Cognitive Engine**: Universal AI processing
- **Model Registry**: Model management and deployment
- **Workflow Orchestrator**: Multi-step workflow execution
- **Security Layer**: Input/output validation and filtering

### Frontend (Next.js)

- **Dashboard**: System monitoring and management
- **Model Manager**: Model deployment interface
- **Workflow Builder**: Visual workflow creation
- **Admin Panel**: System administration

### Security Features

- JWT-based authentication
- Role-based access control (RBAC)
- Rate limiting
- Input validation
- Output filtering (PII redaction)
- Content safety checks
- Audit logging

## Development

### Running Tests

```powershell
# Backend tests
.\scripts\dev\run-tests.ps1 -Coverage

# Frontend tests
cd frontend
npm run test
```

### Code Formatting

```powershell
# Python
black backend\src
isort backend\src

# TypeScript
cd frontend
npm run lint
```

## Deployment

### Docker Compose

```powershell
docker-compose -f infrastructure\docker\docker-compose.yml up -d
```

### Kubernetes

```powershell
kubectl apply -f infrastructure\kubernetes\base\
```

## Configuration

Configuration files are located in the `config/` directory:

- `environments/` - Environment-specific configs
- `security/` - Security settings
- `models/` - Model registry and settings

## Environment Variables

Create a `.env` file in the root directory:

```env
# API
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agi_platform

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256

# Models
HF_TOKEN=your-huggingface-token
MODEL_CACHE_DIR=./model_cache
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Security

For security concerns, please see [Security Documentation](docs/security/README.md).

## License

[Your License Here]

## Support

For questions and support:
- Documentation: `docs/`
- Issues: GitHub Issues
- Email: support@agi-platform.com

## Roadmap

- [ ] Additional model providers (Anthropic, Cohere)
- [ ] Advanced workflow features
- [ ] Multi-modal support
- [ ] Enhanced monitoring and observability
- [ ] Production-ready Kubernetes configurations
- [ ] CI/CD pipelines

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)
- [Hugging Face](https://huggingface.co/)
- [PostgreSQL](https://www.postgresql.org/)
- [Redis](https://redis.io/)
