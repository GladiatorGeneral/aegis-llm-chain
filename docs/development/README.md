# Development Guide

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- Git

### Initial Setup

1. **Clone the repository**
```powershell
git clone <repository-url>
cd aegis-llm-chain
```

2. **Set up Python environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements\dev.txt
```

3. **Set up Frontend**
```powershell
cd frontend
npm install
cd ..
```

4. **Create environment file**
```powershell
Copy-Item .env.example .env
# Edit .env with your configuration
```

5. **Start services**
```powershell
docker-compose -f infrastructure\docker\docker-compose.yml up -d
```

## Project Structure

```
aegis-llm-chain/
├── backend/            # FastAPI backend
│   ├── src/           # Source code
│   ├── tests/         # Tests
│   └── requirements/  # Dependencies
├── frontend/          # Next.js frontend
│   └── src/           # Source code
├── infrastructure/    # Docker, K8s, Terraform
├── config/           # Configuration files
├── docs/             # Documentation
└── scripts/          # Utility scripts
```

## Development Workflow

### Backend Development

**Run backend locally:**
```powershell
cd backend
uvicorn src.main:app --reload
```

**Run tests:**
```powershell
pytest backend\tests -v
```

**Code formatting:**
```powershell
black backend\src
isort backend\src
```

### Frontend Development

**Run frontend locally:**
```powershell
cd frontend
npm run dev
```

**Type check:**
```powershell
npm run type-check
```

**Lint:**
```powershell
npm run lint
```

## Testing

### Backend Tests
```powershell
# Run all tests
pytest backend\tests

# Run with coverage
pytest backend\tests --cov=backend\src

# Run specific test file
pytest backend\tests\test_api\test_cognitive.py
```

### Frontend Tests
```powershell
# Unit tests
npm run test

# E2E tests
npm run test:e2e
```

## Debugging

### VS Code Debug Configurations

**Backend API**
- Set breakpoints in Python code
- Press F5 and select "Backend API"

**Frontend**
- Set breakpoints in TypeScript code
- Press F5 and select "Frontend Dev"

**Full Stack**
- Press F5 and select "Full Stack"

## Common Tasks

### Add a new API endpoint

1. Create route in `backend/src/api/v1/`
2. Add business logic
3. Update API documentation
4. Add tests

### Add a new React component

1. Create component in `frontend/src/components/`
2. Add TypeScript types
3. Export from index
4. Add tests

### Add a new model

1. Update `config/models/model-registry.yaml`
2. Add model metadata in `backend/src/models/registry.py`
3. Test model loading

## Best Practices

### Code Style
- Follow PEP 8 for Python
- Use ESLint rules for TypeScript
- Write descriptive commit messages

### Security
- Never commit secrets
- Use environment variables
- Validate all inputs
- Sanitize outputs

### Performance
- Use async/await for I/O
- Implement caching where appropriate
- Monitor database queries
- Optimize bundle size

## Troubleshooting

### Common Issues

**Port already in use:**
```powershell
# Find process using port
netstat -ano | findstr :8000
# Kill process
taskkill /PID <pid> /F
```

**Module not found:**
```powershell
# Reinstall dependencies
pip install -r backend\requirements\dev.txt
```

**Docker issues:**
```powershell
# Restart Docker
docker-compose down
docker-compose up -d
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Python Best Practices](https://docs.python-guide.org/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
