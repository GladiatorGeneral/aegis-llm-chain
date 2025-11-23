# API Documentation

## Base URL
```
Development: http://localhost:8000
Production: https://api.agi-platform.com
```

## Authentication

All API requests require authentication using JWT tokens.

### Get Token
```bash
POST /api/v1/auth/token

Body:
{
  "username": "user@example.com",
  "password": "your_password"
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Use Token
```bash
Authorization: Bearer <access_token>
```

## Endpoints

### Cognitive Engine

#### Process Task
```bash
POST /api/v1/cognitive/process

Headers:
  Authorization: Bearer <token>
  Content-Type: application/json

Body:
{
  "prompt": "Analyze this text...",
  "task_type": "analysis",
  "model_id": "llama-2-7b",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1024
  }
}

Response:
{
  "result": "Analysis results...",
  "model_used": "llama-2-7b",
  "task_type": "analysis",
  "metadata": {
    "tokens_used": 245,
    "latency_ms": 850
  }
}
```

#### Get Capabilities
```bash
GET /api/v1/cognitive/capabilities

Response:
{
  "capabilities": [
    "generation",
    "analysis",
    "reasoning",
    "classification",
    "summarization"
  ],
  "supported_models": ["llama-2-7b", "gpt-3.5-turbo"]
}
```

### Model Management

#### List Models
```bash
GET /api/v1/models/list

Response:
[
  {
    "model_id": "llama-2-7b",
    "name": "Llama 2 7B",
    "type": "generative",
    "size": "7B",
    "status": "available",
    "capabilities": ["generation", "chat"]
  }
]
```

#### Get Model Info
```bash
GET /api/v1/models/{model_id}

Response:
{
  "model_id": "llama-2-7b",
  "name": "Llama 2 7B",
  "type": "generative",
  "size": "7B",
  "status": "available",
  "capabilities": ["generation", "chat"]
}
```

#### Deploy Model
```bash
POST /api/v1/models/deploy

Body:
{
  "model_id": "mistral-7b",
  "deployment_config": {
    "gpu_count": 1,
    "memory_gb": 16
  }
}

Response:
{
  "status": "deploying",
  "model_id": "mistral-7b",
  "message": "Model deployment initiated"
}
```

### Workflow Management

#### Create Workflow
```bash
POST /api/v1/workflows/create

Body:
{
  "name": "Analysis Pipeline",
  "description": "Multi-step analysis workflow",
  "steps": [
    {
      "step_id": "step1",
      "task_type": "generation",
      "parameters": {"prompt": "Generate summary"}
    },
    {
      "step_id": "step2",
      "task_type": "analysis",
      "parameters": {"input": "${step1.output}"}
    }
  ]
}

Response:
{
  "workflow_id": "wf_12345",
  "status": "created",
  "message": "Workflow created successfully"
}
```

#### Execute Workflow
```bash
POST /api/v1/workflows/{workflow_id}/execute

Response:
{
  "workflow_id": "wf_12345",
  "status": "executing",
  "message": "Workflow execution started"
}
```

#### Get Workflow Status
```bash
GET /api/v1/workflows/{workflow_id}

Response:
{
  "workflow_id": "wf_12345",
  "name": "Analysis Pipeline",
  "status": "running",
  "created_at": "2025-11-23T12:00:00Z",
  "steps_completed": 1,
  "total_steps": 2
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid input detected"
}
```

### 401 Unauthorized
```json
{
  "detail": "Could not validate credentials"
}
```

### 429 Too Many Requests
```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 3600
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error occurred"
}
```

## Rate Limits

- **Default**: 60 requests/minute
- **Cognitive Processing**: 100 requests/hour
- **Model Deployment**: 10 requests/hour
- **Workflow Execution**: 50 requests/hour

## Webhooks

Subscribe to events:
```bash
POST /api/v1/webhooks/subscribe

Body:
{
  "url": "https://your-app.com/webhook",
  "events": ["workflow.completed", "model.deployed"]
}
```
