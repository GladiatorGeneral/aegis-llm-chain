# 🚀 AEGIS Model Organization System

## Overview

Transform your scattered model inference code into a **production-ready, enterprise-grade system** with centralized management, automatic routing, and comprehensive monitoring.

## ✨ What This System Does

**Before:**
```python
# Hard-coded, manual, scattered
client = InferenceClient(token="hf_...")
completion = client.chat.completions.create(
    model="deepcogito/cogito-671b-v2.1",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**After:**
```python
# Organized, automatic, scalable
completion = await inference_client.chat_completion(
    model_key="cogito-671b",  # Clean key
    messages=[{"role": "user", "content": "Hello"}]
)
```

## 📦 System Components

### 1. Model Registry (`backend/src/models/registry.py`)
**13 pre-configured models** organized by type and capability:

**Chat Models (10):**
- `cogito-671b` ⭐ - Your Cogito 67B (primary)
- `llama2-70b` - Llama 2 70B Chat
- `llama3-8b` - Llama 3 8B Instruct
- `mistral-7b` - Mistral 7B Instruct
- `mixtral-8x7b` - Mixtral 8x7B (MoE)
- `phi-3-mini` - Phi-3 Mini (lightweight)
- `phi-3-medium` - Phi-3 Medium
- `codellama-34b` - CodeLlama 34B (coding)
- `zephyr-7b-local` - Local 4-bit
- `mistral-7b-local` - Local 4-bit

**Embedding Models (3):**
- `bge-large` - BGE Large English
- `gte-large` - GTE Large
- `e5-large` - E5 Large v2

### 2. Unified Inference Client (`backend/src/models/inference_client.py`)
Handles both API and local inference automatically:
- ✅ Hugging Face Inference API
- ✅ Local GPU inference (Transformers)
- ✅ 4-bit/8-bit quantization
- ✅ Model caching
- ✅ Usage tracking

### 3. REST API (`backend/src/api/v1/models.py`)
9 production-ready endpoints:
```
GET    /api/v1/models/                     - List models
GET    /api/v1/models/{key}                - Model info
POST   /api/v1/models/{key}/chat           - Chat
POST   /api/v1/models/{key}/completion     - Complete
POST   /api/v1/models/{key}/embedding      - Embed
GET    /api/v1/models/{key}/health         - Health
GET    /api/v1/models/search/query         - Search
GET    /api/v1/models/local/loaded         - Loaded
DELETE /api/v1/models/local/{key}          - Unload
```

## 🎯 Quick Start

### 1. Setup
```bash
# Set your token
export HF_TOKEN=your_hugging_face_token

# Install dependencies (if needed)
pip install -r backend/requirements/base.txt
```

### 2. Test System
```bash
# Run tests
python examples/test_model_system.py

# Expected output:
# ✅ Registry initialized with 13 models
# ✅ All tests passed!
```

### 3. Run Examples
```bash
# Run 7 comprehensive examples
python examples/model_inference_examples.py
```

### 4. Start API Server
```bash
cd backend/src
python main.py

# Server starts on http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Test API
```powershell
# Test endpoints
.\test-model-api.ps1
```

## 💻 Usage Examples

### Basic Chat
```python
from models.inference_client import inference_client

result = await inference_client.chat_completion(
    model_key="cogito-671b",
    messages=[{"role": "user", "content": "What is AI?"}],
    temperature=0.7,
    max_tokens=100
)

print(result['content'])
print(f"Tokens: {result['usage']['total_tokens']}")
```

### Multi-turn Conversation
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses qubits..."},
    {"role": "user", "content": "How is it different?"}
]

result = await inference_client.chat_completion(
    model_key="mistral-7b",
    messages=messages
)
```

### Compare Models
```python
models = ["cogito-671b", "mistral-7b", "phi-3-mini"]

for model_key in models:
    result = await inference_client.chat_completion(
        model_key=model_key,
        messages=[{"role": "user", "content": "Write a haiku"}]
    )
    print(f"{model_key}: {result['content']}")
```

### Generate Embeddings
```python
result = await inference_client.embedding(
    model_key="bge-large",
    texts=["AI is amazing", "Machine learning rocks"]
)

print(f"Dimension: {result['dimension']}")
print(f"Embeddings: {result['embeddings']}")
```

### Browse Models
```python
# Get all models
models = inference_client.get_available_models()

# Filter by type
chat_models = [m for m in models if m['type'] == 'chat']

# Search
from models.registry import model_registry
coding_models = model_registry.search_models("coding")
```

## 🌐 API Examples

### List Models
```bash
curl http://localhost:8000/api/v1/models/
```

### Get Model Info
```bash
curl http://localhost:8000/api/v1/models/cogito-671b
```

### Chat Completion
```bash
curl -X POST http://localhost:8000/api/v1/models/cogito-671b/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Search Models
```bash
curl "http://localhost:8000/api/v1/models/search/query?q=coding"
```

## 🔧 Adding Your Own Models

Edit `backend/src/models/registry.py`:

```python
self._registry["your-model"] = ModelConfig(
    model_id="username/model-name",
    model_type=ModelType.CHAT,
    provider=ModelProvider.HUGGINGFACE,
    name="Your Model Name",
    description="Model description",
    context_length=4096,
    max_tokens=1024,
    supported_tasks=["chat", "qa"],
    api_key_env="HF_TOKEN"
)
```

Then use it:
```python
result = await inference_client.chat_completion(
    model_key="your-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 📊 Benefits

| Feature | Before | After |
|---------|--------|-------|
| Model IDs | ❌ Hard-coded | ✅ Clean keys |
| Switching | ❌ Complex | ✅ Easy |
| Management | ❌ Scattered | ✅ Centralized |
| Discovery | ❌ None | ✅ Search |
| Tracking | ❌ Manual | ✅ Automatic |
| Local inference | ❌ None | ✅ Supported |
| API | ❌ None | ✅ REST API |
| Monitoring | ❌ None | ✅ Health checks |

## 📁 Files Structure

```
backend/src/
├── models/
│   ├── registry.py           # ✅ 13 models configured
│   ├── inference_client.py   # ✅ API + local inference
│   └── ...
├── api/v1/
│   ├── models.py            # ✅ 9 REST endpoints
│   └── ...
└── main.py                  # ✅ Startup integration

examples/
├── model_inference_examples.py  # ✅ 7 examples
└── test_model_system.py         # ✅ Test suite

docs/
└── MODEL_ORGANIZATION_GUIDE.md  # ✅ Full guide

test-model-api.ps1              # ✅ API tester
MODEL_SYSTEM_COMPLETE.md        # ✅ Summary
```

## 🧪 Testing

```bash
# Test system
python examples/test_model_system.py
# ✅ 2/2 tests passed

# Test API
.\test-model-api.ps1
# ✅ API endpoints working

# Run examples
python examples/model_inference_examples.py
# ✅ 7 examples complete
```

## 📚 Documentation

- **Quick Start**: This file
- **Full Guide**: `docs/MODEL_ORGANIZATION_GUIDE.md`
- **Complete Summary**: `MODEL_SYSTEM_COMPLETE.md`
- **API Docs**: http://localhost:8000/docs (when server running)

## 🎉 Key Features

1. **🎨 Organized** - Central model registry
2. **⚡ Fast** - Automatic caching
3. **🔧 Flexible** - Easy to extend
4. **📊 Monitored** - Usage tracking
5. **🏥 Healthy** - Health checks
6. **🔍 Discoverable** - Search & filter
7. **💾 Efficient** - Local caching
8. **🌐 Scalable** - REST API ready

## 💡 Pro Tips

1. **Model Selection**: Use `cogito-671b` for complex reasoning, `phi-3-mini` for speed
2. **Temperature**: 0.7 for balanced, 0.9 for creative, 0.3 for factual
3. **Tokens**: Set `max_tokens` to avoid long responses
4. **Local Models**: Download to `./models/` for GPU inference
5. **Caching**: Models load once, reuse many times
6. **Health**: Check before production use

## 🚀 Next Steps

1. ✅ Set `HF_TOKEN` environment variable
2. ✅ Run tests: `python examples/test_model_system.py`
3. ✅ Try examples: `python examples/model_inference_examples.py`
4. ✅ Start server: `python backend/src/main.py`
5. ✅ Build your app!

## 📞 Quick Reference

```python
# Import
from models.inference_client import inference_client
from models.registry import model_registry

# Chat
result = await inference_client.chat_completion(
    model_key="cogito-671b",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Browse
models = inference_client.get_available_models()

# Search
results = model_registry.search_models("coding")

# Filter
chat_models = model_registry.list_models(ModelType.CHAT)
```

## ✅ System Status

- ✅ **13 Models** configured and ready
- ✅ **9 API Endpoints** production-ready
- ✅ **7 Examples** working
- ✅ **2 Test Suites** passing
- ✅ **Complete Documentation** available

**Your model system is ready to use! 🎉**

---

*For detailed documentation, see `docs/MODEL_ORGANIZATION_GUIDE.md`*
