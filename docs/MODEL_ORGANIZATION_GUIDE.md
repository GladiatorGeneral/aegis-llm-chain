# Model Organization Quick Start Guide

## 🎯 Overview

The AEGIS LLM Chain now has a **comprehensive model organization system** that makes it easy to:

- ✅ Manage multiple models from one centralized registry
- ✅ Use both Hugging Face API and local inference seamlessly
- ✅ Switch between models with a simple key
- ✅ Monitor usage, costs, and performance
- ✅ Scale from lightweight to heavy models

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Set your Hugging Face token
export HF_TOKEN=your_hugging_face_token_here

# Install dependencies (if not already done)
pip install -r backend/requirements/base.txt
```

### 2. Your Original Code - Enhanced!

**BEFORE (Your Original Code):**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_...")

completion = client.chat.completions.create(
    model="deepcogito/cogito-671b-v2.1",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(completion.choices[0].message.content)
```

**AFTER (Organized & Better):**
```python
from models.inference_client import inference_client

completion = await inference_client.chat_completion(
    model_key="cogito-671b",  # ✅ Clean model key
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(completion['content'])
```

### 3. Available Models

**Chat Models:**
- `cogito-671b` - Cogito 67B v2.1 (Your primary model) ⭐
- `llama2-70b` - Llama 2 70B Chat
- `llama3-8b` - Llama 3 8B Instruct
- `mistral-7b` - Mistral 7B Instruct
- `mixtral-8x7b` - Mixtral 8x7B (MoE)
- `phi-3-mini` - Phi-3 Mini 4K (Lightweight)
- `phi-3-medium` - Phi-3 Medium 4K
- `codellama-34b` - CodeLlama 34B (Coding specialist)

**Embedding Models:**
- `bge-large` - BGE Large English
- `gte-large` - GTE Large
- `e5-large` - E5 Large v2

**Local Models:** (for GPU inference)
- `zephyr-7b-local` - Zephyr 7B (4-bit quantized)
- `mistral-7b-local` - Mistral 7B (4-bit quantized)

## 📖 Usage Examples

### Example 1: Basic Chat
```python
from models.inference_client import inference_client

completion = await inference_client.chat_completion(
    model_key="cogito-671b",
    messages=[
        {"role": "user", "content": "What is AI?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(completion['content'])
print(f"Tokens used: {completion['usage']['total_tokens']}")
```

### Example 2: Multi-turn Conversation
```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "Explain quantum computing"
    },
    {
        "role": "assistant",
        "content": "Quantum computing uses qubits..."
    },
    {
        "role": "user",
        "content": "How is it different from classical?"
    }
]

completion = await inference_client.chat_completion(
    model_key="llama3-8b",
    messages=messages
)
```

### Example 3: Text Completion
```python
completion = await inference_client.text_completion(
    model_key="mistral-7b",
    prompt="Write a poem about AI:",
    temperature=0.9,
    max_tokens=200
)

print(completion['content'])
```

### Example 4: Generate Embeddings
```python
texts = [
    "AI is transforming the world",
    "Machine learning learns from data"
]

result = await inference_client.embedding(
    model_key="bge-large",
    texts=texts
)

print(f"Dimension: {result['dimension']}")
print(f"Embeddings: {result['embeddings']}")
```

### Example 5: Compare Models
```python
models = ["cogito-671b", "mistral-7b", "phi-3-mini"]

for model_key in models:
    result = await inference_client.chat_completion(
        model_key=model_key,
        messages=[{"role": "user", "content": "Say hi"}]
    )
    print(f"{model_key}: {result['content']}")
```

### Example 6: Browse Models
```python
from models.inference_client import inference_client

# Get all available models
models = inference_client.get_available_models()

for model in models:
    print(f"{model['key']}: {model['name']}")
    print(f"  Type: {model['type']}")
    print(f"  Tasks: {model['supported_tasks']}")
    print(f"  Context: {model['context_length']}")
```

### Example 7: Search Models
```python
from models.registry import model_registry

# Search for coding models
coding_models = model_registry.search_models("coding")

for config in coding_models:
    print(f"{config.name}: {config.description}")
```

## 🌐 REST API Endpoints

### Start the Backend
```bash
cd backend/src
python main.py
```

### API Endpoints

**List Models:**
```bash
curl http://localhost:8000/api/v1/models/
```

**Get Model Info:**
```bash
curl http://localhost:8000/api/v1/models/cogito-671b
```

**Chat Completion:**
```bash
curl -X POST http://localhost:8000/api/v1/models/cogito-671b/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Text Completion:**
```bash
curl -X POST http://localhost:8000/api/v1/models/mistral-7b/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'
```

**Generate Embeddings:**
```bash
curl -X POST http://localhost:8000/api/v1/models/bge-large/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "How are you?"]
  }'
```

**Health Check:**
```bash
curl http://localhost:8000/api/v1/models/cogito-671b/health
```

**Search Models:**
```bash
curl "http://localhost:8000/api/v1/models/search/query?q=coding"
```

**Filter by Type:**
```bash
# Get only chat models
curl "http://localhost:8000/api/v1/models/?model_type=chat"

# Get embedding models
curl "http://localhost:8000/api/v1/models/?model_type=embedding"

# Filter by task
curl "http://localhost:8000/api/v1/models/?task=coding"
```

## 🔧 Adding Your Own Models

### Add to Registry

Edit `backend/src/models/registry.py`:

```python
# Add your model
self._registry["your-model-key"] = ModelConfig(
    model_id="your-username/your-model-name",
    model_type=ModelType.CHAT,
    provider=ModelProvider.HUGGINGFACE,
    name="Your Model Display Name",
    description="Description of your model",
    context_length=4096,
    max_tokens=1024,
    supported_tasks=["chat", "qa", "summarization"],
    api_key_env="HF_TOKEN"
)
```

### Use Your Model

```python
completion = await inference_client.chat_completion(
    model_key="your-model-key",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 🖥️ Local Model Inference

### Download Model

```bash
# Create models directory
mkdir -p ./models

# Download model (example)
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 ./models/mistral-7b-instruct-v0.2
```

### Configure Local Model

Already configured in registry:
- `zephyr-7b-local` - Ready for local 4-bit inference
- `mistral-7b-local` - Ready for local 4-bit inference

### Use Local Model

```python
# Automatically uses local inference if model is marked as local
completion = await inference_client.chat_completion(
    model_key="mistral-7b-local",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Model is loaded once, then cached
```

### Unload Model

```python
# Free up GPU memory
inference_client.unload_local_model("mistral-7b-local")
```

## 🧪 Run Examples

```bash
cd examples
python model_inference_examples.py
```

This will run 7 comprehensive examples:
1. ✅ Basic Chat Completion
2. ✅ Multi-turn Conversation
3. ✅ Model Comparison
4. ✅ Text Embeddings
5. ✅ Browse Available Models
6. ✅ Model Health Checks
7. ✅ Batch Processing

## 📊 Benefits

### Before (Your Original Code)
- ❌ Hard-coded model IDs
- ❌ Manual client initialization
- ❌ No model management
- ❌ No usage tracking
- ❌ Complex model switching

### After (Organized System)
- ✅ Clean model keys
- ✅ Automatic client routing
- ✅ Centralized registry
- ✅ Built-in usage tracking
- ✅ Easy model switching
- ✅ API and local inference
- ✅ Health checks
- ✅ Search and discovery

## 🎯 Key Features

1. **Centralized Management** - All models in one registry
2. **Dual Mode** - API and local inference seamlessly
3. **Easy Configuration** - Add models without code changes
4. **Cost Tracking** - Monitor token usage and costs
5. **Health Monitoring** - Check model accessibility
6. **Search & Discovery** - Find models by task/capability
7. **Caching** - Local models loaded once, reused
8. **Type Safety** - Proper TypeScript-like type hints

## 🚦 Next Steps

1. ✅ Set your `HF_TOKEN`
2. ✅ Run the examples: `python examples/model_inference_examples.py`
3. ✅ Start the API: `python backend/src/main.py`
4. ✅ Test with your Cogito model
5. ✅ Add more models as needed
6. ✅ Build your application!

## 📚 Documentation

- **Model Registry**: `backend/src/models/registry.py`
- **Inference Client**: `backend/src/models/inference_client.py`
- **API Routes**: `backend/src/api/v1/models.py`
- **Examples**: `examples/model_inference_examples.py`

## 🎉 You're Ready!

Your original code now runs through a **production-ready, scalable model management system**!

**Try it now:**
```python
from models.inference_client import inference_client

result = await inference_client.chat_completion(
    model_key="cogito-671b",
    messages=[{"role": "user", "content": "Hello AI!"}]
)

print(result['content'])
```

Happy coding! 🚀
