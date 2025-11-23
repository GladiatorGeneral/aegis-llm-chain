"""
Model Management API Routes
Provides REST API endpoints for model management and inference
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.inference_client import inference_client
from models.registry import model_registry, ModelType

logger = logging.getLogger(__name__)

router = APIRouter()

# ==================== Request/Response Models ====================

class ChatMessage(BaseModel):
    """Chat message structure"""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    """Chat completion request"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")

class TextCompletionRequest(BaseModel):
    """Text completion request"""
    prompt: str = Field(..., description="Input prompt")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")

class EmbeddingRequest(BaseModel):
    """Embedding request"""
    texts: List[str] = Field(..., description="List of texts to embed")

# ==================== API Endpoints ====================

@router.get("/")
async def list_models(
    model_type: Optional[str] = None,
    task: Optional[str] = None,
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """List all available models with optional filtering"""
    try:
        models = inference_client.get_available_models()
        
        # Apply filters
        if model_type:
            models = [m for m in models if m["type"] == model_type]
        
        if task:
            models = [m for m in models if task in m["supported_tasks"]]
        
        if provider:
            models = [m for m in models if m["provider"] == provider]
        
        return {
            "count": len(models),
            "models": models,
            "filters": {
                "model_type": model_type,
                "task": task,
                "provider": provider
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/{model_key}")
async def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get detailed information about a specific model"""
    model_config = model_registry.get_model(model_key)
    if not model_config:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_key}' not found in registry"
        )
    
    return {
        "key": model_key,
        "name": model_config.name,
        "model_id": model_config.model_id,
        "type": model_config.model_type.value,
        "provider": model_config.provider.value,
        "description": model_config.description,
        "context_length": model_config.context_length,
        "max_tokens": model_config.max_tokens,
        "supported_tasks": model_config.supported_tasks,
        "is_local": model_config.is_local,
        "quantization": model_config.quantization,
        "cost_per_1k_tokens": model_config.cost_per_1k_tokens,
        "requires_auth": model_config.requires_auth
    }

@router.post("/{model_key}/chat")
async def chat_completion(
    model_key: str,
    request: ChatCompletionRequest
) -> Dict[str, Any]:
    """Generate chat completion from model"""
    try:
        # Validate model exists
        model_config = model_registry.get_model(model_key)
        if not model_config:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not found"
            )
        
        # Validate messages
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="Messages list cannot be empty"
            )
        
        # Convert Pydantic models to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Call inference client
        result = await inference_client.chat_completion(
            model_key=model_key,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )
        
        return {
            "model": model_key,
            "model_name": model_config.name,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result["content"]
                },
                "finish_reason": result["finish_reason"]
            }],
            "usage": result["usage"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat completion error for {model_key}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat completion failed: {str(e)}"
        )

@router.post("/{model_key}/completion")
async def text_completion(
    model_key: str,
    request: TextCompletionRequest
) -> Dict[str, Any]:
    """Generate text completion from model"""
    try:
        # Validate model exists
        model_config = model_registry.get_model(model_key)
        if not model_config:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not found"
            )
        
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        # Call inference client
        result = await inference_client.text_completion(
            model_key=model_key,
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )
        
        return {
            "model": model_key,
            "model_name": model_config.name,
            "choices": [{
                "text": result["content"],
                "finish_reason": result["finish_reason"]
            }],
            "usage": result["usage"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Text completion error for {model_key}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text completion failed: {str(e)}"
        )

@router.post("/{model_key}/embedding")
async def generate_embedding(
    model_key: str,
    request: EmbeddingRequest
) -> Dict[str, Any]:
    """Generate embeddings for text(s)"""
    try:
        # Validate model exists and is embedding type
        model_config = model_registry.get_model(model_key)
        if not model_config:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not found"
            )
        
        if model_config.model_type != ModelType.EMBEDDING:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_key}' is not an embedding model"
            )
        
        # Validate texts
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="Texts list cannot be empty"
            )
        
        # Call inference client
        result = await inference_client.embedding(
            model_key=model_key,
            texts=request.texts
        )
        
        return {
            "model": model_key,
            "model_name": model_config.name,
            "embeddings": result["embeddings"],
            "dimension": result["dimension"],
            "count": len(result["embeddings"])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Embedding error for {model_key}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )

@router.get("/{model_key}/health")
async def model_health(model_key: str) -> Dict[str, Any]:
    """Check if a model is healthy and accessible"""
    try:
        model_config = model_registry.get_model(model_key)
        if not model_config:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not found"
            )
        
        # Test with a simple prompt
        test_messages = [{"role": "user", "content": "Say 'OK' if you're working."}]
        
        import time
        start_time = time.time()
        
        result = await inference_client.chat_completion(
            model_key=model_key,
            messages=test_messages,
            max_tokens=10
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "model": model_key,
            "model_name": model_config.name,
            "response_time_ms": round(response_time_ms, 2),
            "accessible": True,
            "test_response": result["content"][:50]  # First 50 chars
        }
    
    except Exception as e:
        logger.error(f"❌ Health check failed for {model_key}: {str(e)}")
        return {
            "status": "unhealthy",
            "model": model_key,
            "accessible": False,
            "error": str(e)
        }

@router.get("/search/query")
async def search_models(q: str) -> Dict[str, Any]:
    """Search models by name, description, or tasks"""
    try:
        if not q or len(q.strip()) < 2:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 2 characters"
            )
        
        results = model_registry.search_models(q)
        
        # Convert to API format
        models = []
        for config in results:
            # Find the key for this config
            model_key = None
            for key, cfg in model_registry._registry.items():
                if cfg == config:
                    model_key = key
                    break
            
            if model_key:
                models.append({
                    "key": model_key,
                    "name": config.name,
                    "model_id": config.model_id,
                    "type": config.model_type.value,
                    "description": config.description,
                    "supported_tasks": config.supported_tasks
                })
        
        return {
            "query": q,
            "count": len(models),
            "results": models
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/local/loaded")
async def get_loaded_models() -> Dict[str, Any]:
    """Get list of currently loaded local models"""
    try:
        loaded = inference_client.get_loaded_models()
        
        return {
            "count": len(loaded),
            "models": loaded
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting loaded models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get loaded models: {str(e)}"
        )

@router.delete("/local/{model_key}")
async def unload_model(model_key: str) -> Dict[str, Any]:
    """Unload a local model from memory"""
    try:
        success = inference_client.unload_local_model(model_key)
        
        if success:
            return {
                "status": "success",
                "message": f"Model '{model_key}' unloaded",
                "model": model_key
            }
        else:
            return {
                "status": "not_loaded",
                "message": f"Model '{model_key}' was not loaded",
                "model": model_key
            }
    
    except Exception as e:
        logger.error(f"❌ Error unloading model {model_key}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )
