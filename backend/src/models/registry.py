"""Model registry for managing available models."""

from typing import Dict, List, Optional
from pydantic import BaseModel
from enum import Enum

class ModelType(str, Enum):
    """Types of models."""
    GENERATIVE = "generative"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"

class ModelCapability(str, Enum):
    """Model capabilities."""
    GENERATION = "generation"
    CHAT = "chat"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"

class ModelMetadata(BaseModel):
    """Model metadata."""
    model_id: str
    name: str
    type: ModelType
    size: str
    provider: str
    capabilities: List[ModelCapability]
    status: str = "available"
    deployment_url: Optional[str] = None

class ModelRegistry:
    """Central registry for all available models."""
    
    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default model configurations."""
        # Add default models
        self.register_model(ModelMetadata(
            model_id="llama-2-7b",
            name="Llama 2 7B",
            type=ModelType.GENERATIVE,
            size="7B",
            provider="meta",
            capabilities=[ModelCapability.GENERATION, ModelCapability.CHAT]
        ))
        
        self.register_model(ModelMetadata(
            model_id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            type=ModelType.GENERATIVE,
            size="175B",
            provider="openai",
            capabilities=[
                ModelCapability.GENERATION,
                ModelCapability.CHAT,
                ModelCapability.REASONING
            ]
        ))
    
    def register_model(self, metadata: ModelMetadata):
        """Register a new model."""
        self.models[metadata.model_id] = metadata
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    def list_models(
        self, 
        model_type: Optional[ModelType] = None,
        capability: Optional[ModelCapability] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.type == model_type]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        return models
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model."""
        if model_id in self.models:
            del self.models[model_id]
            return True
        return False

# Global model registry instance
model_registry = ModelRegistry()
