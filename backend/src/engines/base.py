"""Base abstract classes for universal engines."""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GenerationTask(str, Enum):
    """Supported generation tasks"""
    TEXT_COMPLETION = "text_completion"
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_TRANSLATION = "text_translation"
    DATA_GENERATION = "data_generation"

class ModelProvider(str, Enum):
    """Supported model providers"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"  # For fallback
    CUSTOM = "custom"

class GenerationRequest(BaseModel):
    """Universal generation request"""
    model_config = ConfigDict(protected_namespaces=(), use_enum_values=True)
    task: GenerationTask
    prompt: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    output_format: str = "text"
    provider: ModelProvider = ModelProvider.HUGGINGFACE
    model_id: Optional[str] = None  # Specific model to use
    safety_checks: bool = True

class GenerationResponse(BaseModel):
    """Universal generation response"""
    model_config = ConfigDict(protected_namespaces=())
    content: Any
    model_used: str
    provider: str
    latency: float
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    safety_flagged: bool = False
    safety_reason: Optional[str] = None

class BaseGenerator(ABC):
    """Abstract base class for all generators"""
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        pass
    
    @abstractmethod
    def supports_task(self, task: GenerationTask) -> bool:
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        pass
    
    def validate_request(self, request: GenerationRequest) -> bool:
        """Basic request validation"""
        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return True

# Legacy compatibility classes
class EngineConfig(BaseModel):
    """Base engine configuration."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

class EngineRequest(BaseModel):
    """Base engine request."""
    input_text: str
    config: EngineConfig
    metadata: Optional[Dict[str, Any]] = {}

class EngineResponse(BaseModel):
    """Base engine response."""
    model_config = ConfigDict(protected_namespaces=())
    output: str
    model_used: str
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any] = {}

class BaseEngine(ABC):
    """Abstract base class for all engines."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
    
    @abstractmethod
    async def process(self, request: EngineRequest) -> EngineResponse:
        """Process a request through the engine."""
        pass
    
    @abstractmethod
    async def validate_input(self, text: str) -> bool:
        """Validate input before processing."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the engine is healthy and ready."""
        pass
