"""Base abstract classes for universal engines."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class EngineConfig(BaseModel):
    """Base engine configuration."""
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
