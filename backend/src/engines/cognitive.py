"""Unified cognitive engine combining multiple capabilities."""

from typing import Dict, Any, Optional
from enum import Enum

from engines.base import BaseEngine, EngineRequest, EngineResponse, EngineConfig
from engines.generator import GenerativeEngine
from engines.analyzer import AnalysisEngine

class CognitiveTaskType(str, Enum):
    """Types of cognitive tasks."""
    GENERATION = "generation"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"

class CognitiveEngine:
    """Unified cognitive engine orchestrating multiple specialized engines."""
    
    def __init__(self):
        self.engines: Dict[str, BaseEngine] = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all specialized engines."""
        # TODO: Load from configuration
        default_config = EngineConfig(model_id="default", max_tokens=2048)
        self.engines["generation"] = GenerativeEngine(default_config)
        self.engines["analysis"] = AnalysisEngine(default_config)
    
    async def process(
        self, 
        task_type: CognitiveTaskType, 
        request: EngineRequest
    ) -> EngineResponse:
        """Process a cognitive task."""
        # Route to appropriate engine
        if task_type == CognitiveTaskType.GENERATION:
            return await self.engines["generation"].process(request)
        elif task_type == CognitiveTaskType.ANALYSIS:
            return await self.engines["analysis"].process(request)
        else:
            # TODO: Implement other task types
            raise NotImplementedError(f"Task type {task_type} not yet implemented")
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities of the cognitive engine."""
        return {
            "task_types": [task.value for task in CognitiveTaskType],
            "engines": list(self.engines.keys()),
            "models": ["default"]  # TODO: Get from model registry
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all engines."""
        health_status = {}
        for engine_name, engine in self.engines.items():
            health_status[engine_name] = await engine.health_check()
        return health_status

# Global cognitive engine instance
cognitive_engine = CognitiveEngine()
