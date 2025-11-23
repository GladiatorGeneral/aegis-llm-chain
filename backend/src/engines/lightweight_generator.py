"""
Lightweight generator for testing without GPU requirements
Uses smaller models and mock responses for development
"""
import time
import random
import asyncio
import logging
from typing import Dict, Any, List

from engines.base import BaseGenerator, GenerationRequest, GenerationResponse, GenerationTask, ModelProvider
from core.security import security_layer

logger = logging.getLogger(__name__)

class LightweightGenerator(BaseGenerator):
    """Lightweight generator for development and testing"""
    
    def __init__(self):
        self.mock_responses = {
            GenerationTask.TEXT_COMPLETION: [
                "This is a generated completion based on your input.",
                "The model has processed your request and provided this response.",
                "Based on the context, here is the completed text."
            ],
            GenerationTask.CHAT: [
                "Hello! I'm your AI assistant. How can I help you today?",
                "That's an interesting question. Let me think about that...",
                "I understand what you're asking. Here's my response."
            ],
            GenerationTask.CODE_GENERATION: [
                "def hello_world():\n    print('Hello, World!')",
                "function calculateSum(a, b) {\n    return a + b;\n}",
                "public class Main {\n    public static void main(String[] args) {\n        System.out.println('Hello World');\n    }\n}"
            ],
            GenerationTask.TEXT_SUMMARIZATION: [
                "This text discusses important concepts that are summarized here.",
                "The main points of the input text have been condensed into this summary.",
                "Key takeaways from the provided content are presented in this summary."
            ],
            GenerationTask.TEXT_TRANSLATION: [
                "This is the translated version of your text.",
                "The translation has been completed successfully.",
                "Your input has been translated to the target language."
            ]
        }
        
        logger.info("LightweightGenerator initialized - using mock responses")
    
    def supports_task(self, task: GenerationTask) -> bool:
        return task in self.mock_responses
    
    def get_supported_models(self) -> List[str]:
        return ["lightweight-mock-model"]
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate mock responses for testing"""
        start_time = time.time()
        
        # Security validation
        is_valid, error_msg = await security_layer.validate_input(request.prompt)
        if not is_valid:
            return GenerationResponse(
                content="",
                model_used="security_layer",
                provider=request.provider.value,
                latency=0.0,
                safety_flagged=True,
                safety_reason=f"Security violation: {error_msg}"
            )
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Get mock response
        responses = self.mock_responses.get(request.task, ["Mock response"])
        content = random.choice(responses)
        
        # Add prompt context to response
        content = f"Prompt: '{request.prompt[:50]}...'\n\nResponse: {content}"
        
        # Safety check
        sanitized_content = await security_layer.sanitize_output(content)
        safety_flagged = sanitized_content != content
        
        latency = time.time() - start_time
        
        return GenerationResponse(
            content=sanitized_content,
            model_used="lightweight-mock-model",
            provider=request.provider.value if hasattr(request.provider, 'value') else str(request.provider),
            latency=latency,
            tokens_used=len(content.split()),
            metadata={
                "task": request.task.value if hasattr(request.task, 'value') else str(request.task),
                "mock": True,
                "parameters_used": request.parameters
            },
            safety_flagged=safety_flagged,
            safety_reason="PII detected and redacted" if safety_flagged else None
        )

# Global lightweight instance
lightweight_generator = LightweightGenerator()
