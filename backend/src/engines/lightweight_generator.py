"""
Lightweight generator for testing without GPU requirements.
Modified to bridge functionality using external APIs (DeepSeek/OpenAI) when available.
"""
import time
import random
import asyncio
import logging
import os
import aiohttp
import json
from typing import Dict, Any, List

from engines.base import BaseGenerator, GenerationRequest, GenerationResponse, GenerationTask, ModelProvider
from core.security import security_layer

logger = logging.getLogger(__name__)

class LightweightGenerator(BaseGenerator):
    """Lightweight generator that proxies to external APIs or falls back to mocks"""
    
    def __init__(self):
        # Try to get API configuration
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") if os.getenv("DEEPSEEK_API_KEY") else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        self.use_api = bool(self.api_key)
        
        if self.use_api:
            logger.info(f"LightweightGenerator initialized in BRIDGE mode (using API at {self.base_url})")
        else:
            logger.warning("LightweightGenerator initialized in MOCK mode (no API key found)")

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

    def supports_task(self, task: GenerationTask) -> bool:
        return True

    def get_supported_models(self) -> List[str]:
        if self.use_api:
            return ["deepseek-chat", "gpt-3.5-turbo", "lightweight-bridge"]
        return ["lightweight-mock-model"]

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response using API or mock fallback"""
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

        # Try API if available
        if self.use_api:
            try:
                return await self._generate_via_api(request, start_time)
            except Exception as e:
                logger.error(f"API generation failed, falling back to mock: {str(e)}")
                # Fallthrough to mock

        # Mock Fallback
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        responses = self.mock_responses.get(request.task, ["Mock response"])
        content = random.choice(responses)
        
        # Add prompt context to response for clarity it's a mock
        content = f"[MOCK MODE] Prompt: '{request.prompt[:50]}...'\n\nResponse: {content}"
        
        sanitized_content = await security_layer.sanitize_output(content)
        safety_flagged = sanitized_content != content
        latency = time.time() - start_time

        return GenerationResponse(
            content=sanitized_content,
            model_used="lightweight-mock-model",
            provider=str(request.provider),
            latency=latency,
            tokens_used=len(content.split()),
            metadata={
                "task": str(request.task),
                "mock": True,
                "parameters_used": request.parameters
            },
            safety_flagged=safety_flagged,
            safety_reason="PII detected and redacted" if safety_flagged else None
        )

    async def _generate_via_api(self, request: GenerationRequest, start_time: float) -> GenerationResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Construct messages based on task
        messages = []
        system_prompt = self._get_system_prompt(request.task)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": request.prompt})

        # Determine model name
        model_name = "deepseek-chat" if "deepseek" in self.base_url else "gpt-3.5-turbo"
        if request.parameters and "model" in request.parameters:
            model_name = request.parameters["model"]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": request.parameters.get("temperature", 0.7),
            "max_tokens": request.parameters.get("max_tokens", 2000),
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API Error {response.status}: {error_text}")
                
                data = await response.json()
                content = data['choices'][0]['message']['content']
                
                # Calculate usage
                usage = data.get('usage', {})
                tokens_used = usage.get('total_tokens', len(content.split()))

                sanitized_content = await security_layer.sanitize_output(content)
                safety_flagged = sanitized_content != content
                latency = time.time() - start_time

                return GenerationResponse(
                    content=sanitized_content,
                    model_used=model_name,
                    provider="external-api",
                    latency=latency,
                    tokens_used=tokens_used,
                    metadata={
                        "task": str(request.task),
                        "api_bridge": True,
                        "finish_reason": data['choices'][0].get('finish_reason')
                    },
                    safety_flagged=safety_flagged,
                    safety_reason="PII detected" if safety_flagged else None
                )

    def _get_system_prompt(self, task: GenerationTask) -> str:
        if task == GenerationTask.CODE_GENERATION:
            return "You are an expert coding assistant. Provide clean, efficient, and well-commented code."
        elif task == GenerationTask.TEXT_SUMMARIZATION:
            return "You are a summarization assistant. Provide a concise summary of the following text."
        elif task == GenerationTask.TEXT_TRANSLATION:
            return "You are a translation assistant. Translate the text accurately."
        elif task == GenerationTask.CHAT:
            return "You are a helpful AI assistant named Aegis."
        return "You are a helpful AI assistant."

# Global lightweight instance
lightweight_generator = LightweightGenerator()
