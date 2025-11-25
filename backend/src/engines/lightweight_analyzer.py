"""
Lightweight analyzer for testing without heavy model dependencies.
Modified to bridge functionality using external APIs (DeepSeek/OpenAI) when available.
"""
import time
import random
import logging
from typing import Dict, Any, List, Optional
import asyncio
import os
import aiohttp
import json

from .analyzer import BaseAnalyzer, AnalysisRequest, AnalysisResponse, AnalysisTask, AnalysisConfidence
from core.security import security_layer

logger = logging.getLogger(__name__)

class LightweightAnalyzer(BaseAnalyzer):
    """Lightweight analyzer that proxies to external APIs or falls back to mocks"""
    
    def __init__(self):
        # Try to get API configuration
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") if os.getenv("DEEPSEEK_API_KEY") else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        self.use_api = bool(self.api_key)
        
        if self.use_api:
            logger.info(f"LightweightAnalyzer initialized in BRIDGE mode (using API at {self.base_url})")
        else:
            logger.warning("LightweightAnalyzer initialized in MOCK mode (no API key found)")

        self.mock_responses = {
            AnalysisTask.SENTIMENT_ANALYSIS: {
                "positive": {"label": "POSITIVE", "score": 0.95},
                "negative": {"label": "NEGATIVE", "score": 0.87},
                "neutral": {"label": "NEUTRAL", "score": 0.76}
            },
            AnalysisTask.ENTITY_EXTRACTION: {
                "PERSON": ["John Doe", "Jane Smith"],
                "ORG": ["OpenAI", "Google"],
                "LOC": ["New York", "London"]
            },
            AnalysisTask.TEXT_CLASSIFICATION: {
                "labels": ["technology", "science", "politics"],
                "scores": [0.8, 0.15, 0.05]
            },
            AnalysisTask.QUESTION_ANSWERING: {
                "answer": "This is a simulated answer based on the context.",
                "score": 0.82
            }
        }

    def supports_task(self, task: AnalysisTask) -> bool:
        return True

    def get_supported_models(self) -> List[str]:
        if self.use_api:
            return ["deepseek-chat", "gpt-3.5-turbo", "lightweight-analyzer-bridge"]
        return ["lightweight-mock-analyzer"]

    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Generate analysis using API or mock fallback"""
        start_time = time.time()

        # Security validation
        input_text = self._extract_text_from_input(request.input_data)
        is_valid, error_msg = await security_layer.validate_input(input_text)
        if not is_valid:
            return AnalysisResponse(
                result={},
                confidence=0.0,
                confidence_level=AnalysisConfidence.LOW,
                processing_time=0.0,
                model_used="security_layer",
                metadata={"error": f"Security violation: {error_msg}"}
            )

        # Try API if available
        if self.use_api:
            try:
                return await self._analyze_via_api(request, input_text, start_time)
            except Exception as e:
                logger.error(f"API analysis failed, falling back to mock: {str(e)}")
                # Fallthrough to mock

        # Mock Fallback
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Get task value
        task_value = request.task.value if hasattr(request.task, 'value') else str(request.task)
        
        if task_value == "sentiment_analysis":
            result = await self._mock_sentiment_analysis(input_text)
        elif task_value == "entity_extraction":
            result = await self._mock_entity_extraction(input_text)
        elif task_value == "text_classification":
            result = await self._mock_text_classification(input_text, request.parameters)
        elif task_value == "question_answering":
            result = await self._mock_question_answering(input_text, request.context)
        elif task_value == "summarization":
            result = await self._mock_summarization(input_text)
        else:
            result = await self._mock_general_analysis(input_text, request.task)

        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            result=result["result"],
            confidence=result["confidence"],
            confidence_level=result["confidence_level"],
            processing_time=processing_time,
            model_used="lightweight-mock-analyzer",
            metadata=result.get("metadata", {})
        )

    async def _analyze_via_api(self, request: AnalysisRequest, input_text: str, start_time: float) -> AnalysisResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        prompt = self._construct_analysis_prompt(request.task, input_text, request.parameters)
        
        # Determine model name
        model_name = "deepseek-chat" if "deepseek" in self.base_url else "gpt-3.5-turbo"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert text analysis engine. Return the result in valid JSON format only. Do not include markdown formatting like ```json."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        # Add response_format if supported (OpenAI specific, but harmless to try or omit if DeepSeek ignores)
        # payload["response_format"] = {"type": "json_object"}

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API Error {response.status}: {error_text}")
                
                data = await response.json()
                content = data['choices'][0]['message']['content']
                
                # Clean content if it has markdown code blocks
                content = content.replace("```json", "").replace("```", "").strip()

                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if not valid JSON, wrap raw content
                    result = {"raw_analysis": content}

                processing_time = time.time() - start_time
                
                return AnalysisResponse(
                    result=result,
                    confidence=0.9,
                    confidence_level=AnalysisConfidence.HIGH,
                    processing_time=processing_time,
                    model_used=model_name,
                    metadata={"api_bridge": True}
                )

    def _construct_analysis_prompt(self, task: AnalysisTask, text: str, params: Dict) -> str:
        task_str = str(task)
        if "sentiment" in task_str:
            return f"Analyze the sentiment of the following text. Return JSON with keys 'label' (POSITIVE, NEGATIVE, NEUTRAL) and 'score' (0.0-1.0).\n\nText: {text}"
        elif "entity" in task_str:
            return f"Extract named entities from the text. Return JSON where keys are entity types (PERSON, ORG, LOC) and values are lists of entities.\n\nText: {text}"
        elif "classification" in task_str:
            labels = params.get("candidate_labels", ["technology", "science", "politics"])
            return f"Classify the text into one of these labels: {labels}. Return JSON with keys 'label' and 'score'.\n\nText: {text}"
        elif "summarization" in task_str:
             return f"Summarize the text. Return JSON with key 'summary'.\n\nText: {text}"
        elif "question" in task_str:
             return f"Answer the question based on the text. Return JSON with key 'answer'.\n\nText: {text}"
        else:
            return f"Analyze the following text for {task_str}. Return the result as JSON.\n\nText: {text}"

    # ... Mock methods ...
    async def _mock_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        sentiments = list(self.mock_responses[AnalysisTask.SENTIMENT_ANALYSIS].values())
        result = random.choice(sentiments)
        confidence, confidence_level = self._calculate_confidence(score=result["score"])
        return {"result": result, "confidence": confidence, "confidence_level": confidence_level, "metadata": {"method": "mock"}}

    async def _mock_entity_extraction(self, text: str) -> Dict[str, Any]:
        entities = {}
        for entity_type, examples in self.mock_responses[AnalysisTask.ENTITY_EXTRACTION].items():
            num_entities = min(len(examples), max(1, len(text) // 50))
            selected_entities = random.sample(examples, num_entities)
            entities[entity_type] = selected_entities
        confidence, confidence_level = self._calculate_confidence(score=0.8)
        return {"result": entities, "confidence": confidence, "confidence_level": confidence_level, "metadata": {"method": "mock"}}

    async def _mock_text_classification(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        candidate_labels = parameters.get("candidate_labels", ["technology", "science", "politics"])
        scores = [random.random() for _ in candidate_labels]
        total = sum(scores)
        normalized_scores = [score/total for score in scores]
        result = {"sequence": text, "labels": candidate_labels, "scores": normalized_scores}
        confidence, confidence_level = self._calculate_confidence(score=max(normalized_scores))
        return {"result": result, "confidence": confidence, "confidence_level": confidence_level, "metadata": {"method": "mock"}}

    async def _mock_question_answering(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        result = {"answer": "Mock answer", "score": 0.8}
        confidence, confidence_level = self._calculate_confidence(score=0.8)
        return {"result": result, "confidence": confidence, "confidence_level": confidence_level, "metadata": {"method": "mock"}}

    async def _mock_summarization(self, text: str) -> Dict[str, Any]:
        summary = f"Mock summary of {len(text)} chars."
        return {"result": {"summary": summary}, "confidence": 0.8, "confidence_level": AnalysisConfidence.HIGH, "metadata": {"method": "mock"}}

    async def _mock_general_analysis(self, text: str, task: AnalysisTask) -> Dict[str, Any]:
        return {"result": {"analysis": "Mock analysis"}, "confidence": 0.6, "confidence_level": AnalysisConfidence.MEDIUM, "metadata": {"method": "mock"}}

    def _extract_text_from_input(self, input_data: Any) -> str:
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            for key in ['text', 'content', 'input', 'question']:
                if key in input_data and isinstance(input_data[key], str):
                    return input_data[key]
            return str(input_data)
        else:
            return str(input_data)

    def _calculate_confidence(self, score: float) -> tuple:
        if score >= 0.9:
            level = AnalysisConfidence.HIGH
        elif score >= 0.7:
            level = AnalysisConfidence.MEDIUM
        else:
            level = AnalysisConfidence.LOW
        return score, level

# Global lightweight analyzer instance
lightweight_analyzer = LightweightAnalyzer()
