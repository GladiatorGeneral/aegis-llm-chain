"""
Lightweight analyzer for testing without heavy model dependencies
"""
import time
import random
import logging
from typing import Dict, Any, List, Optional
import asyncio

from .analyzer import BaseAnalyzer, AnalysisRequest, AnalysisResponse, AnalysisTask, AnalysisConfidence
from core.security import security_layer

logger = logging.getLogger(__name__)

class LightweightAnalyzer(BaseAnalyzer):
    """Lightweight analyzer for development and testing"""
    
    def __init__(self):
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
        
        logger.info("LightweightAnalyzer initialized - using mock analysis")
    
    def supports_task(self, task: AnalysisTask) -> bool:
        return task in [
            AnalysisTask.SENTIMENT_ANALYSIS,
            AnalysisTask.ENTITY_EXTRACTION, 
            AnalysisTask.TEXT_CLASSIFICATION,
            AnalysisTask.QUESTION_ANSWERING,
            AnalysisTask.SUMMARIZATION
        ]
    
    def get_supported_models(self) -> List[str]:
        return ["lightweight-mock-analyzer"]
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Generate mock analysis responses"""
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
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Get task value
        task_value = request.task.value if hasattr(request.task, 'value') else str(request.task)
        
        # Generate mock response based on task
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
        
        # Build reasoning chain if requested
        reasoning_chain = None
        if request.require_reasoning_chain:
            reasoning_chain = [
                f"Mock analysis for {task_value}",
                f"Input: {input_text[:50]}...",
                f"Generated simulated results",
                f"Confidence level: {result['confidence_level'].value if hasattr(result['confidence_level'], 'value') else str(result['confidence_level'])}"
            ]
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            result=result["result"],
            confidence=result["confidence"],
            confidence_level=result["confidence_level"],
            reasoning_chain=reasoning_chain,
            processing_time=processing_time,
            model_used="lightweight-mock-analyzer",
            metadata=result.get("metadata", {})
        )
    
    async def _mock_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Mock sentiment analysis"""
        sentiments = list(self.mock_responses[AnalysisTask.SENTIMENT_ANALYSIS].values())
        result = random.choice(sentiments)
        
        confidence, confidence_level = self._calculate_confidence(score=result["score"])
        
        return {
            "result": result,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {"method": "mock_sentiment_analysis"}
        }
    
    async def _mock_entity_extraction(self, text: str) -> Dict[str, Any]:
        """Mock entity extraction"""
        # Extract some mock entities based on text length
        entities = {}
        for entity_type, examples in self.mock_responses[AnalysisTask.ENTITY_EXTRACTION].items():
            # Select random subset of entities
            num_entities = min(len(examples), max(1, len(text) // 50))
            selected_entities = random.sample(examples, num_entities)
            entities[entity_type] = selected_entities
        
        confidence = 0.75 + (random.random() * 0.2)  # 0.75-0.95
        confidence, confidence_level = self._calculate_confidence(score=confidence)
        
        return {
            "result": entities,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {
                "method": "mock_entity_extraction",
                "entity_count": sum(len(ents) for ents in entities.values())
            }
        }
    
    async def _mock_text_classification(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock text classification"""
        candidate_labels = parameters.get("candidate_labels", ["technology", "science", "politics"])
        
        # Generate random scores that sum to 1
        scores = [random.random() for _ in candidate_labels]
        total = sum(scores)
        normalized_scores = [score/total for score in scores]
        
        result = {
            "sequence": text,
            "labels": candidate_labels,
            "scores": normalized_scores
        }
        
        max_confidence = max(normalized_scores)
        confidence, confidence_level = self._calculate_confidence(score=max_confidence)
        
        return {
            "result": result,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {
                "method": "mock_text_classification",
                "labels_used": candidate_labels
            }
        }
    
    async def _mock_question_answering(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Mock question answering"""
        context_text = context.get("context", "Mock context") if context else "General knowledge"
        
        answers = [
            "Based on the available information, this appears to be the case.",
            "The evidence suggests this is likely true.",
            "Further investigation would be needed for a definitive answer.",
            "This aligns with established knowledge in the field."
        ]
        
        result = {
            "answer": random.choice(answers),
            "score": 0.7 + (random.random() * 0.25),  # 0.7-0.95
            "start": 0,
            "end": len(question)
        }
        
        confidence, confidence_level = self._calculate_confidence(score=result["score"])
        
        return {
            "result": result,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {
                "method": "mock_question_answering",
                "context_used": bool(context)
            }
        }
    
    async def _mock_summarization(self, text: str) -> Dict[str, Any]:
        """Mock text summarization"""
        summary = f"This is a mock summary of the text which was {len(text)} characters long. The main points have been extracted and condensed."
        
        return {
            "result": {"summary": summary},
            "confidence": 0.8,
            "confidence_level": AnalysisConfidence.HIGH,
            "metadata": {
                "method": "mock_summarization",
                "original_length": len(text),
                "summary_length": len(summary)
            }
        }
    
    async def _mock_general_analysis(self, text: str, task: AnalysisTask) -> Dict[str, Any]:
        """Mock general analysis fallback"""
        task_value = task.value if hasattr(task, 'value') else str(task)
        analysis = f"Mock analysis for {task_value}: The input text has been processed and analyzed using simulated methods."
        
        return {
            "result": {"analysis": analysis},
            "confidence": 0.6,
            "confidence_level": AnalysisConfidence.MEDIUM,
            "metadata": {
                "method": "mock_general_analysis",
                "task": task_value
            }
        }
    
    def _extract_text_from_input(self, input_data: Any) -> str:
        """Extract text for security validation"""
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
        """Calculate confidence score and level"""
        if score >= 0.9:
            level = AnalysisConfidence.HIGH
        elif score >= 0.7:
            level = AnalysisConfidence.MEDIUM
        else:
            level = AnalysisConfidence.LOW
        return score, level

# Global lightweight analyzer instance
lightweight_analyzer = LightweightAnalyzer()
