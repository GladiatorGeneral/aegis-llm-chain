"""Universal Analysis Engine with multi-task capabilities."""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.security import security_layer

logger = logging.getLogger(__name__)

class AnalysisTask(str, Enum):
    """Supported analysis tasks"""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_EXTRACTION = "entity_extraction"
    TEXT_CLASSIFICATION = "text_classification"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CODE_ANALYSIS = "code_analysis"
    LOGICAL_REASONING = "logical_reasoning"
    QUALITY_EVALUATION = "quality_evaluation"
    FACT_VERIFICATION = "fact_verification"
    INTENT_DETECTION = "intent_detection"

class AnalysisConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class AnalysisRequest(BaseModel):
    """Universal analysis request"""
    model_config = ConfigDict(use_enum_values=True)
    task: AnalysisTask
    input_data: Any
    context: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    require_reasoning_chain: bool = False
    safety_checks: bool = True

class AnalysisResponse(BaseModel):
    """Universal analysis response"""
    result: Any
    confidence: float
    confidence_level: AnalysisConfidence
    reasoning_chain: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float
    model_used: str

class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers"""
    
    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        pass
    
    @abstractmethod
    def supports_task(self, task: AnalysisTask) -> bool:
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        pass
    
    def _calculate_confidence(self, logits: Any = None, score: float = None) -> tuple:
        """Calculate confidence score and level"""
        if score is not None:
            confidence = score
        elif logits is not None:
            # Convert logits to confidence (simplified)
            try:
                import torch
                if hasattr(logits, 'max'):
                    confidence = float(torch.softmax(logits, dim=-1).max())
                else:
                    confidence = 0.8  # Default for non-logit outputs
            except:
                confidence = 0.8
        else:
            confidence = 0.7  # Default confidence
        
        # Determine confidence level
        if confidence >= 0.9:
            level = AnalysisConfidence.HIGH
        elif confidence >= 0.7:
            level = AnalysisConfidence.MEDIUM
        else:
            level = AnalysisConfidence.LOW
            
        return confidence, level


try:
    import torch
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForCausalLM
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - HuggingFaceAnalyzer will not work")

class HuggingFaceAnalyzer(BaseAnalyzer):
    """Universal analyzer using Hugging Face models for various analysis tasks"""
    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required for HuggingFaceAnalyzer")
            
        self.hf_token = hf_token
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Model registry for analysis tasks
        self.model_registry = {
            AnalysisTask.SENTIMENT_ANALYSIS: {
                "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "pipeline": "text-classification",
                "description": "Sentiment analysis for text"
            },
            AnalysisTask.ENTITY_EXTRACTION: {
                "model_id": "dslim/bert-base-NER",
                "pipeline": "token-classification", 
                "description": "Named entity recognition"
            },
            AnalysisTask.TEXT_CLASSIFICATION: {
                "model_id": "facebook/bart-large-mnli",
                "pipeline": "zero-shot-classification",
                "description": "Zero-shot text classification"
            },
            AnalysisTask.QUESTION_ANSWERING: {
                "model_id": "deepset/roberta-base-squad2",
                "pipeline": "question-answering",
                "description": "Extractive question answering"
            },
            AnalysisTask.CODE_ANALYSIS: {
                "model_id": "microsoft/codebert-base",
                "pipeline": "text-classification",
                "description": "Code analysis and understanding"
            },
            AnalysisTask.SUMMARIZATION: {
                "model_id": "facebook/bart-large-cnn",
                "pipeline": "summarization",
                "description": "Text summarization"
            }
        }
        
        # Loaded models cache
        self.loaded_pipelines: Dict[str, Any] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        
        logger.info(f"HuggingFaceAnalyzer initialized with device: {device}")
    
    def supports_task(self, task: AnalysisTask) -> bool:
        return task in self.model_registry
    
    def get_supported_models(self) -> List[str]:
        return list(set([config["model_id"] for config in self.model_registry.values()]))
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute analysis with security and reasoning chain support"""
        start_time = time.time()
        
        # Security validation
        if request.safety_checks:
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
        
        try:
            # Get model configuration
            if not self.supports_task(request.task):
                raise ValueError(f"Task {request.task} not supported")
            
            model_config = self.model_registry[request.task]
            model_id = request.parameters.get("model_id", model_config["model_id"])
            
            # Execute analysis based on task
            task_value = request.task.value if hasattr(request.task, 'value') else str(request.task)
            
            if task_value == "sentiment_analysis":
                result = await self._analyze_sentiment(request.input_data, model_id, request.parameters)
            elif task_value == "entity_extraction":
                result = await self._extract_entities(request.input_data, model_id, request.parameters)
            elif task_value == "text_classification":
                result = await self._classify_text(request.input_data, model_id, request.parameters)
            elif task_value == "question_answering":
                result = await self._answer_question(request.input_data, model_id, request.parameters, request.context)
            elif task_value == "code_analysis":
                result = await self._analyze_code(request.input_data, model_id, request.parameters)
            elif task_value == "summarization":
                result = await self._summarize_text(request.input_data, model_id, request.parameters)
            else:
                # Fallback to general analysis
                result = await self._general_analysis(request.input_data, model_id, request.parameters, request.task)
            
            # Build reasoning chain if requested
            reasoning_chain = None
            if request.require_reasoning_chain:
                reasoning_chain = await self._build_reasoning_chain(request, result)
            
            processing_time = time.time() - start_time
            
            return AnalysisResponse(
                result=result["result"],
                confidence=result["confidence"],
                confidence_level=result["confidence_level"],
                reasoning_chain=reasoning_chain,
                processing_time=processing_time,
                model_used=model_id,
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            processing_time = time.time() - start_time
            return AnalysisResponse(
                result={"error": str(e)},
                confidence=0.0,
                confidence_level=AnalysisConfidence.LOW,
                processing_time=processing_time,
                model_used="error",
                metadata={"error_type": type(e).__name__}
            )
    
    async def _analyze_sentiment(self, text: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        pipeline_obj = await self._get_pipeline("text-classification", model_id)
        
        def _analyze():
            return pipeline_obj(text, **parameters)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _analyze)
        
        # Handle both single and batch results
        if isinstance(result, list) and len(result) > 0:
            best_result = result[0] if isinstance(result[0], dict) else result
            confidence = best_result.get('score', 0.8)
        else:
            best_result = result[0] if isinstance(result, list) else result
            confidence = best_result.get('score', 0.8)
        
        confidence, confidence_level = self._calculate_confidence(score=confidence)
        
        return {
            "result": best_result,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {
                "task": "sentiment_analysis",
                "model_id": model_id
            }
        }
    
    async def _extract_entities(self, text: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract named entities from text"""
        pipeline_obj = await self._get_pipeline("token-classification", model_id)
        
        def _extract():
            return pipeline_obj(text, **parameters)
        
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(self.executor, _extract)
        
        # Group entities by type
        grouped_entities = {}
        for entity in entities:
            entity_type = entity['entity_group']
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            grouped_entities[entity_type].append({
                'word': entity['word'],
                'score': entity['score'],
                'start': entity.get('start'),
                'end': entity.get('end')
            })
        
        # Calculate average confidence
        scores = [entity['score'] for entity in entities]
        avg_confidence = sum(scores) / len(scores) if scores else 0.8
        confidence, confidence_level = self._calculate_confidence(score=avg_confidence)
        
        return {
            "result": grouped_entities,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {
                "task": "entity_extraction",
                "model_id": model_id,
                "entity_count": len(entities)
            }
        }
    
    async def _classify_text(self, text: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Classify text using zero-shot classification"""
        pipeline_obj = await self._get_pipeline("zero-shot-classification", model_id)
        
        # Get candidate labels from parameters or use defaults
        candidate_labels = parameters.get(
            "candidate_labels", 
            ["politics", "technology", "science", "sports", "entertainment"]
        )
        
        def _classify():
            return pipeline_obj(text, candidate_labels, **parameters)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _classify)
        
        # Get highest confidence score
        max_confidence = max(result['scores']) if result['scores'] else 0.5
        confidence, confidence_level = self._calculate_confidence(score=max_confidence)
        
        return {
            "result": result,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {
                "task": "text_classification",
                "model_id": model_id,
                "labels_used": candidate_labels
            }
        }
    
    async def _answer_question(self, question: str, model_id: str, parameters: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Answer questions based on context"""
        pipeline_obj = await self._get_pipeline("question-answering", model_id)
        
        # Extract context from request or use default
        context_text = context.get("context", "") if context else ""
        if not context_text:
            context_text = "This is a general knowledge question without specific context."
        
        def _answer():
            return pipeline_obj(question=question, context=context_text, **parameters)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _answer)
        
        confidence, confidence_level = self._calculate_confidence(score=result.get('score', 0.5))
        
        return {
            "result": result,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "metadata": {
                "task": "question_answering",
                "model_id": model_id,
                "context_length": len(context_text)
            }
        }
    
    async def _analyze_code(self, code: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code for quality and issues"""
        # Simple code analysis - in production, use specialized code analysis models
        analysis_categories = ["quality", "complexity", "security", "readability"]
        
        def _analyze_code():
            # This is a simplified analysis - real implementation would use code-specific models
            results = {}
            for category in analysis_categories:
                # Simulate analysis results
                results[category] = {
                    "score": 0.7,  # Placeholder
                    "issues": []   # Placeholder
                }
            return results
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _analyze_code)
        
        return {
            "result": result,
            "confidence": 0.6,  # Lower confidence for simulated analysis
            "confidence_level": AnalysisConfidence.MEDIUM,
            "metadata": {
                "task": "code_analysis",
                "model_id": model_id,
                "code_language": "python"  # Would detect automatically in production
            }
        }
    
    async def _summarize_text(self, text: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize text"""
        pipeline_obj = await self._get_pipeline("summarization", model_id)
        
        def _summarize():
            return pipeline_obj(text, **parameters)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _summarize)
        
        summary_text = result[0]['summary_text'] if result else "Summary not available."
        
        return {
            "result": {"summary": summary_text},
            "confidence": 0.8,
            "confidence_level": AnalysisConfidence.HIGH,
            "metadata": {
                "task": "summarization",
                "model_id": model_id,
                "original_length": len(text),
                "summary_length": len(summary_text)
            }
        }
    
    async def _general_analysis(self, input_data: Any, model_id: str, parameters: Dict[str, Any], task: AnalysisTask) -> Dict[str, Any]:
        """General analysis fallback"""
        task_value = task.value if hasattr(task, 'value') else str(task)
        analysis = f"Mock analysis for {task_value}: The input has been processed."
        
        return {
            "result": {"analysis": analysis},
            "confidence": 0.7,
            "confidence_level": AnalysisConfidence.MEDIUM,
            "metadata": {
                "task": task_value,
                "model_id": model_id,
                "method": "general_analysis"
            }
        }
    
    async def _get_pipeline(self, task: str, model_id: str) -> Any:
        """Get or create a pipeline for the given task and model"""
        pipeline_key = f"{task}_{model_id}"
        
        if pipeline_key not in self.loaded_pipelines:
            logger.info(f"Loading pipeline for {task} with model {model_id}")
            try:
                pipeline_obj = pipeline(
                    task,
                    model=model_id,
                    tokenizer=model_id,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.loaded_pipelines[pipeline_key] = pipeline_obj
                logger.info(f"Successfully loaded pipeline: {pipeline_key}")
            except Exception as e:
                logger.error(f"Failed to load pipeline {pipeline_key}: {str(e)}")
                raise
        
        return self.loaded_pipelines[pipeline_key]
    
    async def _build_reasoning_chain(self, request: AnalysisRequest, result: Dict[str, Any]) -> List[str]:
        """Build a reasoning chain explaining the analysis process"""
        task_value = request.task.value if hasattr(request.task, 'value') else str(request.task)
        confidence_level_value = result['confidence_level'].value if hasattr(result['confidence_level'], 'value') else str(result['confidence_level'])
        
        reasoning_chain = [
            f"Received {task_value} request",
            f"Selected model: {result['metadata'].get('model_id', 'unknown')}",
            f"Processed input data of type: {type(request.input_data).__name__}",
            f"Applied {task_value} analysis",
            f"Generated result with confidence: {result['confidence']:.2f}",
            f"Confidence level: {confidence_level_value}"
        ]
        
        # Add task-specific reasoning steps
        if task_value == "entity_extraction":
            entity_count = result['metadata'].get('entity_count', 0)
            reasoning_chain.append(f"Extracted {entity_count} entities across different categories")
        
        elif task_value == "sentiment_analysis":
            sentiment_result = result['result']
            if isinstance(sentiment_result, dict):
                label = sentiment_result.get('label', 'unknown')
                reasoning_chain.append(f"Determined sentiment: {label}")
        
        return reasoning_chain
    
    def _extract_text_from_input(self, input_data: Any) -> str:
        """Extract text from various input types for security validation"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            # Try to find text in common keys
            for key in ['text', 'content', 'input', 'prompt', 'question']:
                if key in input_data and isinstance(input_data[key], str):
                    return input_data[key]
            return str(input_data)
        else:
            return str(input_data)

# Global analyzer instance - create only if transformers available
if TRANSFORMERS_AVAILABLE:
    try:
        universal_analyzer = HuggingFaceAnalyzer()
    except Exception as e:
        logger.warning(f"Could not initialize HuggingFaceAnalyzer: {e}")
        universal_analyzer = None
else:
    universal_analyzer = None
