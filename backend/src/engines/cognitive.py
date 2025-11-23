"""
Unified Cognitive Engine - Orchestrates between Generation and Analysis
The brain of our AGI platform that intelligently routes between capabilities
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)

try:
    from .generator import universal_generator
    GENERATOR_AVAILABLE = universal_generator is not None
except (ImportError, Exception) as e:
    logger.warning(f"Could not import universal_generator: {e}")
    universal_generator = None
    GENERATOR_AVAILABLE = False

from .lightweight_generator import lightweight_generator
from .lightweight_analyzer import lightweight_analyzer
from .base import GenerationRequest, GenerationTask

try:
    from .analyzer import AnalysisRequest, AnalysisTask, universal_analyzer
    ANALYZER_AVAILABLE = universal_analyzer is not None
except (ImportError, Exception) as e:
    logger.warning(f"Could not import universal_analyzer: {e}")
    universal_analyzer = None
    ANALYZER_AVAILABLE = False
    from .analyzer import AnalysisRequest, AnalysisTask

from core.security import security_layer, SecurityViolation

class CognitiveObjective(str, Enum):
    GENERATE = "generate"
    ANALYZE = "analyze" 
    EVALUATE = "evaluate"
    SYNTHESIZE = "synthesize"
    REASON = "reason"

class CognitiveRequest(BaseModel):
    """Unified cognitive request"""
    input: Any
    objectives: List[CognitiveObjective]
    context: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    require_reasoning: bool = True
    use_lightweight: bool = False
    
    class Config:
        use_enum_values = True

class CognitiveResponse(BaseModel):
    """Unified cognitive response"""
    results: Dict[str, Any]
    reasoning_trace: List[Dict[str, Any]]
    final_output: Any
    processing_sequence: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UnifiedCognitiveEngine:
    """
    The core AGI brain - intelligently routes between generation and analysis
    to accomplish complex cognitive tasks
    """
    
    def __init__(self):
        # Use lightweight versions as fallback if full versions unavailable
        self.generator = universal_generator if GENERATOR_AVAILABLE else lightweight_generator
        self.analyzer = universal_analyzer if ANALYZER_AVAILABLE else lightweight_analyzer
        self.lightweight_generator = lightweight_generator
        self.lightweight_analyzer = lightweight_analyzer
        
        # Task routing intelligence
        self.analysis_keywords = ['analyze', 'classify', 'extract', 'evaluate', 'check', 'verify', 'detect']
        self.generation_keywords = ['generate', 'create', 'write', 'compose', 'build', 'make']
        
        logger.info(f"UnifiedCognitiveEngine initialized (Generator: {'Full' if GENERATOR_AVAILABLE else 'Lightweight'}, Analyzer: {'Full' if ANALYZER_AVAILABLE else 'Lightweight'})")
    
    async def process(self, request: CognitiveRequest) -> CognitiveResponse:
        """Process cognitive request with intelligent routing between engines"""
        reasoning_trace = []
        results = {}
        processing_sequence = []
        
        # Select engines based on lightweight flag
        generator = self.lightweight_generator if request.use_lightweight else self.generator
        analyzer = self.lightweight_analyzer if request.use_lightweight else self.analyzer
        
        try:
            # Security validation
            input_text = self._extract_input_text(request.input)
            is_valid, error_msg = await security_layer.validate_input(input_text)
            if not is_valid:
                raise SecurityViolation(f"Input validation failed: {error_msg}")
            
            # Process each objective in sequence
            for objective in request.objectives:
                obj_value = objective.value if hasattr(objective, 'value') else str(objective)
                processing_sequence.append(f"Processing objective: {obj_value}")
                
                if obj_value == "analyze":
                    result = await self._handle_analysis(request, analyzer)
                    results["analysis"] = result
                    reasoning_trace.append({
                        "step": "analysis",
                        "result": result.get("summary", "Analysis completed"),
                        "confidence": result.get("confidence", 0.0)
                    })
                
                elif obj_value == "generate":
                    result = await self._handle_generation(request, generator)
                    results["generation"] = result
                    reasoning_trace.append({
                        "step": "generation", 
                        "result": result.get("summary", "Generation completed"),
                        "confidence": result.get("confidence", 0.0)
                    })
                
                elif obj_value == "evaluate":
                    result = await self._handle_evaluation(request, analyzer, generator)
                    results["evaluation"] = result
                    reasoning_trace.append({
                        "step": "evaluation",
                        "result": result.get("assessment", "Evaluation completed"),
                        "confidence": result.get("confidence", 0.0)
                    })
                
                elif obj_value == "synthesize":
                    result = await self._handle_synthesis(request, analyzer, generator)
                    results["synthesis"] = result
                    reasoning_trace.append({
                        "step": "synthesis",
                        "result": result.get("synthesis", "Synthesis completed"),
                        "confidence": result.get("confidence", 0.0)
                    })
            
            # Generate final output
            final_output = await self._synthesize_final_output(results, request)
            
            return CognitiveResponse(
                results=results,
                reasoning_trace=reasoning_trace,
                final_output=final_output,
                processing_sequence=processing_sequence,
                metadata={
                    "objectives_processed": len(request.objectives),
                    "engines_used": "lightweight" if request.use_lightweight else "full",
                    "success": True
                }
            )
            
        except Exception as e:
            logger.error(f"Cognitive processing error: {str(e)}")
            return CognitiveResponse(
                results={},
                reasoning_trace=[{"step": "error", "result": str(e)}],
                final_output="Cognitive processing failed",
                processing_sequence=processing_sequence,
                metadata={"success": False, "error": str(e)}
            )
    
    async def _handle_analysis(self, request: CognitiveRequest, analyzer) -> Dict[str, Any]:
        """Handle analysis objectives"""
        # Determine analysis type from input
        input_text = self._extract_input_text(request.input)
        analysis_task = self._determine_analysis_task(input_text)
        
        analysis_request = AnalysisRequest(
            task=analysis_task,
            input_data=request.input,
            context=request.context,
            parameters=request.parameters.get("analysis", {}),
            require_reasoning_chain=request.require_reasoning
        )
        
        result = await analyzer.analyze(analysis_request)
        
        task_value = analysis_task.value if hasattr(analysis_task, 'value') else str(analysis_task)
        confidence_level_value = result.confidence_level.value if hasattr(result.confidence_level, 'value') else str(result.confidence_level)
        
        return {
            "task": task_value,
            "result": result.result,
            "confidence": result.confidence,
            "summary": f"Analysis completed with {confidence_level_value} confidence"
        }
    
    async def _handle_generation(self, request: CognitiveRequest, generator) -> Dict[str, Any]:
        """Handle generation objectives"""
        input_text = self._extract_input_text(request.input)
        generation_task = self._determine_generation_task(input_text)
        
        generation_request = GenerationRequest(
            task=generation_task,
            prompt=input_text,
            parameters=request.parameters.get("generation", {}),
            safety_checks=True
        )
        
        result = await generator.generate(generation_request)
        
        task_value = generation_task.value if hasattr(generation_task, 'value') else str(generation_task)
        
        return {
            "task": task_value,
            "result": result.content,
            "confidence": 0.8,  # Generation confidence
            "summary": f"Generated {task_value} with {len(result.content)} characters"
        }
    
    async def _handle_evaluation(self, request: CognitiveRequest, analyzer, generator) -> Dict[str, Any]:
        """Handle evaluation objectives (analysis + generation)"""
        # First analyze the input
        analysis_result = await self._handle_analysis(request, analyzer)
        
        # Then generate an evaluation based on analysis
        eval_prompt = f"Evaluate the following based on the analysis: {analysis_result['result']}"
        
        eval_request = GenerationRequest(
            task=GenerationTask.CHAT,
            prompt=eval_prompt,
            parameters=request.parameters.get("evaluation", {}),
            safety_checks=True
        )
        
        eval_result = await generator.generate(eval_request)
        
        return {
            "analysis_basis": analysis_result,
            "evaluation": eval_result.content,
            "confidence": min(analysis_result['confidence'], 0.8),
            "assessment": "Evaluation completed based on analysis"
        }
    
    async def _handle_synthesis(self, request: CognitiveRequest, analyzer, generator) -> Dict[str, Any]:
        """Handle synthesis objectives (multiple analyses + generation)"""
        # Perform multiple analyses
        analyses = {}
        for task in [AnalysisTask.SENTIMENT_ANALYSIS, AnalysisTask.TEXT_CLASSIFICATION]:
            try:
                analysis_request = AnalysisRequest(
                    task=task,
                    input_data=request.input,
                    context=request.context,
                    parameters=request.parameters.get("analysis", {})
                )
                result = await analyzer.analyze(analysis_request)
                task_value = task.value if hasattr(task, 'value') else str(task)
                analyses[task_value] = result.result
            except Exception as e:
                logger.warning(f"Analysis task {task} failed: {e}")
        
        # Synthesize results
        synthesis_prompt = f"Synthesize the following analyses into a coherent summary: {analyses}"
        
        synthesis_request = GenerationRequest(
            task=GenerationTask.TEXT_SUMMARIZATION,
            prompt=synthesis_prompt,
            parameters=request.parameters.get("synthesis", {}),
            safety_checks=True
        )
        
        synthesis_result = await generator.generate(synthesis_request)
        
        return {
            "analyses": analyses,
            "synthesis": synthesis_result.content,
            "confidence": 0.7,
            "summary": "Synthesis completed from multiple analyses"
        }
    
    def _determine_analysis_task(self, input_text: str) -> AnalysisTask:
        """Intelligently determine the appropriate analysis task"""
        input_lower = input_text.lower()
        
        if any(word in input_lower for word in ['sentiment', 'feeling', 'emotion']):
            return AnalysisTask.SENTIMENT_ANALYSIS
        elif any(word in input_lower for word in ['entity', 'name', 'person', 'organization']):
            return AnalysisTask.ENTITY_EXTRACTION
        elif any(word in input_lower for word in ['classify', 'category', 'type']):
            return AnalysisTask.TEXT_CLASSIFICATION
        elif any(word in input_lower for word in ['question', 'answer', 'what', 'how']):
            return AnalysisTask.QUESTION_ANSWERING
        elif any(word in input_lower for word in ['code', 'program', 'function']):
            return AnalysisTask.CODE_ANALYSIS
        else:
            return AnalysisTask.TEXT_CLASSIFICATION  # Default
    
    def _determine_generation_task(self, input_text: str) -> GenerationTask:
        """Intelligently determine the appropriate generation task"""
        input_lower = input_text.lower()
        
        if any(word in input_lower for word in ['code', 'program', 'function']):
            return GenerationTask.CODE_GENERATION
        elif any(word in input_lower for word in ['summarize', 'summary']):
            return GenerationTask.TEXT_SUMMARIZATION
        elif any(word in input_lower for word in ['translate']):
            return GenerationTask.TEXT_TRANSLATION
        else:
            return GenerationTask.CHAT  # Default
    
    def _extract_input_text(self, input_data: Any) -> str:
        """Extract text from various input types"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            for key in ['text', 'content', 'input', 'prompt']:
                if key in input_data and isinstance(input_data[key], str):
                    return input_data[key]
            return str(input_data)
        else:
            return str(input_data)
    
    async def _synthesize_final_output(self, results: Dict[str, Any], request: CognitiveRequest) -> Any:
        """Synthesize final output from all results"""
        if not results:
            return "No results generated"
        
        # Simple synthesis - in production, use more sophisticated methods
        if "generation" in results:
            return results["generation"]["result"]
        elif "synthesis" in results:
            return results["synthesis"]["synthesis"]
        elif "analysis" in results:
            return results["analysis"]["result"]
        else:
            return list(results.values())[0]["result"]

# Global cognitive engine instance
cognitive_engine = UnifiedCognitiveEngine()
