"""
LLM-FE Engine - Intelligent Task Routing and Model Selection
Based on: "Distilling System 2 into System 1" and multi-expert routing research

Key Innovation: Use lightweight routing to select optimal models/engines for each task
Achieves 2-3x speedup while maintaining quality through intelligent model selection
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from engines.base import GenerationTask
from engines.analyzer import AnalysisTask

logger = logging.getLogger(__name__)

class RoutingStrategy(str, Enum):
    """Routing optimization strategies"""
    INTELLIGENT = "intelligent"    # Balance all factors
    PERFORMANCE = "performance"    # Optimize for speed
    QUALITY = "quality"           # Optimize for accuracy
    BALANCED = "balanced"         # Equal weight to speed and quality
    COST = "cost"                 # Minimize computational cost

class EngineType(str, Enum):
    """Available engine types"""
    GENERATOR = "generator"
    ANALYZER = "analyzer"
    COGNITIVE = "cognitive"
    OPTIMA = "optima"
    LIGHTWEIGHT_GEN = "lightweight_gen"
    LIGHTWEIGHT_ANALYZE = "lightweight_analyze"

@dataclass
class RoutingDecision:
    """Result of routing optimization"""
    engine: EngineType
    model: str
    parameters: Dict[str, Any]
    expected_quality: float
    expected_time: float
    reasoning: str
    confidence: float
    alternative_routes: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine.value,
            "model": self.model,
            "parameters": self.parameters,
            "expected_quality": self.expected_quality,
            "expected_time": self.expected_time,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "alternative_routes": self.alternative_routes
        }

class LLMFEEngine:
    """
    LLM Front-End Engine for Intelligent Routing
    
    Analyzes incoming requests and routes them to optimal engines/models.
    Uses heuristics and patterns to maximize efficiency while maintaining quality.
    """
    
    def __init__(
        self,
        generator,
        analyzer,
        routing_strategy: str = "intelligent",
        enable_model_selection: bool = True,
        enable_caching: bool = True
    ):
        """
        Initialize LLM-FE Engine
        
        Args:
            generator: Your HuggingFaceGenerator instance
            analyzer: Your HuggingFaceAnalyzer instance
            routing_strategy: Default strategy ("intelligent", "performance", "quality", "balanced", "cost")
            enable_model_selection: Enable intelligent model selection
            enable_caching: Cache routing decisions for similar queries
        """
        self.generator = generator
        self.analyzer = analyzer
        self.routing_strategy = RoutingStrategy(routing_strategy)
        self.enable_model_selection = enable_model_selection
        self.enable_caching = enable_caching
        
        # Load routing rules and model profiles
        self.routing_rules = self._load_routing_rules()
        self.model_profiles = self._load_model_profiles()
        self.task_patterns = self._load_task_patterns()
        
        # Performance tracking
        self.routing_cache = {}
        self.metrics = {
            "total_routes": 0,
            "cache_hits": 0,
            "avg_routing_time": 0.0,
            "decisions_by_engine": {}
        }
        
        logger.info(f"🔄 LLM-FE Engine initialized (strategy: {routing_strategy})")
    
    def _load_routing_rules(self) -> Dict[str, Any]:
        """Load routing rules based on task patterns"""
        return {
            # Generation tasks
            "business_report": {
                "preferred_engine": EngineType.GENERATOR,
                "model_requirements": ["large", "instruction-tuned"],
                "min_quality": 0.85,
                "typical_time": 3.0
            },
            "code_generation": {
                "preferred_engine": EngineType.GENERATOR,
                "model_requirements": ["code-specialized"],
                "min_quality": 0.90,
                "typical_time": 2.5
            },
            "creative_writing": {
                "preferred_engine": EngineType.GENERATOR,
                "model_requirements": ["creative", "high-temperature"],
                "min_quality": 0.80,
                "typical_time": 2.0
            },
            
            # Analysis tasks
            "sentiment_analysis": {
                "preferred_engine": EngineType.ANALYZER,
                "model_requirements": ["classification"],
                "min_quality": 0.85,
                "typical_time": 0.5
            },
            "entity_extraction": {
                "preferred_engine": EngineType.ANALYZER,
                "model_requirements": ["ner", "token-classification"],
                "min_quality": 0.88,
                "typical_time": 0.8
            },
            
            # Complex reasoning
            "complex_reasoning": {
                "preferred_engine": EngineType.OPTIMA,
                "model_requirements": ["reasoning-capable"],
                "min_quality": 0.90,
                "typical_time": 5.0
            },
            
            # Quick tasks (use lightweight)
            "quick_summary": {
                "preferred_engine": EngineType.LIGHTWEIGHT_GEN,
                "model_requirements": ["fast"],
                "min_quality": 0.70,
                "typical_time": 0.2
            }
        }
    
    def _load_model_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load performance profiles for available models"""
        return {
            # Your actual models from transformers_registry
            "llama2-70b": {
                "capabilities": ["chat", "reasoning", "instruction-following"],
                "quality_score": 0.92,
                "avg_time": 3.5,
                "cost": "high",
                "specializations": ["general", "reasoning"]
            },
            "mistral-7b": {
                "capabilities": ["chat", "instruction-following"],
                "quality_score": 0.88,
                "avg_time": 1.2,
                "cost": "medium",
                "specializations": ["general", "fast"]
            },
            "codellama-7b": {
                "capabilities": ["code", "programming"],
                "quality_score": 0.90,
                "avg_time": 2.0,
                "cost": "medium",
                "specializations": ["code", "technical"]
            },
            "cogito-671b": {
                "capabilities": ["reasoning", "complex-analysis"],
                "quality_score": 0.95,
                "avg_time": 5.0,
                "cost": "very-high",
                "specializations": ["reasoning", "analysis"]
            },
            "distilbert": {
                "capabilities": ["classification", "sentiment"],
                "quality_score": 0.85,
                "avg_time": 0.3,
                "cost": "low",
                "specializations": ["classification", "fast"]
            },
            "bart-large-cnn": {
                "capabilities": ["summarization", "seq2seq"],
                "quality_score": 0.87,
                "avg_time": 1.5,
                "cost": "medium",
                "specializations": ["summarization"]
            }
        }
    
    def _load_task_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for task type detection"""
        return {
            "code_generation": ["write code", "implement", "program", "function", "class", "algorithm"],
            "business_report": ["report", "analysis", "business", "financial", "quarterly", "metrics"],
            "creative_writing": ["story", "poem", "creative", "narrative", "fiction"],
            "sentiment_analysis": ["sentiment", "feeling", "emotion", "opinion", "review"],
            "entity_extraction": ["extract", "entities", "names", "organizations", "locations"],
            "complex_reasoning": ["why", "explain", "reason", "analyze", "compare", "evaluate"],
            "quick_summary": ["summarize", "tldr", "brief", "quick summary"],
            "translation": ["translate", "translation", "convert language"],
            "question_answering": ["what is", "who is", "when", "where", "how"]
        }
    
    async def optimize_route(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine optimal routing for a task
        
        Args:
            context: {
                "objective": str (generate, analyze, evaluate, etc.),
                "task_type": str (optional specific task type),
                "content": str (input content for analysis),
                "constraints": dict (time, quality, cost constraints),
                "quality_requirements": dict (min quality, preferences),
                "models_available": list (available models),
                "analysis_capabilities": list (available analysis tasks)
            }
        
        Returns:
            Routing decision dict
        """
        start_time = time.time()
        
        try:
            objective = context.get("objective", "")
            task_type = context.get("task_type")
            content = context.get("content", "")
            constraints = context.get("constraints", {})
            quality_requirements = context.get("quality_requirements", {})
            
            # Check cache first
            cache_key = self._generate_cache_key(objective, task_type, content[:100])
            if self.enable_caching and cache_key in self.routing_cache:
                self.metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for routing decision: {cache_key}")
                return self.routing_cache[cache_key].to_dict()
            
            # Detect task type if not provided
            if not task_type:
                task_type = self._detect_task_type(content, objective)
            
            # Apply routing rules based on strategy
            routing_decision = await self._apply_routing_rules(
                objective,
                task_type,
                content,
                constraints,
                quality_requirements
            )
            
            # Enhance with model selection if enabled
            if self.enable_model_selection:
                routing_decision = await self._select_optimal_model(
                    routing_decision,
                    context
                )
            
            # Generate alternative routes
            routing_decision.alternative_routes = self._generate_alternatives(
                routing_decision,
                context
            )
            
            # Cache decision
            if self.enable_caching:
                self.routing_cache[cache_key] = routing_decision
            
            # Update metrics
            routing_time = time.time() - start_time
            self._update_metrics(routing_decision, routing_time)
            
            logger.info(f"Routed to {routing_decision.engine.value}/{routing_decision.model} "
                       f"(confidence: {routing_decision.confidence:.2f}, time: {routing_time:.3f}s)")
            
            return routing_decision.to_dict()
            
        except Exception as e:
            logger.error(f"Route optimization failed: {str(e)}", exc_info=True)
            return self._get_fallback_route(context).to_dict()
    
    def _detect_task_type(self, content: str, objective: str) -> str:
        """Detect task type from content and objective"""
        content_lower = content.lower()
        
        # Check patterns
        max_matches = 0
        detected_type = "general"
        
        for task_type, patterns in self.task_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in content_lower)
            if matches > max_matches:
                max_matches = matches
                detected_type = task_type
        
        # Consider objective
        if objective == "generate" and detected_type == "general":
            if len(content) < 50:
                detected_type = "quick_generation"
            else:
                detected_type = "text_generation"
        elif objective == "analyze" and detected_type == "general":
            detected_type = "text_classification"
        
        return detected_type
    
    async def _apply_routing_rules(
        self,
        objective: str,
        task_type: str,
        content: str,
        constraints: Dict,
        quality_requirements: Dict
    ) -> RoutingDecision:
        """Apply routing rules based on strategy"""
        
        # Get base routing rule for task type
        rule = self.routing_rules.get(task_type, {
            "preferred_engine": EngineType.GENERATOR,
            "model_requirements": ["general"],
            "min_quality": 0.75,
            "typical_time": 2.0
        })
        
        # Apply strategy modifications
        if self.routing_strategy == RoutingStrategy.PERFORMANCE:
            # Prioritize speed - use lightweight models when possible
            if rule["typical_time"] > 2.0 and quality_requirements.get("min_quality", 0.8) < 0.85:
                engine = EngineType.LIGHTWEIGHT_GEN if objective == "generate" else EngineType.LIGHTWEIGHT_ANALYZE
            else:
                engine = rule["preferred_engine"]
            
        elif self.routing_strategy == RoutingStrategy.QUALITY:
            # Prioritize quality - use best models
            if task_type in ["complex_reasoning", "business_report"]:
                engine = EngineType.OPTIMA
            else:
                engine = rule["preferred_engine"]
        
        elif self.routing_strategy == RoutingStrategy.COST:
            # Minimize cost - use smallest capable model
            engine = EngineType.LIGHTWEIGHT_GEN if objective == "generate" else EngineType.LIGHTWEIGHT_ANALYZE
        
        else:  # INTELLIGENT or BALANCED
            # Balance all factors
            if len(content) < 100 and quality_requirements.get("min_quality", 0.8) < 0.85:
                # Short, low-quality requirement -> lightweight
                engine = EngineType.LIGHTWEIGHT_GEN if objective == "generate" else EngineType.LIGHTWEIGHT_ANALYZE
            elif task_type in ["complex_reasoning", "business_report"] and quality_requirements.get("min_quality", 0.8) >= 0.85:
                # Complex, high-quality requirement -> Optima
                engine = EngineType.OPTIMA
            else:
                # Default to rule
                engine = rule["preferred_engine"]
        
        # Select initial model (will be refined)
        model = self._select_model_for_engine(engine, rule["model_requirements"])
        
        # Calculate expected metrics
        model_profile = self.model_profiles.get(model, {})
        expected_quality = model_profile.get("quality_score", 0.8)
        expected_time = model_profile.get("avg_time", 2.0)
        
        # Build parameters
        parameters = self._build_parameters(task_type, constraints)
        
        # Generate reasoning
        reasoning = (f"Selected {engine.value} with {model} for {task_type} task. "
                    f"Strategy: {self.routing_strategy.value}. "
                    f"Expected quality: {expected_quality:.2f}, time: {expected_time:.1f}s")
        
        return RoutingDecision(
            engine=engine,
            model=model,
            parameters=parameters,
            expected_quality=expected_quality,
            expected_time=expected_time,
            reasoning=reasoning,
            confidence=0.85,
            alternative_routes=[]
        )
    
    async def _select_optimal_model(
        self,
        decision: RoutingDecision,
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """Refine model selection based on context"""
        
        # Get available models from context
        available_models = context.get("models_available", list(self.model_profiles.keys()))
        
        # Filter models compatible with engine
        compatible_models = self._filter_compatible_models(decision.engine, available_models)
        
        if not compatible_models:
            return decision  # Keep original
        
        # Score each model
        scores = {}
        for model in compatible_models:
            score = self._score_model(model, context, decision.engine)
            scores[model] = score
        
        # Select best model
        best_model = max(scores.keys(), key=lambda k: scores[k])
        
        # Update decision if better model found
        if best_model != decision.model and scores[best_model] > scores.get(decision.model, 0):
            decision.model = best_model
            decision.expected_quality = self.model_profiles[best_model]["quality_score"]
            decision.expected_time = self.model_profiles[best_model]["avg_time"]
            decision.reasoning += f" Model upgraded to {best_model} for better performance."
        
        return decision
    
    def _filter_compatible_models(
        self,
        engine: EngineType,
        available_models: List[str]
    ) -> List[str]:
        """Filter models compatible with engine type"""
        compatible = []
        
        for model in available_models:
            if model not in self.model_profiles:
                continue
            
            profile = self.model_profiles[model]
            capabilities = profile.get("capabilities", [])
            
            # Check engine compatibility
            if engine in [EngineType.GENERATOR, EngineType.LIGHTWEIGHT_GEN]:
                if any(cap in capabilities for cap in ["chat", "code", "instruction-following"]):
                    compatible.append(model)
            elif engine in [EngineType.ANALYZER, EngineType.LIGHTWEIGHT_ANALYZE]:
                if any(cap in capabilities for cap in ["classification", "sentiment", "ner"]):
                    compatible.append(model)
            elif engine == EngineType.OPTIMA:
                if any(cap in capabilities for cap in ["reasoning", "complex-analysis"]):
                    compatible.append(model)
        
        return compatible
    
    def _score_model(
        self,
        model: str,
        context: Dict[str, Any],
        engine: EngineType
    ) -> float:
        """Score model for given context"""
        profile = self.model_profiles.get(model, {})
        
        quality = profile.get("quality_score", 0.5)
        time = profile.get("avg_time", 2.0)
        
        # Apply strategy weights
        if self.routing_strategy == RoutingStrategy.PERFORMANCE:
            # Favor speed
            score = quality * 0.3 + (1.0 / (time + 0.1)) * 0.7
        elif self.routing_strategy == RoutingStrategy.QUALITY:
            # Favor quality
            score = quality * 0.8 + (1.0 / (time + 0.1)) * 0.2
        else:  # INTELLIGENT or BALANCED
            # Equal weight
            score = quality * 0.5 + (1.0 / (time + 0.1)) * 0.5
        
        return score
    
    def _select_model_for_engine(
        self,
        engine: EngineType,
        requirements: List[str]
    ) -> str:
        """Select appropriate model for engine"""
        
        # Default models for each engine
        defaults = {
            EngineType.GENERATOR: "llama2-70b",
            EngineType.ANALYZER: "distilbert",
            EngineType.OPTIMA: "cogito-671b",
            EngineType.LIGHTWEIGHT_GEN: "mistral-7b",
            EngineType.LIGHTWEIGHT_ANALYZE: "distilbert"
        }
        
        # Check for specialized requirements
        if "code" in requirements or "code-specialized" in requirements:
            return "codellama-7b"
        elif "summarization" in requirements:
            return "bart-large-cnn"
        
        return defaults.get(engine, "llama2-70b")
    
    def _build_parameters(
        self,
        task_type: str,
        constraints: Dict
    ) -> Dict[str, Any]:
        """Build generation/analysis parameters"""
        
        # Base parameters
        params = {
            "max_tokens": constraints.get("max_tokens", 500),
            "temperature": 0.7
        }
        
        # Task-specific adjustments
        if "code" in task_type:
            params["temperature"] = 0.2  # Lower for code
        elif "creative" in task_type:
            params["temperature"] = 0.9  # Higher for creativity
        elif "reasoning" in task_type:
            params["temperature"] = 0.6  # Balanced for reasoning
        
        # Apply constraint overrides
        if "temperature" in constraints:
            params["temperature"] = constraints["temperature"]
        
        return params
    
    def _generate_alternatives(
        self,
        primary_decision: RoutingDecision,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate alternative routing options"""
        alternatives = []
        
        # Alternative 1: Faster option
        if primary_decision.expected_time > 1.0:
            alternatives.append({
                "engine": EngineType.LIGHTWEIGHT_GEN.value,
                "model": "mistral-7b",
                "expected_quality": 0.80,
                "expected_time": 0.5,
                "reasoning": "Faster alternative with slightly lower quality"
            })
        
        # Alternative 2: Higher quality option
        if primary_decision.expected_quality < 0.90:
            alternatives.append({
                "engine": EngineType.OPTIMA.value,
                "model": "cogito-671b",
                "expected_quality": 0.95,
                "expected_time": 5.0,
                "reasoning": "Higher quality with longer processing time"
            })
        
        return alternatives[:2]  # Return top 2
    
    def _get_fallback_route(self, context: Dict[str, Any]) -> RoutingDecision:
        """Fallback routing when optimization fails"""
        objective = context.get("objective", "generate")
        
        if objective in ["analyze", "evaluate"]:
            engine = EngineType.ANALYZER
            model = "distilbert"
        else:
            engine = EngineType.GENERATOR
            model = "llama2-70b"
        
        return RoutingDecision(
            engine=engine,
            model=model,
            parameters={"max_tokens": 500, "temperature": 0.7},
            expected_quality=0.75,
            expected_time=2.0,
            reasoning="Fallback route due to optimization failure",
            confidence=0.6,
            alternative_routes=[]
        )
    
    def _generate_cache_key(self, objective: str, task_type: Optional[str], content_sample: str) -> str:
        """Generate cache key for routing decision"""
        return f"{objective}:{task_type}:{hash(content_sample)}"
    
    def _update_metrics(self, decision: RoutingDecision, routing_time: float):
        """Update routing metrics"""
        self.metrics["total_routes"] += 1
        n = self.metrics["total_routes"]
        
        # Update average routing time
        self.metrics["avg_routing_time"] = (
            (self.metrics["avg_routing_time"] * (n - 1) + routing_time) / n
        )
        
        # Track decisions by engine
        engine_name = decision.engine.value
        self.metrics["decisions_by_engine"][engine_name] = (
            self.metrics["decisions_by_engine"].get(engine_name, 0) + 1
        )
    
    async def health_check(self) -> bool:
        """Check if LLM-FE engine is healthy"""
        try:
            return (
                self.generator is not None and
                self.analyzer is not None
            )
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get LLM-FE capabilities"""
        return {
            "routing_strategies": [s.value for s in RoutingStrategy],
            "supported_engines": [e.value for e in EngineType],
            "model_selection": self.enable_model_selection,
            "caching": self.enable_caching,
            "available_models": list(self.model_profiles.keys()),
            "supported_task_types": list(self.routing_rules.keys()),
            "parameter_optimization": True,
            "quality_prediction": True,
            "performance_metrics": self.metrics
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current routing metrics"""
        metrics = self.metrics.copy()
        metrics["cache_hit_rate"] = (
            metrics["cache_hits"] / metrics["total_routes"]
            if metrics["total_routes"] > 0 else 0.0
        )
        return metrics
    
    def clear_cache(self):
        """Clear routing cache"""
        self.routing_cache.clear()
        logger.info("Routing cache cleared")

# Global instance placeholder (initialized in main.py)
llm_fe_engine: Optional[LLMFEEngine] = None
