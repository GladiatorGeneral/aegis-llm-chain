"""
Optimized Multi-Model Runner with Parallel Execution
Based on PyTorch best practices for Hugging Face Transformers
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import torch and transformers - make them optional
try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        pipeline,
        BitsAndBytesConfig
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch/transformers not available - OptimizedMultiModelRunner will not work")

class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCHED = "batched"

class QuantizationConfig(str, Enum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"

class OptimizedMultiModelRunner:
    """
    High-performance model runner with parallel execution, quantization, and GPU optimization
    """
    
    def __init__(
        self,
        model_configs: Dict[str, Dict[str, Any]],
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL,
        quantization: QuantizationConfig = QuantizationConfig.NONE,
        max_workers: int = 4,
        device: str = "auto"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers required for OptimizedMultiModelRunner")
        
        self.model_configs = model_configs
        self.execution_mode = execution_mode
        self.quantization = quantization
        self.max_workers = max_workers
        self.device = self._setup_device(device)
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Model cache
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        
        logger.info(f"OptimizedMultiModelRunner initialized with {len(model_configs)} models on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device with GPU optimization"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                # Enable GPU memory optimization
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
            else:
                device = "cpu"
        return device
    
    def _get_quantization_config(self) -> Optional[Any]:
        """Get quantization configuration for memory optimization"""
        if self.quantization == QuantizationConfig.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        elif self.quantization == QuantizationConfig.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        return None
    
    def load_models(self):
        """Load all configured models (called separately to avoid long init)"""
        for model_name, config in self.model_configs.items():
            self._load_model(model_name, config)
    
    def _load_model(self, model_name: str, config: Dict[str, Any]):
        """Load a single model with optimization"""
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Get model type and task
            task = config.get("task", "text-generation")
            
            # Setup quantization
            quant_config = self._get_quantization_config()
            
            if task in ["text-classification", "token-classification", "summarization", "translation"]:
                # Use pipeline for specific tasks
                self.pipelines[model_name] = pipeline(
                    task,
                    model=config.get("model_name", model_name),
                    tokenizer=config.get("model_name", model_name),
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            # Don't raise - allow partial loading
    
    async def predict_sequential(self, text: str, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Sequential prediction across models"""
        results = {}
        model_names = model_names or list(self.model_configs.keys())
        
        for model_name in model_names:
            try:
                result = await self._predict_single(model_name, text)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    async def predict_parallel(self, text: str, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parallel prediction across models using asyncio"""
        model_names = model_names or list(self.model_configs.keys())
        
        async def _predict_wrapper(model_name):
            try:
                result = await self._predict_single(model_name, text)
                return model_name, result
            except Exception as e:
                logger.error(f"Parallel prediction failed for {model_name}: {str(e)}")
                return model_name, {"error": str(e)}
        
        # Create tasks for all models
        tasks = [_predict_wrapper(model_name) for model_name in model_names]
        results_list = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        return dict(results_list)
    
    async def predict_batch(self, texts: List[str], model_names: Optional[List[str]] = None) -> Dict[str, List[Any]]:
        """Batch prediction for multiple texts"""
        model_names = model_names or list(self.model_configs.keys())
        results = {model_name: [] for model_name in model_names}
        
        for model_name in model_names:
            try:
                if model_name in self.pipelines:
                    # Use pipeline for batch processing
                    batch_results = self.pipelines[model_name](texts)
                    results[model_name] = batch_results
                else:
                    results[model_name] = [{"error": "Model not loaded"}] * len(texts)
                        
            except Exception as e:
                logger.error(f"Batch prediction failed for {model_name}: {str(e)}")
                results[model_name] = [{"error": str(e)}] * len(texts)
        
        return results
    
    async def _predict_single(self, model_name: str, text: str) -> Any:
        """Predict using a single model"""
        loop = asyncio.get_event_loop()
        
        if model_name in self.pipelines:
            # Use pipeline
            def _pipeline_predict():
                return self.pipelines[model_name](text)
            
            return await loop.run_in_executor(self.executor, _pipeline_predict)
        else:
            return {"error": "Model not loaded"}
    
    async def predict(
        self, 
        text: Union[str, List[str]], 
        model_names: Optional[List[str]] = None,
        execution_mode: Optional[ExecutionMode] = None
    ) -> Any:
        """Main prediction method with mode selection"""
        execution_mode = execution_mode or self.execution_mode
        
        if isinstance(text, list):
            # Batch processing
            return await self.predict_batch(text, model_names)
        else:
            # Single text processing
            if execution_mode == ExecutionMode.PARALLEL:
                return await self.predict_parallel(text, model_names)
            else:
                return await self.predict_sequential(text, model_names)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "total_models": len(self.models) + len(self.pipelines),
            "loaded_models": list(self.models.keys()) + list(self.pipelines.keys()),
            "execution_mode": self.execution_mode.value if hasattr(self.execution_mode, 'value') else str(self.execution_mode),
            "quantization": self.quantization.value if hasattr(self.quantization, 'value') else str(self.quantization),
            "device": self.device,
            "memory_usage": None
        }
        
        if self.device == "cuda" and TORCH_AVAILABLE:
            try:
                info["memory_usage"] = {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "cached": torch.cuda.memory_reserved() / 1024**3,      # GB
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
                }
            except:
                pass
        
        return info

# Pre-configured model configurations for our AGI platform
DEFAULT_MODEL_CONFIGS = {
    # Sentiment Analysis
    "sentiment-distilbert": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "model_type": "classification",
        "task": "text-classification"
    },
    "sentiment-roberta": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "model_type": "classification", 
        "task": "text-classification"
    },
    
    # Entity Recognition
    "ner-bert": {
        "model_name": "dslim/bert-base-NER",
        "model_type": "token_classification",
        "task": "token-classification"
    },
    
    # Summarization
    "summarization-bart": {
        "model_name": "facebook/bart-large-cnn",
        "model_type": "seq2seq",
        "task": "summarization"
    },
    
    # Translation
    "translation-marian": {
        "model_name": "Helsinki-NLP/opus-mt-en-fr",
        "model_type": "seq2seq",
        "task": "translation"
    }
}

# Global optimized runner instance - create only if torch available
if TORCH_AVAILABLE:
    try:
        optimized_runner = OptimizedMultiModelRunner(
            model_configs=DEFAULT_MODEL_CONFIGS,
            execution_mode=ExecutionMode.PARALLEL,
            quantization=QuantizationConfig.INT8 if torch.cuda.is_available() else QuantizationConfig.NONE,
            max_workers=4
        )
        logger.info("Global optimized_runner created successfully")
    except Exception as e:
        logger.warning(f"Could not create global optimized_runner: {e}")
        optimized_runner = None
else:
    optimized_runner = None
    logger.warning("optimized_runner not available - torch/transformers not installed")
