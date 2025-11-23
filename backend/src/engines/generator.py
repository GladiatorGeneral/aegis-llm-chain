"""Universal Hugging Face generator supporting multiple tasks."""

try:
    import torch
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        GenerationConfig
    )
    from huggingface_hub import HfApi, InferenceClient
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    pipeline = None

from typing import Dict, List, Any, Optional
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from engines.base import BaseGenerator, GenerationRequest, GenerationResponse, GenerationTask, ModelProvider
from core.security import security_layer
from engines.distributed import distributed_engine, DistributedConfig, ParallelismStrategy, CommunicationBackend

logger = logging.getLogger(__name__)

class HuggingFaceGenerator(BaseGenerator):
    """Universal Hugging Face generator supporting multiple tasks"""
    
    def __init__(self, hf_token: Optional[str] = None, device: str = "auto", enable_distributed: bool = False):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required for HuggingFaceGenerator. Install with: pip install transformers torch")
        
        self.hf_token = hf_token
        self.device = device
        self.enable_distributed = enable_distributed
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize distributed engine if enabled
        if enable_distributed:
            self.distributed_config = DistributedConfig(
                parallelism_strategy=ParallelismStrategy.TENSOR_PARALLELISM,
                communication_backend=CommunicationBackend.NVSHMEM,
                num_nodes=1,  # Configurable
                gpus_per_node=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                hierarchical_all_reduce=True
            )
        
        # Model registry - maps tasks to recommended models
        self.model_registry = {
            GenerationTask.TEXT_COMPLETION: {
                "model_id": "microsoft/DialoGPT-medium",
                "description": "General text completion"
            },
            GenerationTask.CHAT: {
                "model_id": "microsoft/DialoGPT-medium", 
                "description": "Conversational AI"
            },
            GenerationTask.CODE_GENERATION: {
                "model_id": "codellama/CodeLlama-7b-hf",
                "description": "Code generation"
            },
            GenerationTask.TEXT_SUMMARIZATION: {
                "model_id": "facebook/bart-large-cnn",
                "description": "Text summarization"
            },
            GenerationTask.TEXT_TRANSLATION: {
                "model_id": "Helsinki-NLP/opus-mt-en-fr",
                "description": "English to French translation"
            }
        }
        
        # Loaded models cache
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        
        logger.info(f"HuggingFaceGenerator initialized with device: {device}, distributed: {enable_distributed}")
    
    async def initialize_distributed(self):
        """Initialize distributed inference engine"""
        if self.enable_distributed:
            await distributed_engine.initialize_distributed(self.distributed_config)
            logger.info("Distributed inference engine initialized")
    
    def _should_use_distributed(self, model_id: str, task: GenerationTask) -> bool:
        """Determine if distributed inference should be used"""
        # Use distributed for large models or compute-intensive tasks
        large_models = ["codellama/CodeLlama-7b-hf", "meta-llama/Llama-2-7b-chat-hf"]
        compute_intensive_tasks = [GenerationTask.CODE_GENERATION, GenerationTask.TEXT_SUMMARIZATION]
        
        return (
            model_id in large_models or 
            task in compute_intensive_tasks or
            any(size in model_id for size in ["7b", "13b", "70b", "405b"])  # Model size indicators
        )
    
    async def _load_model_shards(self, model_id: str, task: GenerationTask) -> List[Any]:
        """Load model shards for tensor parallelism"""
        if model_id not in self.loaded_models:
            await self._load_model(model_id, task)
        
        # In production, this would split the model across multiple GPUs/nodes
        # For now, return single shard (simulated distribution)
        return [self.loaded_models[model_id]]
    
    async def _generate_text_distributed(self, prompt: str, model_id: str, 
                                       parameters: Dict[str, Any], task: GenerationTask) -> Dict[str, Any]:
        """Generate text using distributed model parallelism"""
        if not self.enable_distributed:
            return await self._generate_text(prompt, model_id, parameters, task)
        
        logger.info(f"Using distributed inference for {model_id}")
        
        # Load model shards for tensor parallelism
        model_shards = await self._load_model_shards(model_id, task)
        
        # Prepare inputs
        tokenizer = self.loaded_tokenizers[model_id]
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Convert inputs to tensor
        input_tensor = inputs['input_ids'].float() if hasattr(inputs['input_ids'], 'float') else torch.tensor([[1.0]])
        
        # Distributed forward pass
        start_time = time.time()
        outputs = await distributed_engine.model_parallel_forward(input_tensor, model_shards)
        generation_time = time.time() - start_time
        
        # Decode output (simulated for distributed)
        generated_text = f"[Distributed generation result for: {prompt[:50]}...]"
        
        # Get communication statistics
        comm_stats = distributed_engine.get_communication_stats()
        
        return {
            "content": generated_text.strip(),
            "tokens_used": outputs.nelement() if hasattr(outputs, 'nelement') else 50,
            "metadata": {
                "task": task.value,
                "model_id": model_id,
                "distributed": True,
                "communication_stats": comm_stats,
                "generation_time": generation_time
            }
        }
    
    def supports_task(self, task: GenerationTask) -> bool:
        return task in self.model_registry
    
    def get_supported_models(self) -> List[str]:
        return list(set([config["model_id"] for config in self.model_registry.values()]))
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Execute generation with security and safety checks"""
        start_time = time.time()
        
        # Validate request
        self.validate_request(request)
        
        # Security validation
        if request.safety_checks:
            is_valid, error_msg = await security_layer.validate_input(request.prompt)
            if not is_valid:
                return GenerationResponse(
                    content="",
                    model_used="security_layer",
                    provider=request.provider,
                    latency=0.0,
                    safety_flagged=True,
                    safety_reason=f"Security violation: {error_msg}"
                )
        
        try:
            # Get model ID
            model_id = request.model_id or self.model_registry[request.task]["model_id"]
            
            # Use distributed inference for large models or when explicitly requested
            use_distributed = (
                self.enable_distributed and 
                request.parameters.get("use_distributed", False) and
                self._should_use_distributed(model_id, request.task)
            )
            
            # Execute generation based on task
            if use_distributed:
                result = await self._generate_text_distributed(
                    request.prompt, model_id, request.parameters, request.task
                )
            elif request.task == GenerationTask.TEXT_SUMMARIZATION:
                result = await self._summarize_text(request.prompt, model_id, request.parameters)
            elif request.task == GenerationTask.CODE_GENERATION:
                result = await self._generate_code(request.prompt, model_id, request.parameters)
            elif request.task == GenerationTask.TEXT_TRANSLATION:
                result = await self._translate_text(request.prompt, model_id, request.parameters)
            else:
                result = await self._generate_text(request.prompt, model_id, request.parameters, request.task)
            
            # Safety check output
            sanitized_content = await security_layer.sanitize_output(result["content"])
            safety_flagged = sanitized_content != result["content"]
            
            latency = time.time() - start_time
            
            return GenerationResponse(
                content=sanitized_content,
                model_used=model_id,
                provider=request.provider.value,
                latency=latency,
                tokens_used=result.get("tokens_used"),
                metadata=result.get("metadata", {}),
                safety_flagged=safety_flagged,
                safety_reason="PII detected and redacted" if safety_flagged else None
            )
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            latency = time.time() - start_time
            return GenerationResponse(
                content=f"Error during generation: {str(e)}",
                model_used=request.model_id or "unknown",
                provider=request.provider.value,
                latency=latency,
                safety_flagged=True,
                safety_reason="Generation failed"
            )
    
    async def _generate_text(self, prompt: str, model_id: str, parameters: Dict[str, Any], task: GenerationTask) -> Dict[str, Any]:
        """Generate text using causal LM"""
        loop = asyncio.get_event_loop()
        
        # Get or load model and tokenizer
        if model_id not in self.loaded_models:
            await self._load_model(model_id, task)
        
        tokenizer = self.loaded_tokenizers[model_id]
        model = self.loaded_models[model_id]
        
        # Prepare generation parameters
        gen_config = GenerationConfig(
            max_new_tokens=parameters.get("max_tokens", 100),
            temperature=parameters.get("temperature", 0.7),
            top_p=parameters.get("top_p", 0.9),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        def _generate():
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            return outputs
        
        outputs = await loop.run_in_executor(self.executor, _generate)
        
        # Decode output
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "content": generated_text.strip(),
            "tokens_used": len(generated_tokens),
            "metadata": {
                "task": task.value,
                "model_id": model_id,
                "parameters_used": parameters
            }
        }
    
    async def _summarize_text(self, text: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize text using seq2seq model"""
        loop = asyncio.get_event_loop()
        
        if model_id not in self.loaded_models:
            await self._load_summarization_model(model_id)
        
        summarizer = self.loaded_models[model_id]
        
        def _summarize():
            return summarizer(
                text,
                max_length=parameters.get("max_length", 130),
                min_length=parameters.get("min_length", 30),
                do_sample=False
            )
        
        result = await loop.run_in_executor(self.executor, _summarize)
        
        return {
            "content": result[0]['summary_text'],
            "metadata": {
                "task": "summarization",
                "model_id": model_id
            }
        }
    
    async def _generate_code(self, prompt: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code using code-specific model"""
        # Use text generation but with code-specific parameters
        code_parameters = {
            "max_tokens": parameters.get("max_tokens", 200),
            "temperature": parameters.get("temperature", 0.2),  # Lower temp for code
            "top_p": parameters.get("top_p", 0.95),
        }
        
        result = await self._generate_text(prompt, model_id, code_parameters, GenerationTask.CODE_GENERATION)
        result["metadata"]["task"] = "code_generation"
        return result
    
    async def _translate_text(self, text: str, model_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text using translation model"""
        loop = asyncio.get_event_loop()
        
        if model_id not in self.loaded_models:
            await self._load_translation_model(model_id)
        
        translator = self.loaded_models[model_id]
        
        def _translate():
            return translator(text)
        
        result = await loop.run_in_executor(self.executor, _translate)
        
        return {
            "content": result[0]['translation_text'],
            "metadata": {
                "task": "translation", 
                "model_id": model_id
            }
        }
    
    async def _load_model(self, model_id: str, task: GenerationTask):
        """Load a causal LM model for text generation"""
        logger.info(f"Loading model: {model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.loaded_models[model_id] = model
            self.loaded_tokenizers[model_id] = tokenizer
            
            logger.info(f"Successfully loaded model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise
    
    async def _load_summarization_model(self, model_id: str):
        """Load a summarization model"""
        logger.info(f"Loading summarization model: {model_id}")
        
        try:
            summarizer = pipeline(
                "summarization",
                model=model_id,
                tokenizer=model_id,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.loaded_models[model_id] = summarizer
            logger.info(f"Successfully loaded summarization model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load summarization model {model_id}: {str(e)}")
            raise
    
    async def _load_translation_model(self, model_id: str):
        """Load a translation model"""
        logger.info(f"Loading translation model: {model_id}")
        
        try:
            translator = pipeline(
                "translation",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.loaded_models[model_id] = translator
            logger.info(f"Successfully loaded translation model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load translation model {model_id}: {str(e)}")
            raise

# Global generator instance - create only if transformers available
if TRANSFORMERS_AVAILABLE:
    try:
        universal_generator = HuggingFaceGenerator()
        logger.info("HuggingFaceGenerator initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize HuggingFaceGenerator: {e}")
        universal_generator = None
else:
    logger.warning("Transformers not available - HuggingFaceGenerator disabled")
    universal_generator = None
