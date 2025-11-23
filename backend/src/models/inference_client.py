"""
Unified Inference Client
Handles both Hugging Face Inference API and local model inference
"""
import os
from typing import Dict, List, Optional, Any, Union
import logging
import asyncio

# Hugging Face Hub client
try:
    from huggingface_hub import InferenceClient, AsyncInferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not installed - API inference disabled")

# Transformers for local inference
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed - local inference disabled")

from .registry import ModelConfig, ModelType, ModelProvider, model_registry

logger = logging.getLogger(__name__)

class UnifiedInferenceClient:
    """
    Unified client that handles both API and local model inference
    Automatically routes requests based on model configuration
    """
    
    def __init__(self):
        self.hf_client = None
        self.async_hf_client = None
        self.local_models: Dict[str, Any] = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize inference clients"""
        # Initialize Hugging Face Inference Client
        if HF_HUB_AVAILABLE:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    self.hf_client = InferenceClient(token=hf_token)
                    self.async_hf_client = AsyncInferenceClient(token=hf_token)
                    logger.info("✅ Hugging Face Inference Client initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize HF client: {str(e)}")
            else:
                logger.warning("❌ HF_TOKEN not found - Hugging Face API disabled")
        
        # Check for local inference capabilities
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"✅ Local inference enabled with {torch.cuda.device_count()} GPU(s)")
        elif TRANSFORMERS_AVAILABLE:
            logger.info("✅ Local inference enabled (CPU only)")
        else:
            logger.warning("❌ Transformers not available - local inference disabled")
    
    def _load_local_model(self, model_config: ModelConfig):
        """Load a model locally for GPU inference"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available for local inference")
        
        try:
            logger.info(f"🔄 Loading local model: {model_config.model_id}")
            
            # Check if model is already loaded
            if model_config.model_id in self.local_models:
                logger.info(f"✅ Model already loaded: {model_config.model_id}")
                return self.local_models[model_config.model_id]
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.local_path or model_config.model_id,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            
            # Configure quantization
            quantization_config = None
            if model_config.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif model_config.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load model with optimizations
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "token": os.getenv("HF_TOKEN")
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch.float16
            
            model = AutoModelForCausalLM.from_pretrained(
                model_config.local_path or model_config.model_id,
                **model_kwargs
            )
            
            # Store in cache
            self.local_models[model_config.model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": model_config
            }
            
            logger.info(f"✅ Local model loaded: {model_config.model_id}")
            logger.info(f"   Quantization: {model_config.quantization or 'None'}")
            logger.info(f"   Device: {next(model.parameters()).device}")
            
            return self.local_models[model_config.model_id]
            
        except Exception as e:
            logger.error(f"❌ Failed to load local model {model_config.model_id}: {str(e)}")
            raise
    
    async def chat_completion(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified chat completion interface
        Works with both API and local models
        
        Args:
            model_key: Key from model registry
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, top_p, etc.)
        
        Returns:
            Dict with 'content', 'model', 'usage', 'finish_reason'
        """
        model_config = model_registry.get_model(model_key)
        if not model_config:
            raise ValueError(f"Model '{model_key}' not found in registry")
        
        # Default parameters
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', model_config.max_tokens)
        top_p = kwargs.get('top_p', 0.9)
        
        # Route to appropriate backend
        if model_config.is_local:
            return await self._local_chat_completion(
                model_config, messages, temperature, max_tokens, top_p
            )
        else:
            return await self._hf_chat_completion(
                model_config, messages, temperature, max_tokens, top_p
            )
    
    async def _hf_chat_completion(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> Dict[str, Any]:
        """Hugging Face Inference API completion"""
        if not self.hf_client:
            raise RuntimeError(
                "Hugging Face client not initialized. "
                "Please set HF_TOKEN environment variable."
            )
        
        try:
            logger.info(f"🔄 HF API request: {model_config.model_id}")
            
            # Use async client for better performance
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.hf_client.chat.completions.create(
                    model=model_config.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=False
                )
            )
            
            logger.info(f"✅ HF API response received: {model_config.model_id}")
            
            return {
                "content": completion.choices[0].message.content,
                "model": model_config.model_id,
                "model_key": model_config.name,
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                    "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                    "total_tokens": completion.usage.total_tokens if completion.usage else 0
                },
                "finish_reason": completion.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"❌ HF API error for {model_config.model_id}: {str(e)}")
            raise RuntimeError(f"Inference API failed: {str(e)}")
    
    async def _local_chat_completion(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> Dict[str, Any]:
        """Local model inference completion"""
        try:
            logger.info(f"🔄 Local inference: {model_config.model_id}")
            
            # Load model if not already loaded
            model_data = self._load_local_model(model_config)
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Format messages for the model
            formatted_prompt = self._format_chat_prompt(messages, tokenizer)
            
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            
            # Move to model's device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_content = response[len(formatted_prompt):].strip()
            
            logger.info(f"✅ Local inference complete: {model_config.model_id}")
            
            return {
                "content": response_content,
                "model": model_config.model_id,
                "model_key": model_config.name,
                "usage": {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
                    "total_tokens": outputs.shape[1]
                },
                "finish_reason": "length" if outputs.shape[1] >= max_tokens else "stop"
            }
            
        except Exception as e:
            logger.error(f"❌ Local inference error for {model_config.model_id}: {str(e)}")
            raise RuntimeError(f"Local inference failed: {str(e)}")
    
    def _format_chat_prompt(
        self, 
        messages: List[Dict[str, str]], 
        tokenizer
    ) -> str:
        """Format chat messages into model-specific prompt format"""
        
        # Try to use tokenizer's chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {str(e)}")
        
        # Fallback to generic format
        formatted_prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_prompt += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                formatted_prompt += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}</s>\n"
        
        # Add assistant prefix for the response
        formatted_prompt += "<|assistant|>\n"
        
        return formatted_prompt
    
    async def text_completion(
        self,
        model_key: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Text completion interface
        Converts to chat format internally for compatibility
        """
        # Convert to chat format
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(model_key, messages, **kwargs)
    
    async def embedding(
        self,
        model_key: str,
        texts: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Generate embeddings for text(s)
        
        Args:
            model_key: Key from model registry (must be embedding model)
            texts: Single text or list of texts
        
        Returns:
            Dict with 'embeddings' (list of vectors) and 'model'
        """
        model_config = model_registry.get_model(model_key)
        if not model_config:
            raise ValueError(f"Model '{model_key}' not found in registry")
        
        if model_config.model_type != ModelType.EMBEDDING:
            raise ValueError(f"Model '{model_key}' is not an embedding model")
        
        if not self.hf_client:
            raise RuntimeError("Hugging Face client not initialized for embeddings")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Use HF Inference API for embeddings
            embeddings = []
            for text in texts:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.hf_client.feature_extraction(
                        text,
                        model=model_config.model_id
                    )
                )
                embeddings.append(result)
            
            return {
                "embeddings": embeddings,
                "model": model_config.model_id,
                "dimension": len(embeddings[0]) if embeddings else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Embedding error for {model_config.model_id}: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their capabilities"""
        models = []
        
        for key in model_registry._registry.keys():
            config = model_registry.get_model(key)
            if config:
                models.append({
                    "key": key,
                    "name": config.name,
                    "model_id": config.model_id,
                    "type": config.model_type.value,
                    "provider": config.provider.value,
                    "description": config.description,
                    "context_length": config.context_length,
                    "max_tokens": config.max_tokens,
                    "supported_tasks": config.supported_tasks,
                    "is_local": config.is_local,
                    "quantization": config.quantization,
                    "cost_per_1k_tokens": config.cost_per_1k_tokens
                })
        
        return models
    
    def unload_local_model(self, model_key: str) -> bool:
        """Unload a local model from memory"""
        model_config = model_registry.get_model(model_key)
        if not model_config:
            return False
        
        if model_config.model_id in self.local_models:
            del self.local_models[model_config.model_id]
            
            # Clear CUDA cache if available
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✅ Unloaded model: {model_config.model_id}")
            return True
        
        return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded local models"""
        return list(self.local_models.keys())

# Global inference client
inference_client = UnifiedInferenceClient()
