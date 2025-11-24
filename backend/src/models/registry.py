"""
Model Registry & Configuration System
Organizes models by type, capability, and optimization level
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import os

class ModelType(Enum):
    """Types of models"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class ModelProvider(Enum):
    """Model providers"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

class MultimodalCapability(Enum):
    """Multimodal capabilities"""
    IMAGE_TO_TEXT = "image_to_text"
    TEXT_TO_IMAGE = "text_to_image"
    AUDIO_TO_TEXT = "audio_to_text"
    TEXT_TO_AUDIO = "text_to_audio"
    VIDEO_UNDERSTANDING = "video_understanding"
    CHART_ANALYSIS = "chart_analysis"
    DOCUMENT_UNDERSTANDING = "document_understanding"

@dataclass
class ModelConfig:
    """Configuration for each model"""
    model_id: str
    model_type: ModelType
    provider: ModelProvider
    name: str
    description: str
    context_length: int
    max_tokens: int
    supported_tasks: List[str]
    requires_auth: bool = True
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    cost_per_1k_tokens: Optional[float] = None
    is_local: bool = False
    local_path: Optional[str] = None
    quantization: Optional[str] = None  # "4bit", "8bit", "16bit"
    
    # Converter engine capabilities
    uses_converter_engine: bool = False
    converter_type: str = "linear_projection"  # "cross_attention", "q_former", "linear"
    supported_cross_modal_tasks: List[str] = field(default_factory=list)
    alignment_strategy: str = "clip_style"  # "clip_style", "contrastive", "cross_attention"
    multimodal_capabilities: List[MultimodalCapability] = field(default_factory=list)
    supported_media_types: List[str] = field(default_factory=list)
    max_image_size: Optional[int] = None
    
    def __post_init__(self):
        if self.api_key_env and self.requires_auth:
            self.api_key = os.getenv(self.api_key_env)
        else:
            self.api_key = None

class ModelRegistry:
    """
    Central registry for all models
    Organized by type, provider, and capability
    """
    
    def __init__(self):
        self._registry: Dict[str, ModelConfig] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize with comprehensive model configurations"""
        
        # ==================== CHAT MODELS ====================
        
        # Cogito 67B - Your primary model
        self._registry["cogito-671b"] = ModelConfig(
            model_id="deepcogito/cogito-671b-v2.1",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Cogito 67B v2.1",
            description="Advanced 67B parameter chat model with strong reasoning capabilities",
            context_length=32768,
            max_tokens=4096,
            supported_tasks=["chat", "reasoning", "coding", "analysis"],
            api_key_env="HF_TOKEN",
            cost_per_1k_tokens=0.0,  # Free via Inference API
            is_local=False
        )
        
        # Llama Models
        self._registry["llama2-70b"] = ModelConfig(
            model_id="meta-llama/Llama-2-70b-chat-hf",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Llama 2 70B Chat",
            description="Meta's 70B parameter Llama 2 model optimized for chat",
            context_length=4096,
            max_tokens=4096,
            supported_tasks=["chat", "qa", "summarization"],
            api_key_env="HF_TOKEN",
            requires_auth=True
        )
        
        self._registry["llama3-8b"] = ModelConfig(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Llama 3 8B Instruct",
            description="Latest Llama 3 8B parameter instruct model with improved performance",
            context_length=8192,
            max_tokens=2048,
            supported_tasks=["chat", "instruction", "reasoning"],
            api_key_env="HF_TOKEN"
        )
        
        # Mistral Models
        self._registry["mistral-7b"] = ModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Mistral 7B Instruct v0.2",
            description="Efficient 7B parameter instruct model with 32K context",
            context_length=32768,
            max_tokens=8192,
            supported_tasks=["chat", "instruction", "coding"],
            api_key_env="HF_TOKEN"
        )
        
        self._registry["mixtral-8x7b"] = ModelConfig(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Mixtral 8x7B Instruct",
            description="Mixture of Experts model with 46.7B parameters, 32K context",
            context_length=32768,
            max_tokens=4096,
            supported_tasks=["chat", "reasoning", "coding", "multilingual"],
            api_key_env="HF_TOKEN"
        )
        
        # Microsoft Phi Models
        self._registry["phi-3-mini"] = ModelConfig(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Phi-3 Mini 4K",
            description="Lightweight 3.8B parameter model optimized for efficiency",
            context_length=4096,
            max_tokens=4096,
            supported_tasks=["chat", "instruction", "qa"],
            api_key_env="HF_TOKEN"
        )
        
        self._registry["phi-3-medium"] = ModelConfig(
            model_id="microsoft/Phi-3-medium-4k-instruct",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Phi-3 Medium 4K",
            description="14B parameter model with excellent reasoning capabilities",
            context_length=4096,
            max_tokens=4096,
            supported_tasks=["chat", "reasoning", "coding", "instruction"],
            api_key_env="HF_TOKEN"
        )
        
        # Code-Specialized Models
        self._registry["codellama-34b"] = ModelConfig(
            model_id="codellama/CodeLlama-34b-Instruct-hf",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="CodeLlama 34B Instruct",
            description="Code-specialized 34B model for programming tasks",
            context_length=16384,
            max_tokens=4096,
            supported_tasks=["coding", "code-generation", "debugging", "chat"],
            api_key_env="HF_TOKEN"
        )
        
        # ==================== EMBEDDING MODELS ====================
        
        self._registry["bge-large"] = ModelConfig(
            model_id="BAAI/bge-large-en-v1.5",
            model_type=ModelType.EMBEDDING,
            provider=ModelProvider.HUGGINGFACE,
            name="BGE Large English",
            description="High-quality English embeddings for semantic search",
            context_length=512,
            max_tokens=512,
            supported_tasks=["embedding", "semantic-search", "retrieval"],
            api_key_env="HF_TOKEN"
        )
        
        self._registry["gte-large"] = ModelConfig(
            model_id="thenlper/gte-large",
            model_type=ModelType.EMBEDDING,
            provider=ModelProvider.HUGGINGFACE,
            name="GTE Large",
            description="General text embeddings for semantic search and retrieval",
            context_length=512,
            max_tokens=512,
            supported_tasks=["embedding", "semantic-search", "retrieval"],
            api_key_env="HF_TOKEN"
        )
        
        self._registry["e5-large"] = ModelConfig(
            model_id="intfloat/e5-large-v2",
            model_type=ModelType.EMBEDDING,
            provider=ModelProvider.HUGGINGFACE,
            name="E5 Large v2",
            description="Multilingual embedding model with strong performance",
            context_length=512,
            max_tokens=512,
            supported_tasks=["embedding", "semantic-search", "multilingual"],
            api_key_env="HF_TOKEN"
        )
        
        # ==================== LOCAL MODELS ====================
        
        self._registry["zephyr-7b-local"] = ModelConfig(
            model_id="HuggingFaceH4/zephyr-7b-beta",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Zephyr 7B Local",
            description="7B model optimized for local inference with 4-bit quantization",
            context_length=32768,
            max_tokens=4096,
            supported_tasks=["chat", "instruction"],
            is_local=True,
            local_path="./models/zephyr-7b-beta",
            quantization="4bit",
            api_key_env="HF_TOKEN"
        )
        
        self._registry["mistral-7b-local"] = ModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_type=ModelType.CHAT,
            provider=ModelProvider.HUGGINGFACE,
            name="Mistral 7B Local",
            description="Mistral 7B optimized for local 4-bit inference",
            context_length=32768,
            max_tokens=8192,
            supported_tasks=["chat", "instruction", "coding"],
            is_local=True,
            local_path="./models/mistral-7b-instruct-v0.2",
            quantization="4bit",
            api_key_env="HF_TOKEN"
        )
        
        # ==================== CONVERTER-ENABLED MULTIMODAL MODELS ====================
        
        # LLaVA-style models (Linear Projection Converter)
        self._registry["llava-13b-converter"] = ModelConfig(
            model_id="llava-hf/llava-1.5-13b-hf",
            model_type=ModelType.MULTIMODAL,
            provider=ModelProvider.HUGGINGFACE,
            name="LLaVA 13B with Converter Engine",
            description="Vision-language model with linear projection converter",
            context_length=4096,
            max_tokens=512,
            supported_tasks=["visual_qa", "image_captioning", "cross_modal_reasoning"],
            multimodal_capabilities=[
                MultimodalCapability.IMAGE_TO_TEXT,
                MultimodalCapability.CHART_ANALYSIS,
                MultimodalCapability.DOCUMENT_UNDERSTANDING
            ],
            supported_media_types=["image/jpeg", "image/png", "image/webp"],
            max_image_size=1024,
            uses_converter_engine=True,
            converter_type="linear_projection",
            supported_cross_modal_tasks=[
                "visual_question_answering",
                "image_description", 
                "chart_data_extraction",
                "document_qa"
            ],
            alignment_strategy="clip_style",
            is_local=True,
            local_path="./models/llava-1.5-13b-hf",
            api_key_env="HF_TOKEN"
        )
        
        # BLIP-2 style models (Q-Former Converter)
        self._registry["blip2-converter"] = ModelConfig(
            model_id="Salesforce/blip2-opt-2.7b",
            model_type=ModelType.MULTIMODAL,
            provider=ModelProvider.HUGGINGFACE,
            name="BLIP-2 with Q-Former Converter",
            description="Vision-language model with querying transformer converter",
            context_length=512,
            max_tokens=256,
            supported_tasks=["visual_qa", "image_captioning", "cross_modal_retrieval"],
            multimodal_capabilities=[MultimodalCapability.IMAGE_TO_TEXT],
            supported_media_types=["image/jpeg", "image/png"],
            max_image_size=384,
            uses_converter_engine=True,
            converter_type="q_former",
            supported_cross_modal_tasks=[
                "visual_question_answering",
                "image_text_matching",
                "cross_modal_retrieval"
            ],
            alignment_strategy="contrastive",
            api_key_env="HF_TOKEN"
        )
        
        # Cross-Attention models (ViLBERT/LXMERT style)
        self._registry["vilbert-style-converter"] = ModelConfig(
            model_id="custom/vilbert-style",
            model_type=ModelType.MULTIMODAL,
            provider=ModelProvider.CUSTOM,
            name="ViLBERT-style Cross-Attention Converter",
            description="Custom model with cross-attention fusion layers",
            context_length=1024,
            max_tokens=512,
            supported_tasks=["visual_qa", "referential_expression", "cross_modal_reasoning"],
            multimodal_capabilities=[
                MultimodalCapability.IMAGE_TO_TEXT,
                MultimodalCapability.TEXT_TO_IMAGE
            ],
            supported_media_types=["image/jpeg", "image/png"],
            uses_converter_engine=True,
            converter_type="cross_attention",
            supported_cross_modal_tasks=[
                "fine-grained_visual_qa",
                "referential_expression_grounding",
                "complex_cross_modal_reasoning"
            ],
            alignment_strategy="cross_attention",
            requires_auth=False
        )
        
        # CLIP-style alignment models
        self._registry["clip-alignment-engine"] = ModelConfig(
            model_id="openai/clip-vit-large-patch14",
            model_type=ModelType.MULTIMODAL,
            provider=ModelProvider.HUGGINGFACE,
            name="CLIP Alignment Engine",
            description="Contrastive learning-based modality alignment",
            context_length=77,
            max_tokens=77,
            supported_tasks=["cross_modal_retrieval", "zero_shot_classification"],
            multimodal_capabilities=[
                MultimodalCapability.IMAGE_TO_TEXT,
                MultimodalCapability.TEXT_TO_IMAGE
            ],
            supported_media_types=["image/jpeg", "image/png"],
            uses_converter_engine=True,
            converter_type="contrastive",
            supported_cross_modal_tasks=[
                "image_text_retrieval",
                "zero_shot_image_classification",
                "cross_modal_similarity"
            ],
            alignment_strategy="contrastive",
            api_key_env="HF_TOKEN"
        )
    
    def get_model(self, model_key: str) -> Optional[ModelConfig]:
        """Get model configuration by key"""
        return self._registry.get(model_key)
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelConfig]:
        """List all models, optionally filtered by type"""
        if model_type:
            return [config for config in self._registry.values() if config.model_type == model_type]
        return list(self._registry.values())
    
    def get_models_by_task(self, task: str) -> List[ModelConfig]:
        """Get models that support a specific task"""
        return [config for config in self._registry.values() if task in config.supported_tasks]
    
    def get_models_by_provider(self, provider: ModelProvider) -> List[ModelConfig]:
        """Get models from a specific provider"""
        return [config for config in self._registry.values() if config.provider == provider]
    
    def add_model(self, model_key: str, config: ModelConfig):
        """Add a new model to registry"""
        self._registry[model_key] = config
    
    def get_model_count(self) -> int:
        """Get total number of registered models"""
        return len(self._registry)
    
    def search_models(self, query: str) -> List[ModelConfig]:
        """Search models by name, description, or tasks"""
        query_lower = query.lower()
        results = []
        
        for config in self._registry.values():
            if (query_lower in config.name.lower() or 
                query_lower in config.description.lower() or
                any(query_lower in task.lower() for task in config.supported_tasks)):
                results.append(config)
        
        return results
    
    def get_converter_enabled_models(self) -> List[ModelConfig]:
        """Get all models that use the converter engine"""
        return [config for config in self._registry.values() if config.uses_converter_engine]
    
    def get_models_by_converter_type(self, converter_type: str) -> List[ModelConfig]:
        """Get models using a specific converter type"""
        return [
            config for config in self._registry.values() 
            if config.uses_converter_engine and config.converter_type == converter_type
        ]
    
    def get_models_by_multimodal_capability(self, capability: MultimodalCapability) -> List[ModelConfig]:
        """Get models with specific multimodal capability"""
        return [
            config for config in self._registry.values()
            if capability in config.multimodal_capabilities
        ]

# Global registry instance
model_registry = ModelRegistry()
