"""Models package initialization."""

from . import registry, inference_client, hf_manager

try:
    from . import converter_engine, multimodal_engine
    __all__ = ['registry', 'inference_client', 'hf_manager', 'converter_engine', 'multimodal_engine']
except ImportError:
    __all__ = ['registry', 'inference_client', 'hf_manager']
