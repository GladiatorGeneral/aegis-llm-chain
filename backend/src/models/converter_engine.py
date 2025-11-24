"""
Multi-Modal Alignment and Fusion Engine
The central "lingua franca" that bridges different modalities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion"""
    fusion_type: str = "cross_attention"  # "cross_attention", "linear_projection", "q_former"
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    num_hidden_layers: int = 6
    projection_dim: int = 512
    dropout_prob: float = 0.1

class CrossAttentionFusionLayer(nn.Module):
    """
    Cross-Attention Fusion Layer (ViLBERT/LXMERT style)
    Allows one modality to attend to another
    """
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query_modality: torch.Tensor, key_value_modality: torch.Tensor) -> torch.Tensor:
        """Cross-attention: query modality attends to key-value modality"""
        batch_size = query_modality.size(0)
        
        # Linear projections
        Q = self.query(query_modality)
        K = self.key(key_value_modality)
        V = self.value(key_value_modality)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_size
        )
        
        return self.output(context)

class QFormerLayer(nn.Module):
    """
    Querying Transformer (BLIP-2 style)
    Lightweight transformer that bridges frozen vision and language models
    """
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Learnable query tokens
        self.num_queries = 32
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_queries, config.hidden_size)
        )
        
        # Cross-attention layers
        self.cross_attention = CrossAttentionFusionLayer(
            config.hidden_size, config.num_attention_heads
        )
        
        # Self-attention and FFN would be here in full implementation
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Bridge visual and text features using learnable queries"""
        batch_size = visual_features.size(0)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Cross-attention: queries attend to visual features
        visual_context = self.cross_attention(query_tokens, visual_features)
        visual_context = self.layer_norm(visual_context + query_tokens)
        
        # Cross-attention: visual-enhanced queries attend to text
        fused_features = self.cross_attention(visual_context, text_features)
        fused_features = self.layer_norm(fused_features + visual_context)
        
        return fused_features

class MultiModalConverterEngine:
    """
    Central Converter Engine - The "Lingua Franca" for multi-modal AI
    Implements multiple fusion strategies from research literature
    """
    
    def __init__(self, config: FusionConfig = None):
        self.config = config or FusionConfig()
        self.fusion_modules = {}
        self._initialize_fusion_components()
        
        logger.info("🚀 Multi-Modal Converter Engine Initialized")
        logger.info(f"🎯 Fusion Strategy: {self.config.fusion_type}")
    
    def _initialize_fusion_components(self):
        """Initialize different fusion strategies"""
        
        # Cross-Attention Fusion (ViLBERT/LXMERT style)
        if self.config.fusion_type == "cross_attention":
            self.fusion_modules['cross_attention'] = CrossAttentionFusionLayer(
                self.config.hidden_size, self.config.num_attention_heads
            )
        
        # Q-Former Fusion (BLIP-2 style)
        elif self.config.fusion_type == "q_former":
            self.fusion_modules['q_former'] = QFormerLayer(self.config)
        
        # Linear Projection (LLaVA style)
        elif self.config.fusion_type == "linear_projection":
            self.fusion_modules['linear_projection'] = nn.Linear(
                self.config.hidden_size, self.config.projection_dim
            )
    
    def align_modalities(
        self, 
        modality_features: Dict[str, torch.Tensor],
        target_modality: str = "text"
    ) -> torch.Tensor:
        """
        Align different modalities into a shared semantic space
        Following CLIP-style contrastive learning principles
        """
        try:
            # Get features for each modality
            text_features = modality_features.get("text")
            visual_features = modality_features.get("vision")
            audio_features = modality_features.get("audio")
            
            aligned_features = []
            
            # Align visual features to text space (CLIP principle)
            if visual_features is not None:
                if self.config.fusion_type == "linear_projection":
                    # LLaVA-style: simple linear projection
                    visual_aligned = self.fusion_modules['linear_projection'](visual_features)
                    aligned_features.append(visual_aligned)
                
                elif self.config.fusion_type == "q_former":
                    # BLIP-2 style: Q-Former bridging
                    visual_aligned = self.fusion_modules['q_former'](
                        visual_features, text_features
                    )
                    aligned_features.append(visual_aligned)
            
            # Align audio features if present
            if audio_features is not None:
                # Similar alignment strategies for audio
                audio_aligned = self.fusion_modules['linear_projection'](audio_features)
                aligned_features.append(audio_aligned)
            
            # Combine all aligned features
            if aligned_features:
                # Mean pooling or other fusion strategies
                fused_representation = torch.mean(torch.stack(aligned_features), dim=0)
                
                # If we have text, concatenate or add with fused features
                if text_features is not None:
                    # Ensure dimensions match
                    if text_features.size(-1) != fused_representation.size(-1):
                        text_features = nn.Linear(
                            text_features.size(-1), fused_representation.size(-1)
                        )(text_features)
                    
                    # Combine text with aligned multi-modal features
                    final_representation = text_features + fused_representation
                else:
                    final_representation = fused_representation
                
                return final_representation
            else:
                return text_features  # Fallback to text only
                
        except Exception as e:
            logger.error(f"❌ Modality alignment failed: {str(e)}")
            raise
    
    def cross_modal_attention(
        self,
        query_modality: str,
        key_value_modality: str,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform cross-modal attention (ViLBERT style)
        Let one modality attend to another
        """
        try:
            if self.config.fusion_type != "cross_attention":
                logger.warning("Cross-attention not configured, using linear projection")
                # Fallback to projection
                projected = self.fusion_modules['linear_projection'](key_value_features)
                return projected
            
            # Perform cross-attention
            attended_features = self.fusion_modules['cross_attention'](
                query_features, key_value_features
            )
            
            logger.debug(f"🔀 Cross-modal attention: {query_modality} -> {key_value_modality}")
            return attended_features
            
        except Exception as e:
            logger.error(f"❌ Cross-modal attention failed: {str(e)}")
            raise
    
    def create_unified_representation(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create unified multi-modal representation following Flamingo/LLaVA approach
        Convert everything to a common token sequence
        """
        try:
            unified_tokens = []
            token_types = []
            
            # Process text tokens
            if "text" in inputs and inputs["text"] is not None:
                text_tokens = self._process_text(inputs["text"])
                unified_tokens.extend(text_tokens)
                token_types.extend(["text"] * len(text_tokens))
            
            # Process visual tokens
            if "vision" in inputs and inputs["vision"] is not None:
                visual_tokens = self._process_vision(inputs["vision"])
                unified_tokens.extend(visual_tokens)
                token_types.extend(["vision"] * len(visual_tokens))
            
            # Process audio tokens
            if "audio" in inputs and inputs["audio"] is not None:
                audio_tokens = self._process_audio(inputs["audio"])
                unified_tokens.extend(audio_tokens)
                token_types.extend(["audio"] * len(audio_tokens))
            
            # Create attention mask that understands modality boundaries
            attention_mask = self._create_cross_modal_attention_mask(token_types)
            
            return {
                "unified_tokens": torch.stack(unified_tokens) if unified_tokens else None,
                "token_types": token_types,
                "attention_mask": attention_mask,
                "fused_representation": self.align_modalities(inputs)
            }
            
        except Exception as e:
            logger.error(f"❌ Unified representation creation failed: {str(e)}")
            raise
    
    def _process_vision(self, visual_features: torch.Tensor) -> List[torch.Tensor]:
        """Convert visual features to tokens (ViT/LLaVA style)"""
        # Flatten spatial dimensions and project to token space
        batch_size, channels, height, width = visual_features.shape
        visual_tokens = visual_features.view(batch_size, channels, -1).transpose(1, 2)
        
        # Project to unified token dimension
        if hasattr(self, 'visual_projection'):
            visual_tokens = self.visual_projection(visual_tokens)
        
        return [visual_tokens[i] for i in range(batch_size)]
    
    def _process_text(self, text_features: torch.Tensor) -> List[torch.Tensor]:
        """Process text features (already tokenized)"""
        return [text_features[i] for i in range(text_features.size(0))]
    
    def _process_audio(self, audio_features: torch.Tensor) -> List[torch.Tensor]:
        """Convert audio features to tokens"""
        # Similar to vision but for temporal data
        batch_size, channels, time_steps = audio_features.shape
        audio_tokens = audio_features.transpose(1, 2)
        
        if hasattr(self, 'audio_projection'):
            audio_tokens = self.audio_projection(audio_tokens)
        
        return [audio_tokens[i] for i in range(batch_size)]
    
    def _create_cross_modal_attention_mask(self, token_types: List[str]) -> torch.Tensor:
        """Create attention mask that controls cross-modal interactions"""
        seq_length = len(token_types)
        attention_mask = torch.ones(seq_length, seq_length)
        
        # Implement modality-specific attention patterns
        for i, src_type in enumerate(token_types):
            for j, tgt_type in enumerate(token_types):
                # Allow full attention within same modality
                if src_type == tgt_type:
                    attention_mask[i, j] = 1
                # Control cross-modal attention
                elif src_type == "text" and tgt_type == "vision":
                    attention_mask[i, j] = 1  # Text can attend to vision
                elif src_type == "vision" and tgt_type == "text":
                    attention_mask[i, j] = 1  # Vision can attend to text
                else:
                    attention_mask[i, j] = 1  # Allow all by default
        
        return attention_mask

# Global converter engine instance
converter_engine = MultiModalConverterEngine()
