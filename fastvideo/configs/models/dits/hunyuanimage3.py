# SPDX-License-Identifier: Apache-2.0

"""Hunyuan Image 3.0 model configuration."""

from dataclasses import dataclass, field
from typing import Optional

import torch

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class HunyuanImage3ArchConfig(DiTArchConfig):
    """Architecture configuration for Hunyuan Image 3.0.
    
    IMPORTANT: These are default/template values. The actual pretrained model may
    use different values specified in the checkpoint's config.json. When loading
    official weights, the config will be updated to match the checkpoint.
    
    Reference: https://github.com/Tencent-Hunyuan/HunyuanImage-3.0
    """
    
    # Model dimensions
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = None  # Defaults to num_attention_heads if None
    attention_head_dim: int = 128
    intermediate_size: int = 11008
    
    # MoE configuration - THESE ARE DEFAULTS AND MAY BE OVERRIDDEN BY CHECKPOINT CONFIG
    # Set to conservative defaults; actual pretrained model may use different values
    num_experts: int = 64
    moe_topk: int = 8
    use_mixed_mlp_moe: bool = True
    num_shared_expert: int = 1
    moe_layer_num_skipped: int = 0  # Number of initial layers without MoE
    
    # Normalization and activation
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    
    # Attention configuration
    attention_bias: bool = False
    use_qk_norm: bool = True
    use_rotary_pos_emb: bool = True
    
    # MLP configuration
    mlp_bias: bool = False
    
    # Embeddings
    vocab_size: int = 133120
    max_position_embeddings: int = 32768
    
    # VAE configuration
    latent_channels: int = 32  # VAE latent channels
    num_channels_latents: int = 32  # Same as latent_channels
    
    # Image projection
    patch_size: int = 2  # Patch size for image projection
    patch_embed_hidden_dim: int = 512
    
    # Input/output channels
    in_channels: int = 32
    out_channels: int = 32
    
    # Dtype
    torch_dtype: str = "bfloat16"
    dtype: Optional[torch.dtype] = None
    
    # Parameter mapping from HuggingFace to FastVideo
    param_names_mapping: dict = field(default_factory=lambda: {
        # Word embedding
        r"^model\.wte\.(.*)$": r"wte.\1",
        
        # Layer norms
        r"^model\.layers\.(\d+)\.input_layernorm\.(.*)$": r"layers.\1.input_layernorm.\2",
        r"^model\.layers\.(\d+)\.post_attention_layernorm\.(.*)$": r"layers.\1.post_attention_layernorm.\2",
        r"^model\.norm\.(.*)$": r"norm.\1",
        
        # Attention layers - unified QKV projection
        r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.(.*)$": r"layers.\1.self_attn.qkv_proj.\2",
        r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(.*)$": r"layers.\1.self_attn.o_proj.\2",
        r"^model\.layers\.(\d+)\.self_attn\.query_layernorm\.(.*)$": r"layers.\1.self_attn.query_layernorm.\2",
        r"^model\.layers\.(\d+)\.self_attn\.key_layernorm\.(.*)$": r"layers.\1.self_attn.key_layernorm.\2",
        
        # MoE gating
        r"^model\.layers\.(\d+)\.mlp\.gate\.wg\.(.*)$": r"layers.\1.mlp.gate.wg.\2",
        
        # MoE experts (64 experts per MoE layer)
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_and_up_proj\.(.*)$": 
            r"layers.\1.mlp.experts.\2.gate_and_up_proj.\3",
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(.*)$": 
            r"layers.\1.mlp.experts.\2.down_proj.\3",
        
        # Shared expert MLP
        r"^model\.layers\.(\d+)\.mlp\.shared_mlp\.gate_and_up_proj\.(.*)$": 
            r"layers.\1.mlp.shared_mlp.gate_and_up_proj.\2",
        r"^model\.layers\.(\d+)\.mlp\.shared_mlp\.down_proj\.(.*)$": 
            r"layers.\1.mlp.shared_mlp.down_proj.\2",
        
        # Regular MLP (for non-MoE layers)
        r"^model\.layers\.(\d+)\.mlp\.gate_and_up_proj\.(.*)$": 
            r"layers.\1.mlp.gate_and_up_proj.\2",
        r"^model\.layers\.(\d+)\.mlp\.down_proj\.(.*)$": 
            r"layers.\1.mlp.down_proj.\2",
        
        # LM head
        r"^lm_head\.(.*)$": r"lm_head.\1",
    })
    
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    exclude_lora_layers: list[str] = field(default_factory=lambda: [])
    
    def __post_init__(self):
        super().__post_init__()
        if self.dtype is None:
            self.dtype = torch.bfloat16
        self.num_channels_latents = self.latent_channels


@dataclass
class HunyuanImage3Config(DiTConfig):
    """Full configuration for Hunyuan Image 3.0 model."""
    
    arch_config: HunyuanImage3ArchConfig = field(default_factory=HunyuanImage3ArchConfig)
    prefix: str = "Hunyuan"
