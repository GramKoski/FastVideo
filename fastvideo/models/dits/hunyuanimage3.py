# SPDX-License-Identifier: Apache-2.0

"""Hunyuan Image 3.0 model implementation for FastVideo."""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models.dits import HunyuanImage3Config
from fastvideo.layers.activation import get_act_fn
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed, _apply_rotary_emb
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.base import CachableDiT
from fastvideo.platforms import AttentionBackendEnum


# =======================================================
#     Helper Functions
# =======================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value states for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# =======================================================
#     Normalization
# =======================================================

class HunyuanRMSNorm(nn.Module):
    """RMSNorm used in Hunyuan models."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        output = x_float * torch.rsqrt(variance + self.eps)
        output = output.type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


# =======================================================
#     MoE Components
# =======================================================

class HunyuanMLP(nn.Module):
    """MLP with SwiGLU or GELU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_act = hidden_act
        self.act_fn = get_act_fn(hidden_act if hidden_act != "silu" else "swiglu")

        if hidden_act == "silu":
            # SwiGLU needs 2x intermediate size
            self.gate_and_up_proj = ReplicatedLinear(
                hidden_size,
                intermediate_size * 2,
                bias=bias,
                params_dtype=dtype,
                prefix=f"{prefix}.gate_and_up_proj",
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=bias,
                params_dtype=dtype,
                prefix=f"{prefix}.down_proj",
            )
        elif hidden_act == "gelu":
            self.gate_and_up_proj = ReplicatedLinear(
                hidden_size,
                intermediate_size,
                bias=bias,
                params_dtype=dtype,
                prefix=f"{prefix}.gate_and_up_proj",
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=bias,
                params_dtype=dtype,
                prefix=f"{prefix}.down_proj",
            )
        else:
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "silu":
            gate_up, _ = self.gate_and_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = gate * self.act_fn(up)
        else:  # gelu
            intermediate, _ = self.gate_and_up_proj(x)
            intermediate = self.act_fn(intermediate)
        
        output, _ = self.down_proj(intermediate)
        return output


class HunyuanTopKGate(nn.Module):
    """Top-K gating for MoE."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        topk: int = 8,
        dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.topk = topk
        self.num_experts = num_experts
        
        # Gate always uses float32 for stability
        self.wg = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.wg",
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            topk_weights: [batch_size, seq_len, topk]
            topk_indices: [batch_size, seq_len, topk]
        """
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Compute gating logits
        if self.wg.weight.dtype == torch.float32:
            hidden_states_flat = hidden_states_flat.float()
        
        logits, _ = self.wg(hidden_states_flat)  # [bsz * seq_len, num_experts]
        gates = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(gates, self.topk, dim=-1)
        
        # Normalize topk weights
        weight_sums = topk_weights.sum(dim=-1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        topk_weights = topk_weights / weight_sums
        
        # Reshape back
        topk_weights = topk_weights.view(bsz, seq_len, self.topk)
        topk_indices = topk_indices.view(bsz, seq_len, self.topk)
        
        return topk_weights, topk_indices


class HunyuanMoE(nn.Module):
    """Mixture of Experts layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        topk: int = 8,
        use_shared_expert: bool = False,
        num_shared_expert: int = 1,
        hidden_act: str = "silu",
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self.use_shared_expert = use_shared_expert
        
        # Gating network
        self.gate = HunyuanTopKGate(
            hidden_size,
            num_experts,
            topk=topk,
            dtype=dtype,
            prefix=f"{prefix}.gate",
        )
        
        # Expert MLPs
        self.experts = nn.ModuleList([
            HunyuanMLP(
                hidden_size,
                intermediate_size,
                hidden_act=hidden_act,
                bias=bias,
                dtype=dtype,
                prefix=f"{prefix}.experts.{i}",
            )
            for i in range(num_experts)
        ])
        
        # Optional shared expert
        if use_shared_expert:
            self.shared_mlp = HunyuanMLP(
                hidden_size,
                intermediate_size * num_shared_expert,
                hidden_act=hidden_act,
                bias=bias,
                dtype=dtype,
                prefix=f"{prefix}.shared_mlp",
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Get routing weights and indices
        topk_weights, topk_indices = self.gate(hidden_states)
        topk_weights_flat = topk_weights.view(-1, self.topk)
        topk_indices_flat = topk_indices.view(-1, self.topk)
        
        # Compute expert outputs
        final_output = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices_flat == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # Get tokens for this expert
            expert_tokens = hidden_states_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Get weights for this expert
            expert_weights_mask = (topk_indices_flat == expert_idx)
            expert_weights = topk_weights_flat.masked_fill(
                ~expert_weights_mask, 0.0
            ).sum(dim=-1, keepdim=True)[expert_mask]
            
            # Accumulate weighted output
            final_output[expert_mask] += expert_output * expert_weights
        
        final_output = final_output.view(bsz, seq_len, hidden_size)
        
        # Add shared expert if enabled
        if self.use_shared_expert:
            shared_output = self.shared_mlp(hidden_states)
            final_output = final_output + shared_output
        
        return final_output


# =======================================================
#     Attention
# =======================================================

class HunyuanImage3Attention(nn.Module):
    """Self-attention with GQA, QK normalization, and RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_qk_norm: bool = True,
        use_rotary_pos_emb: bool = True,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        supported_attention_backends: Optional[Tuple[AttentionBackendEnum, ...]] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        
        self.hidden_size_q = head_dim * num_attention_heads
        self.hidden_size_kv = head_dim * num_key_value_heads
        
        # Unified QKV projection
        self.qkv_proj = ReplicatedLinear(
            hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=bias,
            params_dtype=dtype,
            prefix=f"{prefix}.qkv_proj",
        )
        
        # Output projection
        self.o_proj = ReplicatedLinear(
            self.hidden_size_q,
            hidden_size,
            bias=bias,
            params_dtype=dtype,
            prefix=f"{prefix}.o_proj",
        )
        
        # QK normalization
        if use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(head_dim, dtype=dtype)
            self.key_layernorm = HunyuanRMSNorm(head_dim, dtype=dtype)
        
        # Distributed attention
        self.attn = DistributedAttention(
            num_heads=num_attention_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        
        # Unified QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(
            bsz, seq_len, self.num_key_value_heads,
            self.num_key_value_groups + 2, self.head_dim
        )
        
        # Split Q, K, V
        query_states, key_states, value_states = torch.split(
            qkv, [self.num_key_value_groups, 1, 1], dim=3
        )
        
        # Reshape
        query_states = query_states.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.reshape(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.reshape(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply RoPE
        if self.use_rotary_pos_emb and rotary_emb is not None:
            cos, sin = rotary_emb
            query_states, key_states = _apply_rotary_emb(
                query_states, key_states, cos, sin
            )
        
        # QK normalization
        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)
        
        # Expand K/V for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention
        query_states = query_states.transpose(1, 2)  # [bsz, seq_len, num_heads, head_dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        attn_output = self.attn(query_states, key_states, value_states)
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


# =======================================================
#     Transformer Block
# =======================================================

class HunyuanImage3DecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: Optional[int] = None,
        moe_topk: int = 8,
        use_shared_expert: bool = False,
        num_shared_expert: int = 1,
        hidden_act: str = "silu",
        use_qk_norm: bool = True,
        use_rotary_pos_emb: bool = True,
        norm_eps: float = 1e-6,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        supported_attention_backends: Optional[Tuple[AttentionBackendEnum, ...]] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention
        self.self_attn = HunyuanImage3Attention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            use_qk_norm=use_qk_norm,
            use_rotary_pos_emb=use_rotary_pos_emb,
            bias=attention_bias,
            dtype=dtype,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.self_attn",
        )
        
        # MLP or MoE
        if num_experts is not None and num_experts > 1:
            self.mlp = HunyuanMoE(
                hidden_size,
                intermediate_size,
                num_experts,
                topk=moe_topk,
                use_shared_expert=use_shared_expert,
                num_shared_expert=num_shared_expert,
                hidden_act=hidden_act,
                bias=mlp_bias,
                dtype=dtype,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = HunyuanMLP(
                hidden_size,
                intermediate_size,
                hidden_act=hidden_act,
                bias=mlp_bias,
                dtype=dtype,
                prefix=f"{prefix}.mlp",
            )
        
        # Layer norms
        self.input_layernorm = HunyuanRMSNorm(hidden_size, eps=norm_eps, dtype=dtype)
        self.post_attention_layernorm = HunyuanRMSNorm(hidden_size, eps=norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Self-attention with prenorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_emb=rotary_emb)
        hidden_states = residual + hidden_states
        
        # MLP with prenorm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# =======================================================
#     Image Projection Layers
# =======================================================

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            ReplicatedLinear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
                params_dtype=dtype,
            ),
            nn.SiLU(),
            ReplicatedLinear(
                hidden_size,
                hidden_size,
                bias=True,
                params_dtype=dtype,
            ),
        )
        
        # Initialize weights
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size] or [batch_size, 1]
        
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        # Sinusoidal embedding
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period))
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        
        if t.dim() > 1:
            t = t.squeeze(-1)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        
        # MLP
        output, _ = self.mlp[0](embedding)
        output = self.mlp[1](output)
        output, _ = self.mlp[2](output)
        
        return output


# =======================================================
#     Main Model
# =======================================================

class HunyuanImage3TransformerModel(CachableDiT):
    """Hunyuan Image 3.0 transformer backbone."""

    _fsdp_shard_conditions = HunyuanImage3Config().arch_config._fsdp_shard_conditions
    _compile_conditions = HunyuanImage3Config().arch_config._compile_conditions
    param_names_mapping = HunyuanImage3Config().arch_config.param_names_mapping
    reverse_param_names_mapping = HunyuanImage3Config().arch_config.reverse_param_names_mapping
    lora_param_names_mapping = HunyuanImage3Config().arch_config.lora_param_names_mapping

    def __init__(self, config: HunyuanImage3Config, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        
        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        
        # Word embeddings
        self.wte = nn.Embedding(arch.vocab_size, arch.hidden_size, dtype=arch.dtype)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx in range(arch.num_hidden_layers):
            # Determine if this layer uses MoE
            use_moe = (arch.num_experts > 1) and (layer_idx >= arch.moe_layer_num_skipped)
            
            layer = HunyuanImage3DecoderLayer(
                hidden_size=arch.hidden_size,
                num_attention_heads=arch.num_attention_heads,
                num_key_value_heads=arch.num_key_value_heads,
                head_dim=arch.attention_head_dim,
                intermediate_size=arch.intermediate_size,
                num_experts=arch.num_experts if use_moe else None,
                moe_topk=arch.moe_topk,
                use_shared_expert=arch.use_mixed_mlp_moe and use_moe,
                num_shared_expert=arch.num_shared_expert,
                hidden_act=arch.hidden_act,
                use_qk_norm=arch.use_qk_norm,
                use_rotary_pos_emb=arch.use_rotary_pos_emb,
                norm_eps=arch.rms_norm_eps,
                attention_bias=arch.attention_bias,
                mlp_bias=arch.mlp_bias,
                dtype=arch.dtype,
                supported_attention_backends=arch._supported_attention_backends,
                prefix=f"{config.prefix}.layers.{layer_idx}",
            )
            self.layers.append(layer)
        
        # Final norm
        self.norm = HunyuanRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps, dtype=arch.dtype)
        
        # LM head (for potential text generation, but not used in image generation mode)
        self.lm_head = ReplicatedLinear(
            arch.hidden_size,
            arch.vocab_size,
            bias=False,
            params_dtype=arch.dtype,
            prefix=f"{config.prefix}.lm_head",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - image latents
            encoder_hidden_states: [batch_size, text_len, hidden_size] - text embeddings (optional, not used in generation-only)
            timestep: [batch_size] - timestep for diffusion
            rotary_emb: (cos, sin) tuple for RoPE
        
        Returns:
            diffusion_prediction: [batch_size, seq_len, num_channels_latents]
        """
        # For generation mode, hidden_states are already embeddings from image projection
        # We don't use encoder_hidden_states in generation-only mode
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_emb=rotary_emb)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Return hidden states (will be projected by final layer in pipeline)
        return hidden_states


# Export the model class
EntryClass = HunyuanImage3TransformerModel
