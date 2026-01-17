"""
Flash Attention Interface Registration.

Supports FA3 (Hopper) with fallback to FA2.
Uses varlen ragged tensor style for dynamic batching without padding.

Reference:
- https://huggingface.co/Scicom-intl/multilingual-dynamic-entity-decoder
- https://github.com/malaysia-ai/simple-continuous-batching-flashinfer
"""

import torch
from typing import Optional
from transformers import AttentionInterface
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn import __version__ as flash_attn_version

from app.env import logging


# ============================================================================
# Detect FA version based on GPU capability
# ============================================================================
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 9:  # Hopper (sm90) or newer
        FA_VERSION = "FA3"
        logging.info(f"Flash Attention 3 enabled (Hopper GPU detected, sm{capability[0]}{capability[1]})")
    else:
        FA_VERSION = "FA2"
        logging.info(f"Flash Attention 2 enabled (GPU sm{capability[0]}{capability[1]})")
else:
    raise RuntimeError("CUDA not available - Flash Attention requires GPU")

logging.info(f"flash-attn version: {flash_attn_version}")


# ============================================================================
# Custom Attention Interface
# ============================================================================
def register_fa_attention():
    """
    Register custom Flash Attention interface.
    
    Pattern follows: https://huggingface.co/Scicom-intl/multilingual-dynamic-entity-decoder
    
    cu_seqlens are passed via **kwargs from model forward call for varlen batching.
    """
    
    def custom_attention_forward(
        module: AttentionInterface,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Custom Flash Attention forward.
        
        Input shapes (from transformers):
            query: [batch, n_heads, seq_len, head_dim]
            key: [batch, n_kv_heads, seq_len, head_dim]
            value: [batch, n_kv_heads, seq_len, head_dim]
        
        Flash Attention expects:
            For flash_attn_func: [batch, seq_len, n_heads, head_dim]
            For flash_attn_varlen_func: [total_seq, n_heads, head_dim]
        """
        # Get cu_seqlens from kwargs (passed from model forward)
        cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
        cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
        max_seqlen_q = kwargs.get("max_seqlen_q", None)
        max_seqlen_k = kwargs.get("max_seqlen_k", None)
        
        # Permute to (batch, seq_len, n_heads, head_dim)
        query_permute = query.permute(0, 2, 1, 3)
        key_permute = key.permute(0, 2, 1, 3)
        value_permute = value.permute(0, 2, 1, 3)
        
        if cu_seqlens_q is not None and cu_seqlens_k is not None:
            # ============ VARLEN MODE (dynamic batching) ============
            # Squeeze batch dim: (total_seq, n_heads, head_dim)
            attn_output = flash_attn_varlen_func(
                q=query_permute.squeeze(0),
                k=key_permute.squeeze(0),
                v=value_permute.squeeze(0),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
            )
            # Add batch dim back: (1, total_seq, n_heads, head_dim)
            attn_output = attn_output.unsqueeze(0)
        else:
            # ============ SINGLE SEQUENCE MODE ============
            attn_output = flash_attn_func(
                query_permute, key_permute, value_permute,
                causal=True,
            )
        
        return attn_output, None

    AttentionInterface.register("fa_varlen", custom_attention_forward)
    logging.info(f"Registered 'fa_varlen' attention interface with Flash Attention ({FA_VERSION})")
