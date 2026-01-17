"""
Test app/core/attention.py module.

Tests FA version detection and custom attention interface registration.
"""
import pytest
import torch
import torch.nn.functional as F


class TestAttentionModule:
    """Test the attention module from app.core.attention"""
    
    def test_fa_version_detected(self):
        """Test FA_VERSION is detected based on GPU capability."""
        from app.core.attention import FA_VERSION
        
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 9:
            assert FA_VERSION == "FA3"
        else:
            assert FA_VERSION == "FA2"
    
    def test_flash_attn_version_string(self):
        """Test flash_attn_version is a valid string."""
        from app.core.attention import flash_attn_version
        
        assert isinstance(flash_attn_version, str)
        assert len(flash_attn_version) > 0
    
    def test_register_fa_attention(self):
        """Test register_fa_attention function registers without error."""
        from app.core.attention import register_fa_attention
        
        # Should not raise any exception
        register_fa_attention()
    
    def test_logging_imported(self):
        """Test logging is imported in attention module."""
        from app.core.attention import logging
        
        assert logging is not None
    
    def test_flash_attn_imports(self):
        """Test flash_attn functions are imported."""
        from app.core.attention import flash_attn_func, flash_attn_varlen_func
        
        assert callable(flash_attn_func)
        assert callable(flash_attn_varlen_func)
    
    def test_attention_interface_import(self):
        """Test AttentionInterface is imported."""
        from app.core.attention import AttentionInterface
        
        assert AttentionInterface is not None
    
    def test_flash_attn_func_single_sequence(self):
        """Test flash_attn_func with single sequence (same as used in attention.py)."""
        from flash_attn import flash_attn_func
        
        batch_size = 1
        num_heads = 8
        seq_len = 16
        head_dim = 64
        
        # flash_attn expects [batch, seq, heads, head_dim]
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                           dtype=torch.bfloat16, device='cuda')
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                         dtype=torch.bfloat16, device='cuda')
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                           dtype=torch.bfloat16, device='cuda')
        
        # Call flash_attn_func (same as custom_attention_forward uses)
        output = flash_attn_func(query, key, value, causal=True)
        
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    
    def test_flash_attn_varlen_func(self):
        """Test flash_attn_varlen_func with varlen batching (same as used in attention.py)."""
        from flash_attn import flash_attn_varlen_func
        
        seq_lengths = [8, 12, 10]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)
        num_heads = 8
        head_dim = 64
        
        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length
        
        # flash_attn_varlen expects [total_tokens, heads, head_dim]
        query = torch.randn(total_tokens, num_heads, head_dim, 
                           dtype=torch.bfloat16, device='cuda')
        key = torch.randn(total_tokens, num_heads, head_dim, 
                         dtype=torch.bfloat16, device='cuda')
        value = torch.randn(total_tokens, num_heads, head_dim, 
                           dtype=torch.bfloat16, device='cuda')
        
        # Call flash_attn_varlen_func (same as custom_attention_forward uses)
        output = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        
        assert output.shape == (total_tokens, num_heads, head_dim)
    
    def test_attention_output_correctness(self):
        """Test that attention output is numerically correct vs SDPA."""
        from flash_attn import flash_attn_func
        
        batch_size = 1
        num_heads = 4
        seq_len = 8
        head_dim = 32
        
        torch.manual_seed(42)
        # flash_attn format: [batch, seq, heads, head_dim]
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                           dtype=torch.float16, device='cuda')
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                         dtype=torch.float16, device='cuda')
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                           dtype=torch.float16, device='cuda')
        
        # Get FA output
        output_fa = flash_attn_func(query, key, value, causal=True)
        # Transpose to SDPA format [batch, heads, seq, head_dim]
        output_fa = output_fa.transpose(1, 2)
        
        # SDPA format: [batch, heads, seq, head_dim]
        q_sdpa = query.transpose(1, 2)
        k_sdpa = key.transpose(1, 2)
        v_sdpa = value.transpose(1, 2)
        output_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
        
        # Check cosine similarity
        cosine_sim = F.cosine_similarity(
            output_fa.flatten().unsqueeze(0).float(),
            output_sdpa.flatten().unsqueeze(0).float()
        )
        
        assert cosine_sim.item() > 0.99, f"Attention mismatch: cosine_sim={cosine_sim.item()}"
