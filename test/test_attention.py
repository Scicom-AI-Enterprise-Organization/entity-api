"""
Test non-causal attention correctness.

Compares FlashInfer non-causal attention output with SDPA
to verify correctness for encoder models.
"""
import unittest
import torch
import torch.nn.functional as F
import flashinfer


class TestNonCausalAttention(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.num_heads = 16
        self.head_dim = 128
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, kv_layout="NHD"
        )

    def test_non_causal_attention_single_sequence(self):
        """Test non-causal attention for single sequence."""
        seq_len = 32
        q = torch.randn(seq_len, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        k = torch.randn(seq_len, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        v = torch.randn(seq_len, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        
        # Create qo_indptr for single sequence
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        
        # Plan and run FlashInfer
        self.prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=qo_indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            causal=False,  # Non-causal
            q_data_type=q.dtype,
            sm_scale=1.0 / (self.head_dim ** 0.5),
        )
        output_flashinfer = self.prefill_wrapper.run(q, k, v)
        
        # Compare with SDPA (non-causal)
        q_sdpa = q.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        k_sdpa = k.transpose(0, 1).unsqueeze(0)
        v_sdpa = v.transpose(0, 1).unsqueeze(0)
        
        output_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            is_causal=False,
            scale=1.0 / (self.head_dim ** 0.5),
        )
        output_sdpa = output_sdpa.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        
        # Compare outputs (using cosine similarity for numerical stability)
        output_flashinfer_flat = output_flashinfer.flatten()
        output_sdpa_flat = output_sdpa.flatten()
        
        cosine_sim = F.cosine_similarity(
            output_flashinfer_flat.unsqueeze(0),
            output_sdpa_flat.unsqueeze(0)
        )
        
        self.assertGreater(cosine_sim.item(), 0.99, 
                          f"Attention output mismatch: cosine_sim={cosine_sim.item()}")

    def test_non_causal_attention_varlen_batch(self):
        """Test non-causal attention for variable length batch."""
        seq_lengths = [10, 20, 15]
        qo_indptr = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
        for i, length in enumerate(seq_lengths):
            qo_indptr[i + 1] = qo_indptr[i] + length
        
        total_tokens = qo_indptr[-1].item()
        
        q = torch.randn(total_tokens, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        k = torch.randn(total_tokens, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        v = torch.randn(total_tokens, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        
        # Plan and run FlashInfer
        self.prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=qo_indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            causal=False,
            q_data_type=q.dtype,
            sm_scale=1.0 / (self.head_dim ** 0.5),
        )
        output_flashinfer = self.prefill_wrapper.run(q, k, v)
        
        # Compare each sequence separately with SDPA
        for i in range(len(seq_lengths)):
            start = qo_indptr[i].item()
            end = qo_indptr[i + 1].item()
            
            q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
            k_seq = k[start:end].transpose(0, 1).unsqueeze(0)
            v_seq = v[start:end].transpose(0, 1).unsqueeze(0)
            
            output_sdpa = F.scaled_dot_product_attention(
                q_seq, k_seq, v_seq,
                is_causal=False,
                scale=1.0 / (self.head_dim ** 0.5),
            )
            output_sdpa = output_sdpa.squeeze(0).transpose(0, 1)
            
            output_flashinfer_seq = output_flashinfer[start:end]
            
            cosine_sim = F.cosine_similarity(
                output_flashinfer_seq.flatten().unsqueeze(0),
                output_sdpa.flatten().unsqueeze(0)
            )
            
            self.assertGreater(cosine_sim.item(), 0.99,
                             f"Sequence {i} attention mismatch: cosine_sim={cosine_sim.item()}")


if __name__ == "__main__":
    unittest.main()
