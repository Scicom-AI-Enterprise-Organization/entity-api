"""
Test varlen ragged tensor batching for encoder models.

Tests the core functionality of packing variable length sequences
without padding using cu_seqlens (qo_indptr).
"""
import unittest
import torch
import flashinfer
from flash_infer_encoder_non_causal.main import merge_bpe_tokens_tagging


class TestVarlenBatching(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.num_heads = 16
        self.head_dim = 128
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, kv_layout="NHD"
        )

    def test_cu_seqlens_computation(self):
        """Test cumulative sequence lengths computation."""
        seq_lengths = [3, 5, 2, 10]
        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
        
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length
        
        expected = torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32, device='cuda')
        self.assertTrue(torch.equal(cu_seqlens, expected), 
                       f"cu_seqlens mismatch: {cu_seqlens} vs {expected}")

    def test_varlen_packing(self):
        """Test packing variable length sequences without padding."""
        seq_lengths = [3, 5, 2]
        total_tokens = sum(seq_lengths)
        
        # Create dummy input_ids for each sequence
        input_ids_list = [
            torch.randint(0, 1000, (1, length), device='cuda')
            for length in seq_lengths
        ]
        
        # Pack sequences
        packed = torch.cat([ids.squeeze(0) for ids in input_ids_list], dim=0)
        
        self.assertEqual(packed.shape[0], total_tokens, 
                         "Packed tensor should have total_tokens length")
        
        # Verify no padding
        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length
        
        # Check boundaries
        for i in range(len(seq_lengths)):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            self.assertEqual(end - start, seq_lengths[i],
                           f"Sequence {i} length mismatch")

    def test_ragged_prefill_planning(self):
        """Test FlashInfer ragged prefill wrapper planning."""
        seq_lengths = [10, 20, 15]
        qo_indptr = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
        for i, length in enumerate(seq_lengths):
            qo_indptr[i + 1] = qo_indptr[i] + length
        
        total_tokens = qo_indptr[-1].item()
        max_seqlen = max(seq_lengths)
        
        # Create dummy Q, K, V
        q = torch.randn(total_tokens, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        k = torch.randn(total_tokens, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        v = torch.randn(total_tokens, self.num_heads, self.head_dim, 
                       dtype=torch.float16, device='cuda')
        
        # Plan the batch prefill
        self.prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=qo_indptr,  # Same for self-attention
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            causal=False,  # Non-causal for encoder
            q_data_type=q.dtype,
            sm_scale=1.0 / (self.head_dim ** 0.5),
        )
        
        # Run attention
        output = self.prefill_wrapper.run(q, k, v)
        
        self.assertEqual(output.shape, (total_tokens, self.num_heads, self.head_dim),
                        "Output shape mismatch")


if __name__ == "__main__":
    unittest.main()
