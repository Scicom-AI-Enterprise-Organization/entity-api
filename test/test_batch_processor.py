"""
Test app/core/batch_processor.py module.
"""
import pytest
import torch
import asyncio


class TestBatchProcessorModule:
    """Test batch processor queue and utilities"""
    
    def test_inference_queue_exists(self):
        """Test inference_queue is created."""
        from app.core.batch_processor import inference_queue
        
        assert inference_queue is not None
    
    def test_inference_queue_is_asyncio_queue(self):
        """Test inference_queue is an asyncio Queue."""
        from app.core.batch_processor import inference_queue
        
        assert hasattr(inference_queue, 'put')
        assert hasattr(inference_queue, 'get')
        assert hasattr(inference_queue, 'empty')
    
    def test_process_batch_queue_function_exists(self):
        """Test process_batch_queue function exists."""
        from app.core.batch_processor import process_batch_queue
        
        assert callable(process_batch_queue)
    
    def test_args_imported(self):
        """Test args is imported in batch_processor."""
        from app.core.batch_processor import args
        
        assert args is not None
        assert hasattr(args, 'max_batch_size')
        assert hasattr(args, 'max_seq_len')
        assert hasattr(args, 'microsleep')
    
    def test_logging_imported(self):
        """Test logging is imported."""
        from app.core.batch_processor import logging
        
        assert logging is not None
    
    def test_fa_version_imported(self):
        """Test FA_VERSION is imported."""
        from app.core.batch_processor import FA_VERSION
        
        assert FA_VERSION in ["FA2", "FA3"]
    
    def test_cu_seqlens_computation(self):
        """Test cumulative sequence lengths computation logic."""
        seq_lengths = [3, 5, 2, 10]
        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
        
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length
        
        expected = torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32, device='cuda')
        assert torch.equal(cu_seqlens, expected)
    
    def test_varlen_packing_logic(self):
        """Test packing variable length sequences without padding."""
        seq_lengths = [3, 5, 2]
        total_tokens = sum(seq_lengths)
        
        # Create dummy input_ids for each sequence
        input_ids_list = [
            torch.randint(0, 1000, (1, length), device='cuda')
            for length in seq_lengths
        ]
        
        # Pack sequences (same logic as batch_processor)
        packed = torch.cat([ids.squeeze(0) for ids in input_ids_list], dim=0)
        
        assert packed.shape[0] == total_tokens
    
    def test_sequence_boundary_extraction(self):
        """Test extracting sequence boundaries from cu_seqlens."""
        seq_lengths = [10, 20, 15]
        cu_seqlens = torch.tensor([0, 10, 30, 45], dtype=torch.int32, device='cuda')
        
        for i in range(len(seq_lengths)):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            
            assert end - start == seq_lengths[i]
    
    def test_packed_attention_mask(self):
        """Test creating packed attention mask (all 1s)."""
        total_tokens = 45
        
        # Same logic as batch_processor
        packed_attention_mask = torch.ones(1, total_tokens, dtype=torch.long, device='cuda')
        
        assert packed_attention_mask.shape == (1, total_tokens)
        assert packed_attention_mask.sum().item() == total_tokens
    
    @pytest.mark.asyncio
    async def test_queue_put_get(self):
        """Test putting and getting from inference queue."""
        from app.core.batch_processor import inference_queue
        
        # Clear any existing items
        while not inference_queue.empty():
            try:
                inference_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Put an item
        test_data = {'inputs': 'test text', 'future': asyncio.Future()}
        await inference_queue.put(test_data)
        
        # Get the item
        result = await inference_queue.get()
        
        assert result['inputs'] == 'test text'
    
    @pytest.mark.asyncio
    async def test_queue_empty_check(self):
        """Test queue empty check."""
        from app.core.batch_processor import inference_queue
        
        # Clear queue
        while not inference_queue.empty():
            try:
                inference_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        assert inference_queue.empty()
        
        # Add item
        await inference_queue.put({'test': True, 'future': asyncio.Future()})
        assert not inference_queue.empty()
        
        # Clean up
        await inference_queue.get()
