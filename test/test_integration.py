"""
Integration tests that load the real model.

These tests are skipped by default. Run with:
    pytest test/test_integration.py -v --run-integration

To include in coverage:
    pytest test/ --cov=app --cov-report=term-missing --run-integration
"""
import pytest
import asyncio


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model loading and inference"""
    
    def test_model_loads_successfully(self, model_fixtures):
        """Test model loads without error."""
        assert model_fixtures['tokenizer'] is not None
        assert model_fixtures['model'] is not None
    
    def test_tokenizer_encodes_text(self, model_fixtures):
        """Test tokenizer can encode text."""
        tokenizer = model_fixtures['tokenizer']
        
        text = "nama saya Ahmad"
        encoded = tokenizer(text, return_tensors='pt')
        
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].shape[0] == 1
    
    def test_model_forward_pass(self, model_fixtures):
        """Test model can do forward pass."""
        import torch
        
        tokenizer = model_fixtures['tokenizer']
        model = model_fixtures['model']
        
        text = "nama saya Ahmad"
        encoded = tokenizer(text, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = model(**encoded)
        
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1
    
    def test_model_predictions(self, model_fixtures):
        """Test model returns valid predictions."""
        import torch
        
        tokenizer = model_fixtures['tokenizer']
        model = model_fixtures['model']
        
        text = "nama saya Ahmad dari Kuala Lumpur"
        encoded = tokenizer(text, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = model(**encoded)
            predictions = outputs.logits.argmax(dim=-1)[0].tolist()
        
        # Should have predictions for each token
        assert len(predictions) == encoded['input_ids'].shape[1]
        
        # All predictions should be valid label IDs
        for pred in predictions:
            assert pred in model.config.id2label


@pytest.mark.integration
class TestBatchProcessorIntegration:
    """Integration tests for batch processor"""
    
    @pytest.mark.asyncio
    async def test_batch_processor_single_request(self, model_fixtures):
        """Test batch processor with single request."""
        from app.core.batch_processor import inference_queue, process_batch_queue
        
        tokenizer = model_fixtures['tokenizer']
        model = model_fixtures['model']
        
        # Clear queue
        while not inference_queue.empty():
            try:
                inference_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Create a future for the result
        future = asyncio.Future()
        
        # Put a request
        await inference_queue.put({
            'inputs': 'nama saya Ahmad',
            'text_raw': 'nama saya Ahmad',
            'future': future,
            'request_id': 'test-1',
            'is_split_into_words': False,
        })
        
        # Start processor in background
        processor_task = asyncio.create_task(process_batch_queue(tokenizer, model))
        
        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=10.0)
            
            assert result is not None
            assert 'predictions' in result
            assert 'labels' in result
            assert 'tokens' in result
        finally:
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass


@pytest.mark.integration  
class TestAPIIntegration:
    """Integration tests for API endpoints with real model"""
    
    def test_predict_endpoint_structure(self, model_fixtures):
        """Test predict endpoint structure with real model."""
        # This test verifies the model is loaded correctly
        assert model_fixtures['tokenizer'] is not None
        assert model_fixtures['model'] is not None
        
        # The actual predict endpoint requires the background processor
        # which is tested in TestBatchProcessorIntegration


@pytest.mark.integration
class TestBatchProcessorMultiple:
    """Additional batch processor tests"""
    
    @pytest.mark.asyncio
    async def test_batch_processor_multiple_requests(self, model_fixtures):
        """Test batch processor with multiple concurrent requests."""
        from app.core.batch_processor import inference_queue, process_batch_queue
        
        tokenizer = model_fixtures['tokenizer']
        model = model_fixtures['model']
        
        # Clear queue
        while not inference_queue.empty():
            try:
                inference_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Create multiple futures
        futures = []
        texts = ["nama saya Ahmad", "saya dari KL", "email test@gmail.com"]
        
        for i, text in enumerate(texts):
            future = asyncio.Future()
            await inference_queue.put({
                'inputs': text,
                'text_raw': text,
                'future': future,
                'request_id': f'test-{i}',
                'is_split_into_words': False,
            })
            futures.append(future)
        
        # Start processor in background
        processor_task = asyncio.create_task(process_batch_queue(tokenizer, model))
        
        try:
            # Wait for all results
            results = await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=15.0
            )
            
            assert len(results) == 3
            for result in results:
                assert 'predictions' in result
                assert 'labels' in result
        finally:
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_batch_processor_with_split_words(self, model_fixtures):
        """Test batch processor with is_split_into_words=True."""
        from app.core.batch_processor import inference_queue, process_batch_queue
        
        tokenizer = model_fixtures['tokenizer']
        model = model_fixtures['model']
        
        # Clear queue
        while not inference_queue.empty():
            try:
                inference_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        future = asyncio.Future()
        text = "nama saya Ahmad"
        
        await inference_queue.put({
            'inputs': text,
            'text_raw': text,
            'future': future,
            'request_id': 'test-split',
            'is_split_into_words': True,
        })
        
        processor_task = asyncio.create_task(process_batch_queue(tokenizer, model))
        
        try:
            result = await asyncio.wait_for(future, timeout=10.0)
            
            assert result is not None
            assert 'predictions' in result
            assert 'word_ids' in result
        finally:
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
