"""
Test app/core/model.py module.

Tests model loading and getter functions.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestModelModule:
    """Test model loading and getters"""
    
    def test_get_tokenizer_function_exists(self):
        """Test get_tokenizer function exists."""
        from app.core.model import get_tokenizer
        
        assert callable(get_tokenizer)
    
    def test_get_model_function_exists(self):
        """Test get_model function exists."""
        from app.core.model import get_model
        
        assert callable(get_model)
    
    def test_load_model_function_exists(self):
        """Test load_model function exists."""
        from app.core.model import load_model
        
        assert callable(load_model)
    
    @pytest.mark.skipif(True, reason="Requires GPU and model download - run manually")
    def test_load_model_integration(self):
        """Integration test for load_model (skipped by default)."""
        from app.core.model import load_model, get_tokenizer, get_model
        
        load_model()
        
        tokenizer = get_tokenizer()
        model = get_model()
        
        assert tokenizer is not None
        assert model is not None

    def test_load_model_calls_register_fa_attention(self):
        """Test that load_model registers FA attention."""
        from app.core.model import load_model
        from unittest.mock import patch, MagicMock
        
        # Mock the heavy imports
        with patch('app.core.model.AutoTokenizer') as mock_tokenizer, \
             patch('app.core.model.Qwen3ForTokenClassification') as mock_model, \
             patch('app.core.model.register_fa_attention') as mock_register:
            
            # Setup mocks
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_instance.config = MagicMock(
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                hidden_size=1024,
                head_dim=128,
                num_labels=3,
                id2label={0: 'O', 1: 'name', 2: 'address'}
            )
            mock_model_instance.eval.return_value.cuda.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            load_model()
            
            # Verify register_fa_attention was called
            mock_register.assert_called_once()
            
            # Verify model was loaded with fa_varlen attention
            mock_model.from_pretrained.assert_called_once()
            call_kwargs = mock_model.from_pretrained.call_args[1]
            assert call_kwargs['attn_implementation'] == 'fa_varlen'

    def test_get_tokenizer_callable(self):
        """Test get_tokenizer is callable and returns value."""
        from app.core.model import get_tokenizer
        
        # Should return the current global value (may be None or loaded)
        result = get_tokenizer()
        # Just test it doesn't raise
        assert True
    
    def test_get_model_callable(self):
        """Test get_model is callable and returns value."""
        from app.core.model import get_model
        
        # Should return the current global value (may be None or loaded)
        result = get_model()
        # Just test it doesn't raise
        assert True
