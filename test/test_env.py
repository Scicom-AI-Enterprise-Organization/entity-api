"""
Test app/env.py module - Configuration and logging.
"""
import pytest


class TestEnvModule:
    """Test app/env.py configuration"""
    
    def test_args_exists(self):
        """Test args namespace is created."""
        from app.env import args
        
        assert args is not None
    
    def test_args_host_default(self):
        """Test default host value."""
        from app.env import args
        
        assert hasattr(args, 'host')
        assert args.host == "0.0.0.0"
    
    def test_args_port_default(self):
        """Test default port value."""
        from app.env import args
        
        assert hasattr(args, 'port')
        assert args.port == 8000
    
    def test_args_model_default(self):
        """Test default model value."""
        from app.env import args
        
        assert hasattr(args, 'model')
        assert "multilingual-dynamic-entity-decoder" in args.model
    
    def test_args_max_batch_size(self):
        """Test max_batch_size is set."""
        from app.env import args
        
        assert hasattr(args, 'max_batch_size')
        assert args.max_batch_size > 0
    
    def test_args_max_seq_len(self):
        """Test max_seq_len is set."""
        from app.env import args
        
        assert hasattr(args, 'max_seq_len')
        assert args.max_seq_len > 0
    
    def test_args_microsleep(self):
        """Test microsleep is set."""
        from app.env import args
        
        assert hasattr(args, 'microsleep')
        assert args.microsleep >= 0
    
    def test_args_torch_dtype(self):
        """Test torch_dtype is set."""
        import torch
        from app.env import args
        
        assert hasattr(args, 'torch_dtype')
        # torch_dtype can be a string or torch dtype
        valid_dtypes = [torch.float16, torch.bfloat16, torch.float32, 'float16', 'bfloat16', 'float32']
        assert args.torch_dtype in valid_dtypes
    
    def test_logging_exists(self):
        """Test logging is configured."""
        from app.env import logging
        
        assert logging is not None
    
    def test_logging_can_log(self):
        """Test logging functions work."""
        from app.env import logging
        
        # Should not raise
        logging.debug("Test debug")
        logging.info("Test info")
        logging.warning("Test warning")
