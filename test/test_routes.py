"""
Test app/api/routes.py module.
"""
import pytest


class TestRoutesModule:
    """Test API routes structure"""
    
    def test_router_exists(self):
        """Test router is created."""
        from app.api.routes import router
        
        assert router is not None
    
    def test_label_to_type_imported(self):
        """Test LABEL_TO_TYPE is available."""
        from app.api.routes import LABEL_TO_TYPE
        
        assert LABEL_TO_TYPE is not None
        assert 'LABEL_1' in LABEL_TO_TYPE
        assert 'LABEL_2' in LABEL_TO_TYPE
    
    def test_label_to_readable_imported(self):
        """Test LABEL_TO_READABLE is available."""
        from app.api.routes import LABEL_TO_READABLE
        
        assert LABEL_TO_READABLE is not None
    
    def test_merge_bpe_imported(self):
        """Test merge_bpe_tokens_tagging is available."""
        from app.api.routes import merge_bpe_tokens_tagging
        
        assert callable(merge_bpe_tokens_tagging)
    
    def test_extract_regex_entities_imported(self):
        """Test extract_regex_entities is available."""
        from app.api.routes import extract_regex_entities
        
        assert callable(extract_regex_entities)
    
    def test_fa_version_imported(self):
        """Test FA_VERSION is available."""
        from app.api.routes import FA_VERSION
        
        assert FA_VERSION in ['FA2', 'FA3']
    
    def test_inference_queue_imported(self):
        """Test inference_queue is available."""
        from app.api.routes import inference_queue
        
        assert inference_queue is not None
