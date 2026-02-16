"""
Test app/api/routes.py endpoints using FastAPI TestClient.

Uses mocks to avoid loading the actual model.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
import asyncio


class TestAPIEndpoints:
    """Test API endpoints with mocked model"""
    
    def test_index_endpoint(self):
        """Test GET / returns API info."""
        from app.api.routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "model" in data
        assert "flash_attention" in data
        assert "varlen_batching" in data
        assert data["varlen_batching"] == True
    
    def test_health_endpoint(self):
        """Test GET /health returns healthy status."""
        from app.api.routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "flash_attention" in data

    def test_predict_endpoint_empty_text(self):
        """Test POST /predict with empty text returns 400."""
        from app.api.routes import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        
        # Add middleware to set request state
        @app.middleware("http")
        async def add_request_id(request, call_next):
            request.state.request_id = "test-id"
            return await call_next(request)
        
        client = TestClient(app)
        response = client.post("/predict", json={"text": ""})
        
        assert response.status_code == 400


class TestRouteImports:
    """Test that routes module imports are working"""
    
    def test_fa_version_is_valid(self):
        """Test FA_VERSION from routes."""
        from app.api.routes import FA_VERSION
        
        assert FA_VERSION in ["FA2", "FA3"]
    
    def test_flash_attn_version_exists(self):
        """Test flash_attn_version from routes."""
        from app.api.routes import flash_attn_version
        
        assert isinstance(flash_attn_version, str)
    
    def test_args_imported(self):
        """Test args from routes."""
        from app.api.routes import args
        
        assert args is not None
        assert hasattr(args, 'model')


class TestEntityExtraction:
    """Test entity extraction logic used in routes"""
    
    def test_regex_entities_in_predict_logic(self):
        """Test regex entity extraction."""
        from app.api.routes import extract_regex_entities
        
        text = "IC 900101-14-5678, phone 0123456789, email test@gmail.com"
        entities = extract_regex_entities(text)
        
        assert len(entities['ic']) == 1
        assert len(entities['phone']) == 1
        assert len(entities['email']) == 1
    
    def test_label_mapping_in_predict(self):
        """Test label mapping used in predict."""
        from app.api.routes import LABEL_TO_TYPE, LABEL_TO_READABLE
        
        # These are used in the predict endpoint
        assert LABEL_TO_TYPE['LABEL_1'] == 'name'
        assert LABEL_TO_TYPE['LABEL_2'] == 'address'
        assert LABEL_TO_READABLE['LABEL_0'] == 'O'
    
    def test_bpe_merge_in_predict(self):
        """Test BPE merge function used in predict."""
        from app.api.routes import merge_bpe_tokens_tagging
        
        tokens = ["ĠAhmad", "Ġfrom", "ĠKL"]
        labels = ["LABEL_1", "LABEL_0", "LABEL_2"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        assert words == ["Ahmad", "from", "KL"]
        assert word_labels == ["LABEL_1", "LABEL_0", "LABEL_2"]


class TestPredictLogic:
    """Test the logic inside predict endpoint without running the endpoint"""
    
    def test_entity_grouping_logic(self):
        """Test entity grouping logic from predict endpoint."""
        from app.api.routes import LABEL_TO_TYPE
        
        # Simulate merged words and labels
        merged_words = ["nama", "saya", "Ahmad", "dari", "KL"]
        merged_labels = ["LABEL_0", "LABEL_1", "LABEL_1", "LABEL_0", "LABEL_2"]
        
        # Entity grouping logic from predict
        model_entities_raw = {'name': [], 'address': []}
        current_entity = None
        
        for i, (word, label) in enumerate(zip(merged_words, merged_labels)):
            if label in LABEL_TO_TYPE:
                entity_type = LABEL_TO_TYPE[label]
                if current_entity and current_entity['type'] == entity_type:
                    current_entity['words'].append(word)
                else:
                    if current_entity:
                        entity_text = ' '.join(current_entity['words'])
                        model_entities_raw[current_entity['type']].append(entity_text)
                    current_entity = {'type': entity_type, 'words': [word]}
            else:
                if current_entity:
                    entity_text = ' '.join(current_entity['words'])
                    model_entities_raw[current_entity['type']].append(entity_text)
                    current_entity = None
        
        if current_entity:
            entity_text = ' '.join(current_entity['words'])
            model_entities_raw[current_entity['type']].append(entity_text)
        
        assert model_entities_raw['name'] == ['saya Ahmad']
        assert model_entities_raw['address'] == ['KL']
    
    def test_filter_entities_containing_regex(self):
        """Test filtering entities that contain regex matches."""
        # Simulate the filter logic from predict
        entity_list = ["Ahmad 0123456789", "Ali", "Siti test@gmail.com"]
        regex_matches = ["0123456789", "test@gmail.com"]
        
        def filter_entities_containing_regex(entities):
            filtered = []
            for entity in entities:
                entity_lower = entity.lower()
                contains_regex = any(regex_text.lower() in entity_lower for regex_text in regex_matches)
                if not contains_regex:
                    filtered.append(entity)
            return filtered
        
        result = filter_entities_containing_regex(entity_list)
        
        assert result == ["Ali"]
    
    def test_masked_text_replacement(self):
        """Test masked text replacement logic."""
        text = "nama saya Ahmad dari KL"
        
        # Simulate masking
        entities = [("Ahmad", "name"), ("KL", "address")]
        masked_text = text
        
        for entity, etype in sorted(entities, key=lambda x: len(x[0]), reverse=True):
            masked_text = masked_text.replace(entity, f"<{etype}>")
        
        assert masked_text == "nama saya <name> dari <address>"
    
    def test_debug_mode_encoder_output(self):
        """Test encoder_output generation for debug mode."""
        from app.api.routes import LABEL_TO_READABLE
        
        merged_words = ["nama", "Ahmad"]
        merged_labels = ["LABEL_0", "LABEL_1"]
        
        encoder_output = []
        for word, label in zip(merged_words, merged_labels):
            readable_label = LABEL_TO_READABLE.get(label, label)
            encoder_output.append({'word': word, 'label': readable_label})
        
        assert encoder_output[0] == {'word': 'nama', 'label': 'O'}
        assert encoder_output[1] == {'word': 'Ahmad', 'label': 'name'}


