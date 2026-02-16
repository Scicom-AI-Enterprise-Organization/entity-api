"""
Test app/main.py module - FastAPI application.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestMainModule:
    """Test main FastAPI application"""
    
    def test_app_exists(self):
        """Test FastAPI app is created."""
        from app.main import app
        
        assert app is not None
    
    def test_app_title(self):
        """Test app has correct title."""
        from app.main import app
        
        assert app.title == "Flash Attention Encoder API"
    
    def test_app_has_router(self):
        """Test app includes router."""
        from app.main import app
        
        # Check that routes are included
        routes = [route.path for route in app.routes]
        assert "/" in routes or any("/" in str(r) for r in routes)
    
    def test_fa_version_in_description(self):
        """Test FA version is in app description."""
        from app.main import app
        from app.core.attention import FA_VERSION
        
        assert FA_VERSION in app.description


class TestMiddleware:
    """Test middleware configuration"""
    
    def test_add_request_tracking_middleware_exists(self):
        """Test request tracking middleware is added."""
        from app.main import add_request_tracking
        
        # The middleware function should be callable
        assert callable(add_request_tracking)


class TestLifecycleEvents:
    """Test startup/shutdown event handlers"""
    
    def test_startup_event_defined(self):
        """Test startup event is defined."""
        from app.main import startup_event
        
        # Should be an async function
        assert callable(startup_event)
    
    def test_shutdown_event_defined(self):
        """Test shutdown event is defined."""
        from app.main import shutdown_event
        
        # Should be an async function
        assert callable(shutdown_event)


class TestAppConfiguration:
    """Test app configuration from env"""
    
    def test_args_imported(self):
        """Test args are imported in main."""
        from app.main import args
        
        assert args is not None
        assert hasattr(args, 'host')
        assert hasattr(args, 'port')
    
    def test_logging_imported(self):
        """Test logging is imported in main."""
        from app.main import logging
        
        assert logging is not None


class TestMainImports:
    """Test all imports in main module"""
    
    def test_asyncio_imported(self):
        """Test asyncio is available."""
        from app.main import asyncio
        
        assert asyncio is not None
    
    def test_time_imported(self):
        """Test time module."""
        from app.main import time
        
        assert time is not None
    
    def test_uuid_imported(self):
        """Test uuid module."""
        from app.main import uuid
        
        assert uuid is not None
    
    def test_uvicorn_imported(self):
        """Test uvicorn module."""
        from app.main import uvicorn
        
        assert uvicorn is not None
    
    def test_fastapi_imported(self):
        """Test FastAPI is imported."""
        from app.main import FastAPI, Request
        
        assert FastAPI is not None
        assert Request is not None
    
    def test_router_imported(self):
        """Test router is imported."""
        from app.main import router
        
        assert router is not None
    
    def test_load_model_imported(self):
        """Test load_model is imported."""
        from app.main import load_model
        
        assert callable(load_model)
    
    def test_get_tokenizer_imported(self):
        """Test get_tokenizer is imported."""
        from app.main import get_tokenizer
        
        assert callable(get_tokenizer)
    
    def test_get_model_imported(self):
        """Test get_model is imported."""
        from app.main import get_model
        
        assert callable(get_model)
    
    def test_process_batch_queue_imported(self):
        """Test process_batch_queue is imported."""
        from app.main import process_batch_queue
        
        assert callable(process_batch_queue)
    
    def test_fa_version_imported(self):
        """Test FA_VERSION is imported."""
        from app.main import FA_VERSION
        
        assert FA_VERSION in ["FA2", "FA3"]
