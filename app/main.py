"""
Flash Attention Encoder API (FA3/FA2)

Entry point for the NER API with Flash Attention support.

Key features:
1. Custom Attention Interface using FA3/FA2 flash_attn_varlen_func
2. Dynamic batching with varlen ragged tensor - no padding operation needed
3. Encoder model - no KV cache needed (single forward pass)
4. BPE token merging for word-level entity extraction
5. Regex extraction for IC, phone, email

Reference: 
- https://github.com/malaysia-ai/simple-continuous-batching-flashinfer
- https://huggingface.co/Scicom-intl/multilingual-dynamic-entity-decoder
"""

import asyncio
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request

from app.env import args, logging
from app.core.model import load_model, get_tokenizer, get_model
from app.core.batch_processor import process_batch_queue
from app.core.attention import FA_VERSION
from app.api.routes import router


# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(
    title="Flash Attention Encoder API",
    description=f"NER API with Flash Attention ({FA_VERSION}) support",
    version="1.0.0",
)

# Include routes
app.include_router(router)


# ============================================================================
# Middleware
# ============================================================================
@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    """Add request ID and timing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    duration = time.perf_counter() - start_time
    logging.info(f"{request_id} completed in {duration:.4f}s")
    
    return response


# ============================================================================
# Lifecycle Events
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize model and start background batch processor."""
    load_model()
    
    # Start batch processor with model references
    tokenizer = get_tokenizer()
    model = get_model()
    app.state.batch_processor = asyncio.create_task(
        process_batch_queue(tokenizer, model)
    )
    
    logging.info(f"Server started on {args.host}:{args.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if hasattr(app.state, 'batch_processor'):
        app.state.batch_processor.cancel()
        logging.info("Batch processor stopped")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        loop="uvloop",
    )
