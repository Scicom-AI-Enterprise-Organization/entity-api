"""
Flash Attention Encoder Non-Causal API

This module implements a continuous batching API for encoder models (non-causal/bidirectional)
using FlashInfer with variable length sequences (ragged tensor style).

Key features:
1. Custom Attention Interface using FlashInfer BatchPrefillWithRaggedKVCacheWrapper
2. Dynamic batching with qo_indptr (cu_seqlens) for variable length sequences - NO PADDING!
3. No KV cache needed (encoder model - single forward pass)
4. Non-causal attention (bidirectional)

Reference: https://github.com/malaysia-ai/simple-continuous-batching-flashinfer
Model: https://huggingface.co/Scicom-intl/multilingual-dynamic-entity-decoder
"""

from flash_infer_encoder_non_causal.env import args, logging
from flash_infer_encoder_non_causal.parameters import (
    EntityRequest,
    EntityResponse,
    TokenClassificationRequest,
)
from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, Qwen3ForTokenClassification, AttentionInterface
from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
import flashinfer
import asyncio
import uvicorn
import time
import uuid

# FlashInfer ragged batch prefill wrapper (global, reused across requests)
prefill_wrapper = None
workspace_buffer = None

# Attention state for current batch (thread-local style via globals)
current_batch_state: Dict[str, Any] = {}


def init_flashinfer_wrapper():
    """Initialize FlashInfer workspace and wrapper for ragged batch prefill."""
    global prefill_wrapper, workspace_buffer
    
    # Allocate 128MB workspace buffer for FlashInfer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    
    # Create ragged batch prefill wrapper
    # kv_layout="NHD" means [num_tokens, num_heads, head_dim]
    prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )
    logging.info("FlashInfer ragged batch prefill wrapper initialized")


def register_fa_attention():
    """
    Register custom non-causal Flash Attention interface using FlashInfer.
    
    Uses FlashInfer's BatchPrefillWithRaggedKVCacheWrapper for efficient
    varlen ragged tensor batching WITHOUT padding.
    
    Input shapes (from transformers):
        query: [1, num_heads, total_tokens, head_dim]  (packed sequences)
        key: [1, num_kv_heads, total_tokens, head_dim]
        value: [1, num_kv_heads, total_tokens, head_dim]
    
    FlashInfer expects (NHD layout):
        query: [total_tokens, num_heads, head_dim]
        key: [total_tokens, num_kv_heads, head_dim]
        value: [total_tokens, num_kv_heads, head_dim]
    """
    
    def flashinfer_attention_forward(
        module: AttentionInterface,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
    ):
        """
        Custom FlashInfer attention forward with varlen ragged tensor support.
        
        Uses ragged tensor batching with qo_indptr for variable length sequences.
        Handles GQA (Grouped Query Attention) properly.
        """
        global current_batch_state, prefill_wrapper
        
        batch_size, num_heads, seq_len, head_dim = query.shape
        num_kv_heads = key.shape[1]
        
        # Determine is_causal correctly (same logic as native SDPA)
        if is_causal is None:
            is_causal = seq_len > 1 and attention_mask is None and getattr(module, 'is_causal', True)
        
        # Get scaling factor from module or compute default
        sm_scale = scaling if scaling is not None else (1.0 / (head_dim ** 0.5))
        
        # Check if we're in varlen mode (qo_indptr provided)
        qo_indptr = current_batch_state.get("qo_indptr", None)
        
        if qo_indptr is not None and prefill_wrapper is not None:
            # ============ VARLEN RAGGED TENSOR MODE ============
            # Query shape: [1, num_heads, total_tokens, head_dim]
            # Need: [total_tokens, num_heads, head_dim] for FlashInfer NHD layout
            
            q = query.squeeze(0).transpose(0, 1).contiguous()
            k = key.squeeze(0).transpose(0, 1).contiguous()
            v = value.squeeze(0).transpose(0, 1).contiguous()
            
            # Plan the batch prefill with correct parameters
            prefill_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,  # Same as qo for self-attention
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                causal=is_causal,  # Use proper is_causal!
                q_data_type=q.dtype,
                sm_scale=sm_scale,  # Pass scaling factor
            )
            
            # Run the attention
            attn_output = prefill_wrapper.run(q, k, v)
            
            # Output: [total_tokens, num_heads, head_dim] -> [1, total_tokens, num_heads, head_dim]
            attn_output = attn_output.unsqueeze(0)
            
            return attn_output, None
        
        else:
            # ============ STANDARD PADDED BATCH MODE ============
            # Use native SDPA with proper GQA support
            sdpa_kwargs = {'enable_gqa': True} if num_heads != num_kv_heads else {}
            
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                is_causal=is_causal,
                scale=sm_scale,
                **sdpa_kwargs,
            )
            # Transpose to [batch, seq, heads, dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, None

    AttentionInterface.register("fa_noncausal", flashinfer_attention_forward)
    logging.info("Registered 'fa_noncausal' attention interface with FlashInfer ragged batch prefill")


class DynamicBatchManager:
    """
    Manages dynamic batching for encoder inference.
    
    Uses ragged tensor style with cu_seqlens for variable length sequences.
    No KV cache needed since encoder processes all tokens in one forward pass.
    
    Reference for varlen batching:
    https://github.com/malaysia-ai/simple-continuous-batching-flashinfer/blob/main/simple_flashinfer/main.py#L279
    """
    
    def __init__(self, max_batch_size: int = 32, max_seq_len: int = 512):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pending_requests = []
        self.lock = asyncio.Lock()
    
    def compute_cu_seqlens(self, seq_lengths: List[int]) -> tuple:
        """
        Compute cumulative sequence lengths for varlen attention.
        
        This is the key to ragged tensor style batching - instead of padding
        all sequences to max length, we pack them together and use cu_seqlens
        to track where each sequence starts/ends.
        
        Args:
            seq_lengths: List of sequence lengths for each sample in batch
            
        Returns:
            cu_seqlens: Cumulative sequence lengths [0, len1, len1+len2, ...]
            max_seqlen: Maximum sequence length in batch
            
        Example:
            seq_lengths = [3, 5, 2]
            cu_seqlens = [0, 3, 8, 10]
            max_seqlen = 5
        """
        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length
        max_seqlen = max(seq_lengths)
        return cu_seqlens, max_seqlen
    
    def pack_sequences(
        self, 
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor]
    ) -> dict:
        """
        Pack variable length sequences into a single tensor for efficient processing.
        
        This is the ragged tensor style batching where sequences are concatenated
        and cu_seqlens tracks boundaries.
        
        Args:
            input_ids_list: List of input_ids tensors (variable lengths)
            attention_mask_list: List of attention_mask tensors
            
        Returns:
            Dictionary with packed tensors and metadata
        """
        seq_lengths = [ids.shape[-1] for ids in input_ids_list]
        cu_seqlens, max_seqlen = self.compute_cu_seqlens(seq_lengths)
        
        # Concatenate all sequences (no padding needed!)
        packed_input_ids = torch.cat([ids.squeeze(0) for ids in input_ids_list], dim=0)
        packed_attention_mask = torch.cat([mask.squeeze(0) for mask in attention_mask_list], dim=0)
        
        return {
            'input_ids': packed_input_ids.unsqueeze(0),  # [1, total_tokens]
            'attention_mask': packed_attention_mask.unsqueeze(0),
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
            'seq_lengths': seq_lengths,
            'batch_size': len(seq_lengths),
        }


# Global variables
tokenizer = None
model = None
batch_manager = None
inference_queue = asyncio.Queue()

app = FastAPI(title="Flash Attention Encoder Non-Causal API")


def load_model():
    """
    Load the encoder model with custom FlashInfer attention interface.
    
    Uses Qwen3ForTokenClassification with 'fa_noncausal' attention implementation
    for efficient non-causal (bidirectional) attention with varlen ragged batching.
    """
    global tokenizer, model, batch_manager
    
    logging.info(f"Loading model: {args.model}")
    
    # Initialize FlashInfer workspace and wrapper
    init_flashinfer_wrapper()
    
    # Register custom FA attention BEFORE loading model
    register_fa_attention()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Use custom FlashInfer attention with varlen ragged tensor support
    attn_impl = "fa_noncausal"
    logging.info("Using FlashInfer attention with varlen ragged tensor batching")
    
    # Load Qwen3ForTokenClassification model
    try:
        model = Qwen3ForTokenClassification.from_pretrained(
            args.model,
            attn_implementation=attn_impl,
            dtype=args.torch_dtype,
        ).eval().cuda()
        logging.info(f"Model loaded with attention: {attn_impl}")
    except Exception as e:
        logging.warning(f"Could not load with fa_noncausal: {e}")
        logging.info("Falling back to SDPA attention")
        model = Qwen3ForTokenClassification.from_pretrained(
            args.model,
            attn_implementation="sdpa",
            dtype=args.torch_dtype,
        ).eval().cuda()
    
    batch_manager = DynamicBatchManager(
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len
    )
    
    # Log model config
    config = model.config
    logging.info(f"Model config: layers={config.num_hidden_layers}, "
                 f"heads={config.num_attention_heads}, "
                 f"kv_heads={getattr(config, 'num_key_value_heads', config.num_attention_heads)}, "
                 f"hidden={config.hidden_size}, head_dim={config.head_dim}")
    logging.info(f"Number of labels: {config.num_labels}")
    logging.info(f"ID to Label mapping: {config.id2label}")


async def process_batch_queue():
    """
    Background task that processes batched inference requests.
    
    Collects requests from the queue, batches them together using
    ragged tensor style (cu_seqlens), and runs inference.
    
    For encoder models, we support two batching modes:
    1. Padded batching: Pad all sequences to max length (simpler)
    2. Varlen batching: Pack sequences with cu_seqlens (more efficient)
    
    Reference: https://github.com/malaysia-ai/simple-continuous-batching-flashinfer/blob/main/simple_flashinfer/main.py#L279
    """
    while True:
        await asyncio.sleep(args.microsleep)
        
        batch = []
        futures = []
        texts_raw = []
        is_split_words = []
        
        # Collect requests from queue
        while not inference_queue.empty() and len(batch) < args.max_batch_size:
            try:
                request_data = await asyncio.wait_for(inference_queue.get(), timeout=1e-6)
                batch.append(request_data['inputs'])
                futures.append(request_data['future'])
                texts_raw.append(request_data.get('text_raw', request_data['inputs']))
                is_split_words.append(request_data.get('is_split_into_words', False))
            except asyncio.TimeoutError:
                break
        
        if not batch:
            continue
        
        try:
            # Process batch using VARLEN RAGGED TENSOR STYLE (no padding!)
            with torch.no_grad():
                global current_batch_state
                
                # Tokenize all inputs
                encoded_list = []
                word_ids_list = []
                
                for i, text in enumerate(batch):
                    # Support both regular text and pre-split words
                    if is_split_words[i] and isinstance(text, str):
                        words = text.split()
                        encoded = tokenizer(
                            words,
                            is_split_into_words=True,
                            return_tensors='pt',
                            truncation=True,
                            max_length=args.max_seq_len,
                            padding=False,
                        )
                        word_ids = encoded.word_ids()
                    else:
                        encoded = tokenizer(
                            text,
                            return_tensors='pt',
                            truncation=True,
                            max_length=args.max_seq_len,
                            padding=False,
                        )
                        word_ids = None
                    
                    encoded_list.append({
                        'input_ids': encoded['input_ids'].cuda(),
                        'attention_mask': encoded['attention_mask'].cuda(),
                    })
                    word_ids_list.append(word_ids)
                
                # Get sequence lengths
                seq_lengths = [e['input_ids'].shape[1] for e in encoded_list]
                total_tokens = sum(seq_lengths)
                
                # ============ VARLEN RAGGED TENSOR PACKING ============
                # Compute qo_indptr (cumulative sequence lengths) for FlashInfer
                qo_indptr = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
                for i, length in enumerate(seq_lengths):
                    qo_indptr[i + 1] = qo_indptr[i] + length
                
                # Pack all sequences into single tensor WITHOUT PADDING
                packed_input_ids = torch.cat(
                    [e['input_ids'].squeeze(0) for e in encoded_list], 
                    dim=0
                ).unsqueeze(0)  # [1, total_tokens]
                
                # Attention mask is all 1s for packed sequences (no padding!)
                packed_attention_mask = torch.ones(
                    1, total_tokens, 
                    dtype=torch.long, 
                    device='cuda'
                )
                
                # Set global state for the attention function
                current_batch_state = {
                    "qo_indptr": qo_indptr,
                    "seq_lengths": seq_lengths,
                    "batch_size": len(seq_lengths),
                }
                
                logging.debug(f"Varlen batch: {len(seq_lengths)} seqs, {total_tokens} total tokens")
                
                # Forward pass with packed sequences
                outputs = model(
                    input_ids=packed_input_ids,
                    attention_mask=packed_attention_mask,
                )
                
                # Clear batch state
                current_batch_state = {}
                
                # Output logits shape: [1, total_tokens, num_labels]
                logits = outputs.logits.squeeze(0)  # [total_tokens, num_labels]
                predictions = torch.argmax(logits, dim=-1)  # [total_tokens]
                
                # ============ SPLIT RESULTS BACK BY SEQUENCE ============
                for i, future in enumerate(futures):
                    start_idx = qo_indptr[i].item()
                    end_idx = qo_indptr[i + 1].item()
                    seq_len = seq_lengths[i]
                    
                    pred = predictions[start_idx:end_idx].cpu().tolist()
                    logit = logits[start_idx:end_idx].cpu()
                    
                    labels = [model.config.id2label.get(p, f"LABEL_{p}") for p in pred]
                    
                    tokens = tokenizer.convert_ids_to_tokens(
                        encoded_list[i]['input_ids'].squeeze(0).cpu().tolist()
                    )
                    
                    result = {
                        'predictions': pred,
                        'labels': labels,
                        'tokens': tokens,
                        'seq_length': seq_len,
                        'varlen_batch': True,
                    }
                    
                    if word_ids_list[i] is not None:
                        result['word_ids'] = word_ids_list[i]
                    
                    if len(batch) == 1:
                        result['logits'] = logit.tolist()
                    
                    future.set_result(result)
                    
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            for future in futures:
                if not future.done():
                    future.set_exception(e)


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


@app.get('/')
async def index():
    return {
        'message': 'FlashInfer Encoder Non-Causal API (Varlen Ragged Tensor)',
        'model': args.model,
        'flashinfer_enabled': prefill_wrapper is not None,
        'varlen_batching': True,
    }


@app.get('/health')
async def health():
    return {'status': 'healthy', 'model_loaded': model is not None}


@app.post('/predict')
async def predict(form: EntityRequest, request: Request):
    """
    Process entity extraction/token classification request.
    
    Uses dynamic batching with Flash Attention for efficient inference.
    Supports both regular text and word-split input for NER.
    """
    request_id = request.state.request_id
    
    results = []
    for text in form.texts:
        future = asyncio.Future()
        await inference_queue.put({
            'inputs': text,
            'future': future,
            'request_id': request_id,
            'is_split_into_words': False,
        })
        
        result = await future
        results.append(result)
    
    return EntityResponse(
        id=request_id,
        results=results,
        usage={
            'num_texts': len(form.texts),
            'total_tokens': sum(r['seq_length'] for r in results),
        }
    )


@app.post('/ner')
async def ner(request: Request):
    """
    Named Entity Recognition endpoint.
    
    Accepts text and returns extracted entities with their labels.
    Uses is_split_into_words=True for word-level NER.
    
    Example:
        POST /ner
        {"text": "Hi, my name is Alex and I'm from Perlis"}
        
        Response:
        {
            "entities": [
                {"word": "Alex", "label": "PER", "start": 4, "end": 5},
                {"word": "Perlis", "label": "LOC", "start": 9, "end": 10}
            ]
        }
    """
    body = await request.json()
    text = body.get('text', '')
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    future = asyncio.Future()
    await inference_queue.put({
        'inputs': text,
        'text_raw': text,
        'future': future,
        'request_id': request.state.request_id,
        'is_split_into_words': True,
    })
    
    result = await future
    
    # Extract entities from predictions
    entities = []
    current_entity = None
    words = text.split()  # Original words for reference
    word_ids = result.get('word_ids', [])
    
    for i, (label, token) in enumerate(zip(result['labels'], result['tokens'])):
        # Skip special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            continue
        
        # Get word index if available
        word_idx = word_ids[i] if word_ids and i < len(word_ids) else None
        
        if label.startswith('B-'):
            # Start of new entity
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                'label': entity_type,
                'tokens': [token],
                'start_idx': i,
                'word_idx': word_idx,
            }
        elif label.startswith('I-') and current_entity:
            # Continuation of entity
            current_entity['tokens'].append(token)
        else:
            # O label or mismatch
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    # Format entities
    formatted_entities = []
    for ent in entities:
        word_idx = ent.get('word_idx')
        # Get original word if word_idx is available
        original_word = words[word_idx] if word_idx is not None and word_idx < len(words) else None
        formatted_entities.append({
            'text': tokenizer.convert_tokens_to_string(ent['tokens']).strip(),
            'label': ent['label'],
            'start_token': ent['start_idx'],
            'word_idx': word_idx,
            'original_word': original_word,
        })
    
    return {
        'id': request.state.request_id,
        'text': text,
        'entities': formatted_entities,
        'raw_labels': result['labels'],
        'tokens': result['tokens'],
    }


@app.post('/batch_predict')
async def batch_predict(form: TokenClassificationRequest, request: Request):
    """
    Batch prediction endpoint.
    
    All texts are processed together in a single batch using
    ragged tensor style with cu_seqlens.
    """
    request_id = request.state.request_id
    
    # Handle single text or list
    texts = form.text if isinstance(form.text, list) else [form.text]
    
    futures = []
    for text in texts:
        future = asyncio.Future()
        await inference_queue.put({
            'inputs': text,
            'future': future,
            'request_id': request_id,
        })
        futures.append(future)
    
    # Wait for all results
    results = await asyncio.gather(*[f for f in futures])
    
    return {
        'id': request_id,
        'results': results,
        'batch_size': len(texts),
    }


@app.post('/v1/embeddings')
async def embeddings(request: Request):
    """
    Get embeddings from the encoder model.
    
    Returns the hidden states from the last layer.
    """
    body = await request.json()
    texts = body.get('input', [])
    if isinstance(texts, str):
        texts = [texts]
    
    with torch.no_grad():
        encoded = tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=args.max_seq_len,
            padding=True,
        ).to('cuda')
        
        outputs = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            output_hidden_states=True,
        )
        
        # Use last hidden state, mean pool over tokens
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        
        # Mean pooling
        sum_hidden = (hidden_states * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1)
        embeddings = (sum_hidden / count).cpu().tolist()
    
    return {
        'object': 'list',
        'data': [
            {'object': 'embedding', 'index': i, 'embedding': emb}
            for i, emb in enumerate(embeddings)
        ],
        'model': args.model,
        'usage': {'prompt_tokens': encoded['input_ids'].numel()},
    }


@app.on_event("startup")
async def startup_event():
    """Initialize model and start background batch processor."""
    load_model()
    app.state.batch_processor = asyncio.create_task(process_batch_queue())
    logging.info(f"Server started on {args.host}:{args.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if hasattr(app.state, 'batch_processor'):
        app.state.batch_processor.cancel()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
    )
