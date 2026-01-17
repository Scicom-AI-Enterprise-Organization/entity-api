"""
FastAPI routes for NER prediction.
"""

import asyncio
from fastapi import APIRouter, Request, HTTPException

from app.env import args
from app.core.attention import FA_VERSION, flash_attn_version
from app.core.batch_processor import inference_queue
from app.utils.text import merge_bpe_tokens_tagging
from app.utils.patterns import (
    REGEX_PATTERNS, 
    LABEL_TO_TYPE, 
    LABEL_TO_READABLE,
    extract_regex_entities,
)


router = APIRouter()


@router.get('/')
async def index():
    """API information endpoint."""
    return {
        'message': f'Flash Attention Encoder API ({FA_VERSION})',
        'model': args.model,
        'flash_attention': FA_VERSION,
        'flash_attn_version': flash_attn_version,
        'varlen_batching': True,
    }


@router.get('/health')
async def health():
    """Health check endpoint."""
    from app.core.model import model
    return {
        'status': 'healthy', 
        'model_loaded': model is not None, 
        'flash_attention': FA_VERSION
    }


@router.post('/predict')
async def predict_single(request: Request):
    """
    Named Entity Recognition endpoint with token merging and regex support.
    
    Request body:
        - text: Input text to analyze
        - debug_mode: If true, include raw encoder output
        
    Returns:
        - text: Original text
        - masked_text: Text with entities replaced by tags
        - name: List of detected names
        - address: List of detected addresses
        - ic: List of detected IC numbers (regex)
        - phone: List of detected phone numbers (regex)
        - email: List of detected emails (regex)
        - encoder_output: Raw token labels (if debug_mode)
    """
    body = await request.json()
    text = body.get('text', '')
    debug_mode = body.get('debug_mode', False)
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Run model prediction via batch queue
    future = asyncio.Future()
    await inference_queue.put({
        'inputs': text,
        'text_raw': text,
        'future': future,
        'request_id': request.state.request_id,
        'is_split_into_words': False,
    })
    
    result = await future
    
    # Merge BPE tokens with labels
    merged_words, merged_labels = merge_bpe_tokens_tagging(
        result['tokens'], 
        result['labels']
    )
    
    # Build encoder_output for debug mode
    encoder_output = None
    if debug_mode:
        encoder_output = []
        for word, label in zip(merged_words, merged_labels):
            readable_label = LABEL_TO_READABLE.get(label, label)
            encoder_output.append({'word': word, 'label': readable_label})
    
    # Extract model entities by grouping consecutive same labels
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
    
    # Extract regex entities from ORIGINAL text
    regex_entities = extract_regex_entities(text)
    regex_matches = []
    for entity_type in regex_entities:
        for match in regex_entities[entity_type]:
            regex_matches.append(match.lower())
    
    # Filter model entities that contain regex patterns
    def filter_entities_containing_regex(entity_list):
        filtered = []
        for entity in entity_list:
            entity_lower = entity.lower()
            contains_regex = any(regex_text in entity_lower for regex_text in regex_matches)
            if not contains_regex:
                filtered.append(entity)
        return filtered
    
    filtered_names = filter_entities_containing_regex(model_entities_raw['name'])
    filtered_addresses = filter_entities_containing_regex(model_entities_raw['address'])
    
    # Build masked_text
    masked_text = text
    
    all_filtered_entities = []
    for entity in filtered_names:
        all_filtered_entities.append((entity, 'name'))
    for entity in filtered_addresses:
        all_filtered_entities.append((entity, 'address'))
    
    # Sort by length (longest first) to avoid partial replacements
    all_filtered_entities.sort(key=lambda x: len(x[0]), reverse=True)
    
    for entity, etype in all_filtered_entities:
        masked_text = masked_text.replace(entity, f'<{etype}>')
    
    for entity_type, entities in regex_entities.items():
        for entity_text in entities:
            masked_text = masked_text.replace(entity_text, f"<{entity_type}>")
    
    response = {
        'text': text,
        'masked_text': masked_text,
        'name': filtered_names,
        'address': filtered_addresses,
        'ic': regex_entities['ic'],
        'phone': regex_entities['phone'],
        'email': regex_entities['email'],
    }
    
    if debug_mode:
        response['encoder_output'] = encoder_output
    
    return response
