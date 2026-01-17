"""
Dynamic Batch Processor using varlen ragged tensor style.

Uses cu_seqlens for efficient batching without padding.
Reference: https://github.com/malaysia-ai/simple-continuous-batching-flashinfer
"""

import torch
import asyncio

from app.env import args, logging
from app.core.attention import FA_VERSION


# ============================================================================
# Inference queue for batch collection
# ============================================================================
inference_queue = asyncio.Queue()


async def process_batch_queue(tokenizer, model):
    """
    Background task that processes batched inference requests.
    
    Uses varlen ragged tensor style batching:
    - Concatenate sequences on seq_len dimension
    - Store cumsum lengths (cu_seqlens) for optimization
    - No padding operation needed!
    
    Reference: https://github.com/malaysia-ai/simple-continuous-batching-flashinfer/blob/main/simple_flashinfer/main.py#L277
    
    Args:
        tokenizer: The loaded tokenizer
        model: The loaded model
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
            with torch.no_grad():
                # Tokenize all inputs
                encoded_list = []
                word_ids_list = []
                
                for i, text in enumerate(batch):
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
                
                # ============ VARLEN RAGGED TENSOR PACKING ============
                # Compute cu_seqlens (cumulative sequence lengths)
                seq_lengths = [e['input_ids'].shape[1] for e in encoded_list]
                total_tokens = sum(seq_lengths)
                max_seqlen = max(seq_lengths)
                
                # cu_seqlens: [0, len1, len1+len2, len1+len2+len3, ...]
                cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device='cuda')
                for i, length in enumerate(seq_lengths):
                    cu_seqlens[i + 1] = cu_seqlens[i] + length
                
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
                
                logging.debug(f"{FA_VERSION} varlen batch: {len(seq_lengths)} seqs, {total_tokens} tokens, max={max_seqlen}")
                
                # Forward pass with cu_seqlens passed via kwargs
                outputs = model(
                    input_ids=packed_input_ids,
                    attention_mask=packed_attention_mask,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                )
                
                # Output logits shape: [1, total_tokens, num_labels]
                logits = outputs.logits.squeeze(0)  # [total_tokens, num_labels]
                predictions = torch.argmax(logits, dim=-1)  # [total_tokens]
                
                # ============ SPLIT RESULTS BACK BY SEQUENCE ============
                for i, future in enumerate(futures):
                    start_idx = cu_seqlens[i].item()
                    end_idx = cu_seqlens[i + 1].item()
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
                        'flash_attention': FA_VERSION,
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
