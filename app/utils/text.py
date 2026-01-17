"""
BPE Token Merging utilities.

Reference: https://github.com/malaysia-ai/malaya/blob/master/malaya/text/bpe.py#L1047
"""

from typing import List, Tuple, Optional


def merge_bpe_tokens_tagging(
    tokens: List[str], 
    labels: List[str], 
    rejected: Optional[List[str]] = None, 
    prefix_char: str = 'Ġ'
) -> Tuple[List[str], List[str]]:
    """
    Merge BPE subword tokens back to words with their labels.
    
    This function handles common BPE prefixes (Ġ for GPT-style, ▁ for SentencePiece)
    and merges subword tokens into complete words while preserving the label
    from the first subword.
    
    Args:
        tokens: List of BPE tokens from tokenizer
        labels: List of labels corresponding to each token
        rejected: List of special tokens to skip (e.g., [CLS], [SEP])
        prefix_char: The prefix character used by the tokenizer
        
    Returns:
        Tuple of (merged_words, merged_labels)
        
    Reference: https://github.com/malaysia-ai/malaya/blob/master/malaya/text/bpe.py#L1047
    """
    if rejected is None:
        rejected = ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]', '<unk>']
    
    new_paired_tokens = []
    n_tokens = len(tokens)
    i = 0
    
    while i < n_tokens:
        current_token, current_label = tokens[i], labels[i]
        
        # Skip rejected tokens
        if current_token in rejected:
            i += 1
            continue
        
        # Check if this token is a continuation (no prefix = part of previous word)
        if i > 0 and not current_token.startswith(prefix_char) and not current_token.startswith('▁'):
            if new_paired_tokens:
                # Merge with previous token
                previous_token, previous_label = new_paired_tokens.pop()
                merged_token = previous_token
                merged_label = previous_label  # Keep label from first subword
                
                # Continue merging while tokens don't have prefix
                while (not current_token.startswith(prefix_char) and 
                       not current_token.startswith('▁') and 
                       current_token not in rejected):
                    merged_token = merged_token + current_token
                    i += 1
                    if i < n_tokens:
                        current_token, current_label = tokens[i], labels[i]
                    else:
                        break
                
                new_paired_tokens.append((merged_token, merged_label))
            else:
                new_paired_tokens.append((current_token, current_label))
                i += 1
        else:
            # New word - clean the prefix
            clean_token = current_token
            if clean_token.startswith(prefix_char):
                clean_token = clean_token[1:]
            elif clean_token.startswith('▁'):
                clean_token = clean_token[1:]
            
            new_paired_tokens.append((clean_token, current_label))
            i += 1
    
    # Filter out empty tokens and rejected ones
    words = [t[0] for t in new_paired_tokens if t[0] not in rejected and t[0]]
    word_labels = [t[1] for t in new_paired_tokens if t[0] not in rejected and t[0]]
    
    return words, word_labels
