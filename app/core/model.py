"""
Model loading and initialization.
"""

from transformers import AutoTokenizer, Qwen3ForTokenClassification

from app.env import args, logging
from app.core.attention import register_fa_attention, FA_VERSION


# ============================================================================
# Global model references
# ============================================================================
tokenizer = None
model = None


def load_model():
    """
    Load the encoder model with custom FA attention interface.
    
    Registers custom Flash Attention before loading to ensure model uses FA.
    """
    global tokenizer, model
    
    logging.info(f"Loading model: {args.model}")
    logging.info(f"Flash Attention version: {FA_VERSION}")
    
    # Register custom FA attention BEFORE loading model
    register_fa_attention()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load model with custom attention
    model = Qwen3ForTokenClassification.from_pretrained(
        args.model,
        attn_implementation="fa_varlen",
        torch_dtype=args.torch_dtype,
    ).eval().cuda()
    
    logging.info(f"Model loaded with attention: fa_varlen")
    
    config = model.config
    logging.info(f"Model config: layers={config.num_hidden_layers}, "
                 f"heads={config.num_attention_heads}, "
                 f"kv_heads={getattr(config, 'num_key_value_heads', config.num_attention_heads)}, "
                 f"hidden={config.hidden_size}, head_dim={config.head_dim}")
    logging.info(f"Number of labels: {config.num_labels}")
    logging.info(f"ID to Label mapping: {config.id2label}")


def get_tokenizer():
    """Get the loaded tokenizer."""
    global tokenizer
    return tokenizer


def get_model():
    """Get the loaded model."""
    global model
    return model
