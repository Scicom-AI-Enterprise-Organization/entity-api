# entity-api

Continuous batching API for encoder models (Token Classification / NER) using FlashInfer.

## Overview

- Custom Attention Interface using FlashInfer BatchPrefillWithRaggedKVCacheWrapper
- Dynamic batching using varlen ragged tensor style - NO PADDING!
- Encoder model doesn't need KV Cache, so straight forward to implement
- Non-causal (bidirectional) attention support

## How to Install

```bash
git clone https://github.com/malaysia-ai/entity-api && cd entity-api
```

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## How to Run Locally

### Run Scicom-intl/multilingual-dynamic-entity-decoder

```bash
CUDA_VISIBLE_DEVICES=2 python3 -m flash_infer_encoder_non_causal.main --host 0.0.0.0 --port 7100
```

### Supported Parameters

```bash
python3 -m flash_infer_encoder_non_causal.main --help
```

```text
usage: main.py [-h] [--host HOST] [--port PORT] [--model MODEL] [--loglevel LOGLEVEL]
               [--max_batch_size MAX_BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN]
               [--microsleep MICROSLEEP] [--memory_utilization MEMORY_UTILIZATION]
               [--torch_dtype {float16,bfloat16,float32}]

Configuration parser

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0)
  --port PORT           port to host the app (default: 7100)
  --model MODEL         Model type (default: Scicom-intl/multilingual-dynamic-entity-decoder)
  --loglevel LOGLEVEL   Logging level (default: INFO)
  --max_batch_size MAX_BATCH_SIZE
                        Maximum batch size for dynamic batching (default: 32)
  --max_seq_len MAX_SEQ_LEN
                        Maximum sequence length (default: 512)
  --microsleep MICROSLEEP
                        Sleep time between batch processing (default: 0.001)
  --memory_utilization MEMORY_UTILIZATION
                        GPU memory utilization (default: 0.9)
  --torch_dtype {float16,bfloat16,float32}
                        Torch dtype for computation (default: bfloat16)
```

### Supported Model

- [Scicom-intl/multilingual-dynamic-entity-decoder](https://huggingface.co/Scicom-intl/multilingual-dynamic-entity-decoder)

## Simple API Example

### NER Endpoint

The `/ner` endpoint extracts named entities and returns a masked text with entity placeholders.

**Supported Entity Types:**
- `name` - Person names (from model)
- `address` - Locations/addresses (from model)
- `ic` - Malaysian IC numbers (regex: `XXXXXX-XX-XXXX` or `XXXXXXXXXXXX`)
- `phone` - Phone numbers (regex)
- `email` - Email addresses (regex)

```bash
curl -X POST http://localhost:7100/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "saya dari Kuala Lumpur"}'
```

Output:

```json
{
  "text": "saya dari Kuala Lumpur",
  "masked_text": "saya dari <address>",
  "name": [],
  "address": ["Kuala Lumpur"],
  "ic": [],
  "phone": [],
  "email": []
}
```

**Example with multiple entity types:**

```bash
curl -X POST http://localhost:7100/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "hubungi Ahmad di 0123456789 atau email test@gmail.com"}'
```

Output:

```json
{
  "text": "hubungi Ahmad di 0123456789 atau email test@gmail.com",
  "masked_text": "hubungi <name> di <phone> atau email <email>",
  "name": ["Ahmad"],
  "address": [],
  "ic": [],
  "phone": ["0123456789"],
  "email": ["test@gmail.com"]
}
```

**Example with IC:**

```bash
curl -X POST http://localhost:7100/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "IC saya 900101-14-5678"}'
```

Output:

```json
{
  "text": "IC saya 900101-14-5678",
  "masked_text": "IC <name> <ic>",
  "name": [],
  "address": [],
  "ic": ["900101-14-5678"],
  "phone": [],
  "email": []
}
```

### Batch Prediction

The `/predict` endpoint processes multiple texts and returns the same format as `/ner` for each text.

```bash
curl -X POST http://localhost:7100/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["saya dari Kuala Lumpur", "hubungi Ahmad di 0123456789"]}'
```

Output:

```json
{
  "id": "1706c984-c035-4a5f-8c61-f7b95c93c5ac",
  "results": [
    {
      "text": "saya dari Kuala Lumpur",
      "masked_text": "saya dari <address>",
      "name": [],
      "address": ["Kuala Lumpur"],
      "ic": [],
      "phone": [],
      "email": []
    },
    {
      "text": "hubungi Ahmad di 0123456789",
      "masked_text": "hubungi <name> di <phone>",
      "name": ["Ahmad"],
      "address": [],
      "ic": [],
      "phone": ["0123456789"],
      "email": []
    }
  ]
}
```
