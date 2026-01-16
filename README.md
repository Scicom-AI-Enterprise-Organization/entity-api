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

### Run meta-llama/Llama-3.2-1B-Instruct

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

```bash
curl -X POST http://localhost:7100/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi my name is Alex from Perlis"}'
```

Output:

```json
{
  "id": "a3d02af9-f4fb-4015-b19b-9a65435ca9d6",
  "text": "Hi my name is Alex from Perlis",
  "entities": [],
  "raw_labels": ["LABEL_0", "LABEL_1", "LABEL_1", "LABEL_1", "LABEL_1", "LABEL_0", "LABEL_2", "LABEL_2"],
  "tokens": ["Hi", "my", "name", "is", "Alex", "from", "Per", "lis"]
}
```

Labels:
- `LABEL_0`: Non-entity (O)
- `LABEL_1`: Name entity
- `LABEL_2`: Address entity

### Batch Prediction

```bash
curl -X POST http://localhost:7100/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "My name is John from KL"]}'
```

Output:

```json
{
  "id": "request-id",
  "results": [
    {
      "predictions": [0, 0],
      "labels": ["LABEL_0", "LABEL_0"],
      "tokens": ["Hello", "world"],
      "seq_length": 2,
      "varlen_batch": true
    },
    {
      "predictions": [1, 1, 1, 1, 0, 2],
      "labels": ["LABEL_1", "LABEL_1", "LABEL_1", "LABEL_1", "LABEL_0", "LABEL_2"],
      "tokens": ["My", "name", "is", "John", "from", "KL"],
      "seq_length": 6,
      "varlen_batch": true
    }
  ],
  "usage": {
    "num_texts": 2,
    "total_tokens": 8
  }
}
```
