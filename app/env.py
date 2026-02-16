import argparse
import logging
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--model', type=str, default='Scicom-intl/multilingual-dynamic-entity-decoder')
parser.add_argument('--loglevel', default='INFO', type=str)
parser.add_argument('--max_batch_size', default=32, type=int, help='Maximum batch size for dynamic batching')
parser.add_argument('--max_seq_len', default=512, type=int, help='Maximum sequence length')
parser.add_argument('--microsleep', default=0.001, type=float, help='Sleep time between batch processing')
parser.add_argument('--memory_utilization', default=0.9, type=float, help='GPU memory utilization')
parser.add_argument(
    '--torch_dtype',
    default='bfloat16',
    type=str,
    choices=['float16', 'bfloat16', 'float32'],
    help='Torch dtype for computation'
)

# Only parse args if not running tests
import sys
if len(sys.argv) > 0 and ('unittest' in sys.argv[0] or 'pytest' in sys.argv[0] or 'test' in sys.argv[0]):
    # Running tests - use defaults without parsing
    args = argparse.Namespace(
        host='0.0.0.0',
        port=8000,
        model='Scicom-intl/multilingual-dynamic-entity-decoder',
        loglevel='INFO',
        max_batch_size=32,
        max_seq_len=512,
        microsleep=0.001,
        memory_utilization=0.9,
        torch_dtype='bfloat16',
    )
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Convert dtype string to torch dtype
dtype_map = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}
args.torch_dtype = dtype_map[args.torch_dtype]
