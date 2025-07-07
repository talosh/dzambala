import os
import sys
import json

# Get the absolute path to ../gemma relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
gemma_path = os.path.abspath(os.path.join(script_dir, "../private/llm"))

# Add to PYTHONPATH
sys.path.insert(0, gemma_path)

import signal
def create_graceful_exit():
    def graceful_exit(signum, frame):
        print()
        exit(0)
        # signal.signal(signum, signal.SIG_DFL)
        # os.kill(os.getpid(), signal.SIGINT)
    return graceful_exit
signal.signal(signal.SIGINT, create_graceful_exit())


import contextlib
import random

import numpy as np
import torch
from absl import app, flags

from gemma import config
from gemma import model as gemma_model

# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt', None, 'Path to the checkpoint file.', required=True)
flags.DEFINE_string('variant', '1b', 'Model variant.')
flags.DEFINE_string('device', 'cpu', 'Device to run the model on.')
flags.DEFINE_integer('output_len', 10, 'Length of the output sequence.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')
flags.DEFINE_string('prompt', 'What are large language models?', 'Input prompt for the model.')

# Define valid text only model variants
_VALID_MODEL_VARIANTS = ['1b', '2b', '2b-v2', '7b', '9b', '27b', '1b']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda', 'cuda:1', 'cuda:2']

# Validator function for the 'variant' flag
def validate_variant(variant):
    if variant not in _VALID_MODEL_VARIANTS:
        raise ValueError(f'Invalid variant: {variant}. Valid variants are: {_VALID_MODEL_VARIANTS}')
    return True

# Validator function for the 'device' flag
def validate_device(device):
    if device not in _VALID_DEVICES:
        raise ValueError(f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}')
    return True

# Register the validator for the 'variant' flag
flags.register_validator('variant', validate_variant, message='Invalid model variant.')

# Register the validator for the 'device' flag
flags.register_validator('device', validate_device, message='Invalid device.')

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def main(_):

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Construct the model config.
    model_config = config.get_model_config(FLAGS.variant)
    model_config.dtype = "float32"
    model_config.quant = FLAGS.quant

    model_config.tokenizer = f'{os.path.dirname(FLAGS.ckpt)}/tokenizer.model'

    # Seed random.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # Create the model and load the weights.
    device = torch.device(FLAGS.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config).to(device).eval()
        current_state_dict = torch.load(FLAGS.ckpt, mmap=True, weights_only=True, map_location=device)
        model.load_state_dict(current_state_dict['model_state_dict'], strict=False)
        del current_state_dict
        import gc
        gc.collect()
        torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()
        # model.load_weights(FLAGS.ckpt)
        # model = model.to(device).eval()
    print("Model loading done")

    warnings.resetwarnings()


    def sanitize_name(name_to_sanitize):
        if name_to_sanitize is None:
            return None
        
        import re

        stripped_name = name_to_sanitize.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        result = exp.sub('_', stripped_name)
        return re.sub('_\_+', '_', result)


    file_path = os.path.join(
        os.path.dirname(__file__),
        'glossary.txt'
        )

    out_dir = os.path.join(
        os.path.dirname(__file__),
        'gloss'
        )

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    data = []
    block = []

    for line in lines:
        if line.strip() == '':
            if len(block) == 2:
                data.append({'prompt': block[0], 'summ': block[1]})
            block = []
        else:
            block.append(line.strip())

    # Handle the last block if the file does not end with an empty line
    if len(block) == 2:
        data.append({'prompt': block[0], 'summ': block[1]})

    # Example: print the result
    for item in data:
        summ_encoded = model.tokenizer.encode(item['summ'], bos=False)
        summ_decoded = model.tokenizer.decode(summ_encoded[:48])
        item['summ'] = summ_decoded

        json_file_name = f'{sanitize_name(item["prompt"].lower())}.json'
        with open(os.path.join(out_dir, json_file_name), "w") as f:
            json.dump(item, f, indent=4, default=str)
        print (f'{json_file_name} ', end='')
        print(item)


if __name__ == "__main__":
    app.run(main)
