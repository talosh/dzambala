import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import contextlib

import time
import argparse
import platform
import json
import textwrap
import random
import queue
import threading
from copy import deepcopy
import shutil

exit_event = threading.Event()  # For all threads
model_loaded_event = threading.Event()

class SummDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data_root, 
            batch_size = 4,
            device = None, 
            sequential = False
            ):

        self.data_root = data_root
        self.batch_size = batch_size
        self.sequential = sequential

        print (f'scanning json files in {self.data_root}...',  flush=True)
        self.train_descriptions = self.find_json_files(data_root)
        print (f'found {len(self.train_descriptions)} files.')

        self.initial_train_descriptions = list(self.train_descriptions)

        if not self.sequential:
            print ('Reshuffling training data...')
            self.reshuffle()
        else:
            print (f'Sequential: {self.sequential}')
        
        self.item_queue = queue.Queue(maxsize=24)
        self.item_thread = threading.Thread(target=self.read_items_thread, args=(self.item_queue, ))
        self.item_thread.daemon = True
        self.item_thread.start()

    def reshuffle(self):
        random.shuffle(self.train_descriptions)

    def find_json_files(self,  root_dir):
        json_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.json'):
                    json_files.append(
                        os.path.abspath(
                            os.path.join(dirpath, filename))
                        )
        return json_files

    def read_items_thread(self, item_queue):
        while not exit_event.is_set():
            for idx in range(len(self.train_descriptions)):
                try:
                    file_path = self.train_descriptions[idx]
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    f.close()
                    data['dataset_index'] = idx
                    data['file_path']  = file_path
                    item_queue.put(data)
                except Exception as e:
                    #  print (f'{e}\n\n')
                    time.sleep(1e-8)

    def __len__(self):
        return len(self.train_descriptions)

    def __getitem__(self, index):
        sample = self.item_queue.get()
        return sample, sample['dataset_index']

def clear_lines(n=2):
    """Clears a specified number of lines in the terminal."""
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

def format_text(text, width):
    return '\n'.join(textwrap.wrap(text, width=width))

current_state_dict = {}
current_state_dict["trained_model_path"]  = 'Test Path'

import signal
def create_graceful_exit(current_state_dict):
    def graceful_exit(signum, frame):
        exit_event.set()
        if model_loaded_event.is_set():
            ckpt_path = current_state_dict.pop("ckpt_path", 'default.pth')
            print(f'\nSaving current state to {ckpt_path}...')
            torch.save(current_state_dict, ckpt_path)
        exit(0)
    return graceful_exit
signal.signal(signal.SIGINT, create_graceful_exit(current_state_dict))

def exeption_handler(exctype, value, tb):
    exit_event.set()
    sys.__excepthook__(exctype, value, tb)
sys.excepthook = exeption_handler

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def main():
    global current_state_dict

    parser = argparse.ArgumentParser(description='Summ training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    # Optional arguments
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--device', type=int, default=1, help='Graphics card index (default: 2)')
    parser.add_argument('--acc', type=int, default=4, help='Gradient accumulation steps (int) (default: 4)')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to the pre-trained model state dict file)')
    parser.add_argument('--save', type=int, default=1000, help='Save model state dict each N steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--reset_stats', action='store_true', dest='reset_stats', default=False, help='Reset saved step and epoch')

    args = parser.parse_args()
    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')

    dataset = SummDataset(
        args.dataset_path
    )

    max_len = 0
    min_len = 100500
    min_file = ''

    for batch_idx in range(len(dataset)):
        sample, idx = dataset[batch_idx]
        
        summ = sample.get('summarized', 'Unable to get data')

        # if len(summ) != 3:
        #    print (sample['file_path'])

        prompt = summ[1]
        gt = summ[2]
        # gt_encoded = model.tokenizer.encode(gt, bos=False)

        if min_len > len(gt):
            min_len = len(gt)
            min_file = sample['file_path']

    print (min_len)
    print (min_file)

if __name__ == "__main__":
    main()
