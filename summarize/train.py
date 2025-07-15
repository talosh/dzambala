import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    def rescan(self):
        self.train_descriptions = self.find_json_files(self.data_root)
        self.initial_train_descriptions = list(self.train_descriptions)
        self.reshuffle()

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

    '''
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
    '''

    def read_items_thread(self, item_queue):
        last_seen_length = len(self.train_descriptions)
        idx = 0

        while not exit_event.is_set():
            current_length = len(self.train_descriptions)

            if current_length < last_seen_length:
                idx = 0  # restart from 0 if dataset shrank

            last_seen_length = current_length

            if current_length == 0:
                time.sleep(0.5)
                continue

            if idx >= current_length:
                idx = 0
                continue

            try:
                file_path = self.train_descriptions[idx]
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['dataset_index'] = idx
                data['file_path'] = file_path
                item_queue.put(data)
            except Exception as e:
                time.sleep(1e-8)

            idx += 1

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

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

import signal
def create_graceful_exit(current_state_dict):
    def graceful_exit(signum, frame):
        exit_event.set()
        # print (f'keys at exit: {current_state_dict.keys()}')
        if model_loaded_event.is_set():
            ckpt_path = current_state_dict["ckpt_path"]
            print(f'\nSaving current state to {ckpt_path}...')
            torch.save(current_state_dict, ckpt_path)
        exit(0)
    return graceful_exit

def exeption_handler(exctype, value, tb):
    exit_event.set()
    sys.__excepthook__(exctype, value, tb)
sys.excepthook = exeption_handler


def main():
    current_state_dict = {}

    signal.signal(signal.SIGINT, create_graceful_exit(current_state_dict))
    sys.excepthook = exeption_handler

    parser = argparse.ArgumentParser(description='Summ training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    # Optional arguments
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--decay', type=float, default=1e-3, help='Weight decay (default: 1e-3)')
    parser.add_argument('--device', type=int, default=1, help='Graphics card index (default: 2)')
    parser.add_argument('--acc', type=int, default=4, help='Gradient accumulation steps (int) (default: 4)')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to the pre-trained model state dict file)')
    parser.add_argument('--save', type=int, default=1000, help='Save model state dict each N steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--freeze', type=int, default=0, help='Freeze preset')
    # parser.add_argument('--freeze', action='store_true', dest='freeze', default=False, help='Reset saved step and epoch')
    parser.add_argument('--reset_stats', action='store_true', dest='reset_stats', default=False, help='Reset saved step and epoch')

    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')

    dataset = SummDataset(
        args.dataset_path
    )

    print (f'using device: {device}')

    # gemma related

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gemma_path = os.path.abspath(os.path.join(script_dir, "../private/llm/"))
    sys.path.insert(0, gemma_path)

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    from gemma import config
    from gemma import model as gemma_model

    model_config = config.get_model_config('1b')
    model_config.dtype = "float32"
    model_config.tokenizer = os.path.join(gemma_path, 'checkpoints',  'tokenizer.model')

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    ts = time.time()
    with _set_default_tensor_type(model_config.get_dtype()):
        print ('Loading model...', end='', flush=True)
        model = gemma_model.GemmaForCausalLM(model_config).to(device).train()
        print (f' {(time.time()-ts):.2f}s  Loading weights...', end='', flush=True)
        current_state_dict.clear()
        current_state_dict.update(
            torch.load(args.ckpt, mmap=True, weights_only=True, map_location=device)
        )
        current_state_dict['ckpt_path'] = args.ckpt
        model.load_state_dict(current_state_dict['model_state_dict'], strict=False)
        print (f' Done in {(time.time()-ts):.2f}s', flush=True)
        model_loaded_event.set()
    warnings.resetwarnings()


    if args.freeze:
        print ('\nFreezing parameters')

        if args.freeze == 1:
            # version 001
            for i in range(25):
                for name, param in model.named_parameters():
                    if f'layers.{i}.' in name:
                        param.requires_grad = False
        elif args.freeze == 2:
            # version 002
            for i in range(21):
                for name, param in model.named_parameters():
                    if f'layers.{i}.' in name:
                        param.requires_grad = False
            for name, param in model.named_parameters():
                if 'layers.21' in name:
                    param.requires_grad = True
            for name, param in model.named_parameters():
                if 'layers.22' in name:
                    param.requires_grad = True
            for name, param in model.named_parameters():
                if 'layers.23' in name:
                    param.requires_grad = True
            for name, param in model.named_parameters():
                if 'layers.24' in name:
                    param.requires_grad = True
            for name, param in model.named_parameters():
                if 'layers.25' in name:
                    param.requires_grad = True

        elif args.freeze == 3:
            # version 003
            for i in range(24):
                for name, param in model.named_parameters():
                    if f'layers.{i}.' in name:
                        param.requires_grad = False
            for name, param in model.named_parameters():
                if 'layers.25' in name:
                    param.requires_grad = True

        print ('\nUn-freezing parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)


    def write_model_state(write_model_state_queue):
        while not exit_event.is_set():
            try:
                current_state_dict = write_model_state_queue.get_nowait()
                ckpt_path = current_state_dict['ckpt_path']
                if os.path.isfile(ckpt_path):
                    backup_file = ckpt_path.replace('.pth', '.backup.pth')
                    shutil.copy(ckpt_path, backup_file)
                torch.save(current_state_dict, ckpt_path)
            except:
                time.sleep(1e-2)

    write_model_state_queue = queue.Queue(maxsize=2)
    write_model_state_thread = threading.Thread(target=write_model_state, args=(write_model_state_queue, ))
    write_model_state_thread.daemon = True
    write_model_state_thread.start()

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_id)

    start_timestamp = time.time()
    epoch = 0 if args.reset_stats == -1 else 0 # loaded_epoch
    step = 0 if args.reset_stats == -1 else 0 # loaded_step
    batch_idx = 0
    avg_loss = 0

    '''
    max_len = 0
    min_len = 100500
    min_file = ''

    for batch_idx in range(len(dataset)):
        sample, idx = dataset[batch_idx]
        
        summ = sample.get('summarized', 'Unable to get data')

        if len(summ) != 3:
            print (sample['file_path'])

        prompt = summ[1]
        gt = summ[0]
        # gt_encoded = model.tokenizer.encode(gt, bos=False)

        if min_len > len(gt):
            min_len = len(gt)
            min_file = sample['file_path']

    print (min_len)
    print (min_file)
    '''

    print ('\n')
    lines_to_clear = 0
    max_len = 0


    while True:
        time_stamp = time.time()
        info = ''
        sample, idx = dataset[batch_idx]
        
        step = step + 1
        batch_idx = batch_idx +1

        if 'prompt' in sample:
            prompt = sample['prompt']
            gt = sample['summ']
        else:
            # news first attempt data
            summ = sample.get('summarized', 'Unable to get data')

            prompt = summ[1]
            gt = summ[0]

            if random.uniform(0, 1) < 0.5:
                try:
                    prompt = summ[2]
                    # if random.uniform(0, 1) < 0.5:
                    #   gt = summ[1]
                except:
                    pass
            
            if random.uniform(0, 1) < 0.5:
                prompt_orig = sample.get('text_body', summ[2])
                prompt_orig_encoded = model.tokenizer.encode(prompt_orig)
                if len(prompt_orig_encoded) < 384:
                    prompt = prompt_orig

        if isinstance(gt, list) and len(gt) > 0:
            gt = gt[0]

        prompt = str(prompt)
        gt = str(gt)

        gt_encoded = model.tokenizer.encode(gt, bos=False)

        data_time = time.time() - time_stamp
        time_stamp = time.time()

        # prompt_encoded = model.tokenizer.encode(prompt)
        # max_len = max(len(prompt_encoded), max_len)
        # info += f'prompt length: {len(prompt_encoded)}, max: {max_len}\n'

        try:
            result = model.generate(
                prompt, 
                device,
                temperature=0.02,   # <- greedy decoding
                # top_k=0,
                # top_p=1.0,
                output_len=len(gt_encoded) # FLAGS.output_len
            )

            if not result['all_logits']:
                # print (sample['file_path'])
                continue

            model_time = time.time() - time_stamp
            time_stamp = time.time()
            
            '''
            logits_seq = torch.stack(result['all_logits'][:1], dim=0)
            logits_flat = logits_seq.permute(1, 0, 2).reshape(-1, logits_seq.size(-1))
            target_flat = torch.tensor(gt_encoded[:1], dtype=torch.long).to(device)
            target_flat = target_flat.reshape(-1)
            loss_first = loss_fn(logits_flat, target_flat)

            logits_seq = torch.stack(result['all_logits'][:4], dim=0)
            logits_flat = logits_seq.permute(1, 0, 2).reshape(-1, logits_seq.size(-1))
            target_flat = torch.tensor(gt_encoded[:4], dtype=torch.long).to(device)
            target_flat = target_flat.reshape(-1)
            loss_first4 = loss_fn(logits_flat, target_flat)
            '''

            logits_seq = torch.stack(result['all_logits'], dim=0)
            logits_flat = logits_seq.permute(1, 0, 2).reshape(-1, logits_seq.size(-1))
            target_flat = torch.tensor(gt_encoded, dtype=torch.long).to(device)
            target_flat = target_flat.reshape(-1)
            loss = loss_fn(logits_flat, target_flat) # + 0.4 * loss_first + 0.2 * loss_first4

            '''
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            current_state_dict['model_state_dict'] = model.state_dict()
            current_state_dict['ckpt_path'] = args.ckpt
            '''

            (loss / args.acc).backward()
            if batch_idx % args.acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                current_state_dict['model_state_dict'] = model.state_dict()
                current_state_dict['ckpt_path'] = args.ckpt

        except torch.cuda.OutOfMemoryError:
            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()
            continue

        # torch.cuda.synchronize(device=device)
        train_time = time.time() - time_stamp
        time_stamp = time.time()

        avg_loss = float(loss.item()) if batch_idx == 0 else (avg_loss * (batch_idx - 1) + float(loss.item())) / batch_idx

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        # if step % 1000 == 1:
        # torch.cuda.empty_cache()

        if step % args.save == 1:
            dict_to_write = {
                'model_state_dict': deepcopy(model.state_dict()),
                'ckpt_path': args.ckpt
            }
            write_model_state_queue.put(dict_to_write)

            dataset.rescan()

            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()

        # current_state_dict['step'] = step
        # time.sleep(0.5)

        tail_time = time.time() - time_stamp
        clear_lines(lines_to_clear + 1)
        info += f'[Epoch {(epoch + 1):04} Step {step} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time:.2f}+{model_time:.2f}+{train_time:.2f}+{tail_time:.2f}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}]'
        info += f' Loss: {loss.item():.4f} Avg: {avg_loss:.4f}'
        info += f'\n\n'
        # info += f'Prompt: {prompt}\n\n'
        info += f'T: {format_text(gt, 100)}\n'
        info += f'R: {format_text(result["result"], 100).strip()}\n\n'
        info += f'P: {format_text(prompt, 100)}\n'
        print (f'\r{info}')
        lines_to_clear = len(info.splitlines())

        if ( idx + 1 ) == len(dataset):
            clear_lines(lines_to_clear + 1)
            lines_to_clear = 0
            print (f'[Epoch {(epoch + 1):04} Step {step} - {days:02}d {hours:02}:{minutes:02}] Avg: {avg_loss:.4f}\n')
            avg_loss = 0
            epoch = epoch + 1
            batch_idx = 0

            dataset.rescan()

            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()


    '''

    def check_range_percent(value):
        ivalue = int(value)
        if ivalue < 0 or ivalue > 100:
            raise argparse.ArgumentTypeError(f"Percent must be between 0 and 100, got value={ivalue}")
        return ivalue

    parser = argparse.ArgumentParser(description="A command-line app that takes source, target, and optional learning rate.")
    parser.add_argument("source", type=str, help="Path to the source file")
    parser.add_argument("target", type=str, help="Path to the target file")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")

    args = parser.parse_args()


    cdl = CDL().to(device)
    loss_l1 = nn.L1Loss()

    optimizer = optim.AdamW(cdl.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    
    batch = 0

    current_state_dict['step'] = int(step)
    current_state_dict['epoch'] = int(epoch)
    current_state_dict['start_timestamp'] = start_timestamp
    current_state_dict['lr'] = optimizer_flownet.param_groups[0]['lr']
    current_state_dict['model_info'] = model_info
    if args.all_gpus:
        current_state_dict['flownet_state_dict'] = convert_from_data_parallel(flownet.state_dict())
    else:
        current_state_dict['flownet_state_dict'] = flownet.state_dict()
    current_state_dict['optimizer_flownet_state_dict'] = optimizer_flownet.state_dict()
    current_state_dict['trained_model_path'] = trained_model_path

    cdl.train()

    '''

if __name__ == "__main__":
    main()

