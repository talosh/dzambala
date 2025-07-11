import os
import sys
import random
import shutil
import struct
import ctypes
import argparse
import importlib
import queue
import threading
import time
import platform
from copy import deepcopy

from pprint import pprint

try:
    import numpy as np
    import torch
except:
    python_executable_path = sys.executable
    if '.miniconda' in python_executable_path:
        print ('Unable to import Numpy and PyTorch libraries')
        print (f'Using {python_executable_path} python interpreter')
        sys.exit()
    else:
        # make Flame happy on hooks scan
        class torch(object):
            class nn(object):
                class Module(object):
                    pass
                class Conv2d(object):
                    pass

def read_image_file(file_path, header_only = False):
    import OpenImageIO as oiio

    result = {'spec': None, 'image_data': None}

    inp = oiio.ImageInput.open(file_path)
    if inp :
        spec = inp.spec()
        result['spec'] = spec
        if not header_only:
            height = spec.height
            width = spec.width
            channels = spec.nchannels
            result['image_data'] = inp.read_image(0, 0, 0, channels)
        inp.close()
    return result

def write_exr(image_data, filename, half_float = False, pixelAspectRatio = 1.0):
    import struct
    import numpy as np

    if image_data.dtype == np.float16:
        half_float = True

    height, width, depth = image_data.shape
    red = image_data[:, :, 0]
    green = image_data[:, :, 1]
    blue = image_data[:, :, 2]
    if depth > 3:
        alpha = image_data[:, :, 3]
    else:
        alpha = np.array([])

    channels_list = ['B', 'G', 'R'] if not alpha.size else ['A', 'B', 'G', 'R']

    MAGIC = 20000630
    VERSION = 2
    UINT = 0
    HALF = 1
    FLOAT = 2

    def write_attr(f, name, type, value):
        f.write(name.encode('utf-8') + b'\x00')
        f.write(type.encode('utf-8') + b'\x00')
        f.write(struct.pack('<I', len(value)))
        f.write(value)

    def get_channels_attr(channels_list):
        channel_list = b''
        for channel_name in channels_list:
            name_padded = channel_name[:254] + '\x00'
            bit_depth = 1 if half_float else 2
            pLinear = 0
            reserved = (0, 0, 0)  # replace with your values if needed
            xSampling = 1  # replace with your value
            ySampling = 1  # replace with your value
            channel_list += struct.pack(
                f"<{len(name_padded)}s i B 3B 2i",
                name_padded.encode(), 
                bit_depth, 
                pLinear, 
                *reserved, 
                xSampling, 
                ySampling
                )
        channel_list += struct.pack('c', b'\x00')

            # channel_list += (f'{i}\x00').encode('utf-8')
            # channel_list += struct.pack("<i4B", HALF, 1, 1, 0, 0)
        return channel_list
    
    def get_box2i_attr(x_min, y_min, x_max, y_max):
        return struct.pack('<iiii', x_min, y_min, x_max, y_max)

    with open(filename, 'wb') as f:
        # Magic number and version field
        f.write(struct.pack('I', 20000630))  # Magic number
        f.write(struct.pack('H', 2))  # Version field
        f.write(struct.pack('H', 0))  # Version field
        write_attr(f, 'channels', 'chlist', get_channels_attr(channels_list))
        write_attr(f, 'compression', 'compression', b'\x00')  # no compression
        write_attr(f, 'dataWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
        write_attr(f, 'displayWindow', 'box2i', get_box2i_attr(0, 0, width - 1, height - 1))
        write_attr(f, 'lineOrder', 'lineOrder', b'\x00')  # increasing Y
        write_attr(f, 'pixelAspectRatio', 'float', struct.pack('<f', pixelAspectRatio))
        write_attr(f, 'screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
        write_attr(f, 'screenWindowWidth', 'float', struct.pack('<f', 1.0))
        f.write(b'\x00')  # end of header

        # Scan line offset table size and position
        line_offset_pos = f.tell()
        pixel_data_start = line_offset_pos + 8 * height
        bytes_per_channel = 2 if half_float else 4
        # each scan line starts with 4 bytes for y coord and 4 bytes for pixel data size
        bytes_per_scan_line = width * len(channels_list) * bytes_per_channel + 8 

        for y in range(height):
            f.write(struct.pack('<Q', pixel_data_start + y * bytes_per_scan_line))

        channel_data = {'R': red, 'G': green, 'B': blue, 'A': alpha}

        # Pixel data
        for y in range(height):
            f.write(struct.pack('I', y))  # Line number
            f.write(struct.pack('I', bytes_per_channel * len(channels_list) * width))  # Pixel data size
            for channel in sorted(channels_list):
                f.write(channel_data[channel][y].tobytes())
        f.close

    del image_data, red, green, blue

def get_dataset(
        data_root,
        batch_size = 4, 
        frame_size = 448,
        repeat = 1,
        ):

    class MLDataset(torch.utils.data.Dataset):
        def __init__(   
                self, 
                data_root,
                batch_size = 4,
                frame_size = 448,
                repeat = 1
                ):
            
            self.data_root = data_root
            self.batch_size = batch_size
            self.h = frame_size
            self.w = frame_size

            source_folder_path = os.path.abspath(os.path.join(data_root, 'source'))

            if not os.path.isdir(source_folder_path):
                print (f'Error: No source folder in {data_root}')
                sys.exit()

            source_exr_files = [os.path.join(source_folder_path, file) for file in os.listdir(source_folder_path) if file.endswith('.exr')]

            source_exr_files = sorted(source_exr_files)
            print (f'Found {len(source_exr_files)} files')

            self.train_descriptions = []

            for idx in range(len(source_exr_files)):
                description = {
                    'source': source_exr_files[idx],
                }
                self.train_descriptions.append(description)

            self.initial_train_descriptions = list(self.train_descriptions)

            print ('\nReshuffling training data indices...')

            self.reshuffle()

            self.frames_queue = queue.Queue(maxsize=8)
            self.frame_read_thread = threading.Thread(target=self.read_frames_thread)
            self.frame_read_thread.daemon = True
            self.frame_read_thread.start()

            # print ('reading first block of training data...')
            self.last_train_data = [self.frames_queue.get()]
            self.last_train_data_size = 8
            self.new_sample_shown = False
            self.train_data_index = 0

            self.current_batch_data = []

            self.repeat_count = repeat
            self.repeat_counter = 0

        def reshuffle(self):
            random.shuffle(self.train_descriptions)
            
        def read_frames_thread(self):
            timeout = 1e-8
            while True:
                for index in range(len(self.train_descriptions)):
                    description = self.train_descriptions[index]
                    try:
                        train_data = {}
                        train_data['source'] = read_image_file(description['source'])['image_data']
                        train_data['description'] = description
                        train_data['index'] = index
                        self.frames_queue.put(train_data)
                    except Exception as e:
                        del train_data
                        print (e)
                        time.sleep(timeout)

        def __len__(self):
            return len(self.train_descriptions)
        
        def crop(self, img0, img1, crop_h, crop_w):
            """
            Randomly crops a region of size (crop_h, crop_w) from img0 and img1.
            
            Args:
                img0, img1 (torch.Tensor): Input images of shape (c, h, w).
                crop_h, crop_w (int): The desired crop height and width.
            
            Returns:
                torch.Tensor: Cropped regions from img0 and img1 of shape (c, crop_h, crop_w).
            """
            # Ensure seed is random
            np.random.seed(None)
            
            # Get the original height and width of the images
            _, orig_h, orig_w = img0.shape
            
            # Ensure the crop fits within the original dimensions
            if crop_h > orig_h or crop_w > orig_w:
                raise ValueError(f"Crop size ({crop_h}, {crop_w}) is larger than the image dimensions ({orig_h}, {orig_w}).")

            # Randomly select top-left corner for the crop
            x = np.random.randint(0, orig_h - crop_h + 1)
            y = np.random.randint(0, orig_w - crop_w + 1)

            # print (f'img0: {img0.shape}')
            # print (f'img1: {img1.shape}')

            # Crop both images (slice along height and width dimensions)
            img0_cropped = img0[:, x:x + crop_h, y:y + crop_w]
            img1_cropped = img1[:, x:x + crop_h, y:y + crop_w]

            return img0_cropped, img1_cropped

        def getimg(self, index):
            if self.repeat_counter >= self.repeat_count:
                self.repeat_counter = 1
                try:
                    new_data = self.frames_queue.get_nowait()
                    self.last_train_data[random.randint(0, len(self.last_train_data) - 1)] = new_data
                    self.train_data_index = new_data['index']
                    return new_data
                except queue.Empty:
                    return random.choice(self.last_train_data)
            else:
                self.repeat_counter += 1
                return random.choice(self.last_train_data)

        def __getitem__(self, index):
            train_data = self.getimg(index)
            images_idx = self.train_data_index
            src_img0 = torch.from_numpy(train_data['source']).permute(2, 0, 1)

            _, h, w = src_img0.shape
            src_img1 = torch.nn.functional.interpolate(src_img0.unsqueeze(0), scale_factor = 1/3.2, mode='bicubic', align_corners=True, antialias=True)
            src_img1 = torch.nn.functional.interpolate(src_img1, size = (h, w), mode='bicubic', align_corners=True, antialias=True)[0]

            '''
            if random.uniform(0, 1) > 0.5:
                src_img0 = torch.nn.functional.interpolate(src_img0.unsqueeze(0), scale_factor = 1/1.2, mode='bilinear', align_corners=False)[0]
                src_img1 = torch.nn.functional.interpolate(src_img1.unsqueeze(0), scale_factor = 1/1.2, mode='bilinear', align_corners=False)[0]
            elif random.uniform(0, 1) > 0.5:
                src_img0 = torch.nn.functional.interpolate(src_img0.unsqueeze(0), scale_factor = 1/1.4, mode='bilinear', align_corners=False)[0]
                src_img1 = torch.nn.functional.interpolate(src_img1.unsqueeze(0), scale_factor = 1/1.4, mode='bilinear', align_corners=False)[0]

            c, h, w = src_img0.shape

            horizontal = torch.linspace(-1, 1, w).view(1, w)
            horizontal = horizontal.expand(h, w)
            vertical = torch.linspace(-1, 1, h).view(h, 1)
            vertical = vertical.expand(h, w)
            grid = torch.stack((horizontal, vertical), dim=0)
            src_img0 = torch.cat((src_img0, grid), 0)
            rel_diff = (torch.abs(src_img0 - src_img1) / torch.abs(src_img1) + 1e-8).mean()
            jitter_seed = torch.tanh(torch.ones_like(src_img0[:1, :, :]) * 0 + rel_diff)
            src_img0 = torch.cat((src_img0, jitter_seed), 0)
            '''

            def horizontal_flip(img):
                return img.flip(dims=[2])

            def vertical_flip(img):
                return img.flip(dims=[1])

            def rotate_90(img):
                return img.permute(0, 2, 1).flip(dims=[1])

            def rotate_minus_90(img):
                return img.permute(0, 2, 1).flip(dims=[2])

            def rotate_180(img):
                return img.flip(dims=[1, 2])

            batch_img0 = []
            batch_img1 = []

            for index in range(self.batch_size):

                img0, img1 = self.crop(src_img0, src_img1, self.h, self.w)

                if random.uniform(0, 1) > 0.5:
                    img0 = horizontal_flip(img0)
                    img1 = horizontal_flip(img1)

                if random.uniform(0, 1) > 0.5:
                    img0 = vertical_flip(img0)
                    img1 = vertical_flip(img1)

                '''
                if random.uniform(0, 1) > 0.5:
                    img0 = rotate_90(img0)
                    img1 = rotate_90(img1)

                if random.uniform(0, 1) > 0.5:
                    img0 = rotate_minus_90(img0)
                    img1 = rotate_minus_90(img1)
                '''

                if random.uniform(0, 1) > 0.5:
                    img0 = rotate_180(img0)
                    img1 = rotate_180(img1)

                batch_img0.append(img0)
                batch_img1.append(img1)

            return torch.stack(batch_img0), torch.stack(batch_img1), images_idx

    return MLDataset(
        data_root,
        batch_size = batch_size,
        frame_size = frame_size, 
        repeat = repeat,
        )

def clear_lines(n=2):
    """Clears a specified number of lines in the terminal."""
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

def create_timestamp_uid():
    import random
    import uuid
    from datetime import datetime

    def number_to_letter(number):
        # Map each digit to a letter
        mapping = {
            '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
            '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'
        }
        return ''.join(mapping.get(char, char) for char in number)

    uid = ((str(uuid.uuid4()).replace('-', '')).upper())
    uid = ''.join(random.sample(number_to_letter(uid), 4))
    timestamp = (datetime.now()).strftime('%Y%b%d_%H%M').upper()
    return f'{timestamp}_{uid}'

def find_and_import_model(models_dir='models', base_name=None, model_name=None, model_file=None):
    """
    Dynamically imports the latest version of a model based on the base name,
    or a specific model if the model name/version is given, and returns the Model
    object named after the base model name.

    :param models_dir: Relative path to the models directory.
    :param base_name: Base name of the model to search for.
    :param model_name: Specific name/version of the model (optional).
    :return: Imported Model object or None if not found.
    """

    import os
    import re
    import importlib

    if model_file:
        module_name = model_file[:-3]  # Remove '.py' from filename to get module name
        module_path = f"models.{module_name}"
        module = importlib.import_module(module_path)
        model_object = getattr(module, 'Model')
        return model_object

    # Resolve the absolute path of the models directory
    models_abs_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            models_dir
        )
    )

    # List all files in the models directory
    try:
        files = os.listdir(models_abs_path)
    except FileNotFoundError:
        print(f"Directory not found: {models_abs_path}")
        return None

    # pprint (files)

    # Filter files based on base_name or model_name
    if model_name:
        # Look for a specific model version
        filtered_files = [f for f in files if f == f"{model_name}.py"]
    else:
        # Find all versions of the model and select the latest one
        # regex_pattern = fr"{base_name}_v(\d+)\.py"
        # versions = [(f, int(m.group(1))) for f in files if (m := re.match(regex_pattern, f))]
        versions = [f for f in files if f.endswith('.py')]
        if versions:
            # Sort by version number (second item in tuple) and select the latest one
            # latest_version_file = sorted(versions, key=lambda x: x[1], reverse=True)[0][0]
            latest_version_file = sorted(versions, reverse=True)[0]
            filtered_files = [latest_version_file]

    # Import the module and return the Model object
    if filtered_files:
        module_name = filtered_files[0][:-3]  # Remove '.py' from filename to get module name
        module_path = f"models.{module_name}"
        module = importlib.import_module(module_path)
        model_object = getattr(module, 'Model')
        return model_object
    else:
        print(f"Model not found: {base_name or model_name}")
        return None

def create_csv_file(file_name, fieldnames):
    import csv
    """
    Creates a CSV file with the specified field names as headers.
    """
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def append_row_to_csv(file_name, row):
    import csv
    """
    Appends a single row to an existing CSV file.
    """
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        writer.writerow(row)

def sinusoidal_scale_fn(x):
    import math
    # x is a fraction of the cycle's progress (0 to 1)
    return 0.5 * (1 + math.sin(math.pi * (x - 0.5)))

def scale_and_soft_clamp(tensor, min_val=0.69, max_val=1.0, clamp_limit=-0.1, roll_off_slope=0.1):
    """
    Scale values in the range [min_val, max_val] to [0, 1] and apply a soft clamp
    to negative values to prevent them from going below `clamp_limit`.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        min_val (float): Minimum value of the range to scale from (default: 0.9).
        max_val (float): Maximum value of the range to scale from (default: 1.0).
        clamp_limit (float): Minimum value that the output can reach (default: -0.1).
        roll_off_slope (float): Slope for roll-off on negative values (default: 0.1).
        
    Returns:
        torch.Tensor: Transformed tensor.
    """
    # Step 1: Scale values in the range [min_val, max_val] to [0, 1]
    scaled_tensor = (tensor - min_val) / (max_val - min_val)
    scaled_tensor = torch.clamp(scaled_tensor, min = 0, max=1)
    soft_clamped_tensor = torch.nn.functional.elu(scaled_tensor, alpha = roll_off_slope)

    # Step 2: Apply soft clamping to avoid values going below `clamp_limit`
    # soft_clamped_tensor = torch.where(scaled_tensor >= 0, scaled_tensor, roll_off_slope * scaled_tensor)

    
    # Shift the minimum of soft-clamped values to the specified `clamp_limit`
    # soft_clamped_tensor = torch.maximum(soft_clamped_tensor, torch.tensor(clamp_limit, device=tensor.device))

    return soft_clamped_tensor

class LapLoss(torch.nn.Module):
    def gauss_kernel(self, size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        # kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return torch.nn.functional.interpolate(x, scale_factor= 1. / 2, mode="bilinear", align_corners=False)
        # return x[:, :, ::2, ::2]

    def upsample(self, x):
        return torch.nn.functional.interpolate(x, scale_factor= 2, mode="bilinear", align_corners=False)
        device = x.device
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        x_up = cc.permute(0,1,3,2)
        gauss_kernel = self.gauss_kernel(channels=x.shape[1])
        gauss_kernel = gauss_kernel.to(device)
        return self.conv_gauss(x_up, 4*gauss_kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def laplacian_pyramid(self, img, kernel, max_levels=3):
        current = img
        pyr = []
        for level in range(max_levels):
            filtered = self.conv_gauss(current, kernel)
            n, c, h, w = filtered.shape
            sh, sw = round(h * (1 / 2)), round(w * (1 / 2))
            down = torch.nn.functional.interpolate(filtered, size=(sh, sw), mode="bilinear", align_corners=False)
            up = torch.nn.functional.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
            diff = current-up
            pyr.append(diff)
            current = down
        return pyr

    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.maxdepth = 4 * max_levels
        self.gk = self.gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        '''
        n, c, sh, sw = input.shape
        ph = self.maxdepth - (sh % self.maxdepth)
        pw = self.maxdepth - (sw % self.maxdepth)
        padding = (0, pw, 0, ph)
        input = torch.nn.functional.pad(input, padding)
        target = torch.nn.functional.pad(target, padding)
        '''
        
        self.gk = self.gk.to(device = input.device)
        pyr_input  = self.laplacian_pyramid(img=input, kernel=self.gk, max_levels=self.max_levels)
        pyr_target = self.laplacian_pyramid(img=target, kernel=self.gk, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def rgb_to_yuv(rgb_tensor):
    transformation_matrix = torch.tensor([
        [0.299,  0.587,  0.114],   # Y
        [-0.147, -0.289,  0.436],  # U
        [0.615, -0.515, -0.100]    # V
    ]).T.to(device = rgb_tensor.device, dtype=rgb_tensor.dtype)  # Transpose to match (3, 3)
    rgb_tensor = rgb_tensor.permute(0, 2, 3, 1)  # (n, h, w, 3)
    yuv_tensor = torch.tensordot(rgb_tensor, transformation_matrix, dims=([-1], [0]))
    return yuv_tensor.permute(0, 3, 1, 2)

current_state_dict = {}

def main():
    global current_state_dict
    parser = argparse.ArgumentParser(description='Training script.')

    def check_range_percent(value):
        ivalue = int(value)
        if ivalue < 0 or ivalue > 100:
            raise argparse.ArgumentTypeError(f"Percent must be between 0 and 100, got value={ivalue}")
        return ivalue

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    # Optional arguments
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--preview', type=int, default=100, help='Save preview each N steps (default: 100)')
    parser.add_argument('--save', type=int, default=100, help='Save model state dict each N steps (default: 100)')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat each triade N times with augmentation (default: 1)')
    parser.add_argument('--batch', type=int, default=2, help='Repeat each triade N times with augmentation (default: 2)')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--frame_size', type=int, default=448, help='Frame size in pixels (default: 448)')
    parser.add_argument('--pulse', type=float, default=10000, help='Period in steps to pulse learning rate (float) (default: 10K)')
    parser.add_argument('--pulse_amplitude', type=float, default=25, help='Learning rate pulse amplitude (percentage) (default: 25)')
    parser.add_argument('--state_file', type=str, default=None, help='Path to the pre-trained model state dict file (optional)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--generalize', type=check_range_percent, default=85, help='Generalization level (0 - 100) (default: 85)')
    parser.add_argument('--reset_stats', action='store_true', dest='reset_stats', default=False, help='Reset saved step, epoch and loss stats')
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False, help='Freeze custom parameters (edit code to select)')
    parser.add_argument('--epochs', type=int, default=-1, help='Number of epoch to run (int) (default: Unlimited)')
    parser.add_argument('--onecycle', type=int, default=-1, help='Number of steps for OneCycle schedule (default: steps for 10 epochs)')
    parser.add_argument('--first_epoch', type=int, default=-1, help='Epoch (int) (default: Saved)')

    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')
    frame_size = args.frame_size

    Net = None

    if args.model:
        model_name = args.model
        Net = find_and_import_model(model_name=model_name)            
    else:
        # Find and initialize model
        if args.state_file and os.path.isfile(args.state_file):
            trained_model_path = args.state_file
            try:
                checkpoint = torch.load(trained_model_path, map_location=device, weights_only=False)
                print('loaded previously saved model checkpoint')
            except Exception as e:
                print (f'unable to load saved model checkpoint: {e}')
                sys.exit()

            model_info = checkpoint.get('model_info')
            model_file = model_info.get('file')
            Net = find_and_import_model(model_file=model_file)
        else:
            if not args.state_file:
                print ('Please specify either model name or model state file')
                return
            if not os.path.isfile(args.state_file):
                print (f'Model state file {args.state_file} does not exist and "--model" flag is not set to start from scratch')
                return

    if Net is None:
        print (f'Unable to load model {args.model}')
        return
    
    model_info = Net.get_info()
    print ('Model info:')
    pprint (model_info)

    net = Net().get_training_model()().to(device)

    print ('\n-----')
    print (f'Creating dataset:')
    print (f'dataset_path: {args.dataset_path}')
    print (f'frame_size: {frame_size}')
    print (f'repeat: {args.repeat}')
    print ('-----\n')

    dataset = get_dataset(
        args.dataset_path,
        batch_size = args.batch,
        frame_size=frame_size,
        repeat=args.repeat
        )

    if not os.path.isdir(os.path.join(args.dataset_path, 'preview')):
        os.makedirs(os.path.join(args.dataset_path, 'preview'))

    def write_images(write_image_queue):
        while True:
            try:
                write_data = write_image_queue.get_nowait()
                preview_index = write_data.get('preview_index', 0)
                preview_folder = write_data["preview_folder"]
                if not os.path.isdir(preview_folder):
                    os.makedirs(preview_folder)
                write_exr(write_data['sample_source'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_A_source.exr'), half_float = True)
                write_exr(write_data['sample_result'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_B_result.exr'), half_float = True)
                write_exr(write_data['sample_target'].astype(np.float16), os.path.join(preview_folder, f'{preview_index:02}_C_target.exr'), half_float = True)
                del write_data
            except:
            # except queue.Empty:
                time.sleep(1e-2)

    def write_model_state(write_model_state_queue):
        while True:
            try:
                current_state_dict = write_model_state_queue.get_nowait()
                trained_model_path = current_state_dict['trained_model_path']
                if os.path.isfile(trained_model_path):
                    backup_file = trained_model_path.replace('.pth', '.backup.pth')
                    shutil.copy(trained_model_path, backup_file)
                torch.save(current_state_dict, current_state_dict['trained_model_path'])
            except:
                time.sleep(1e-2)

    write_image_queue = queue.Queue(maxsize=16)
    write_thread = threading.Thread(target=write_images, args=(write_image_queue, ))
    write_thread.daemon = True
    write_thread.start()

    write_model_state_queue = queue.Queue(maxsize=2)
    write_model_state_thread = threading.Thread(target=write_model_state, args=(write_model_state_queue, ))
    write_model_state_thread.daemon = True
    write_model_state_thread.start()

    pulse_dive = args.pulse_amplitude
    pulse_period = args.pulse
    lr = args.lr

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    criterion_huber = torch.nn.HuberLoss(delta=0.001)
    criterion_lap = LapLoss()

    step = 0
    loaded_step = 0
    current_epoch = 0
    preview_index = 0

    weight_decay = 10 ** (-2 - 0.02 * (args.generalize - 1)) if args.generalize > 1 else 1e-4
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    if args.state_file:
        trained_model_path = args.state_file
        try:
            checkpoint = torch.load(trained_model_path, map_location=device, weights_only=False)
            print('loaded previously saved model checkpoint')
        except Exception as e:
            print (f'unable to load saved model: {e}')

        # net_state_dict = checkpoint['net_state_dict']
        # net_state_dict['lut'] = net_state_dict.pop('block0.lut')
        # checkpoint['net_state_dict'] = net_state_dict

        try:
            missing_keys, unexpected_keys = net.load_state_dict(checkpoint['net_state_dict'], strict=False)
            print('loaded previously saved network state')
            if missing_keys:
                print (f'\nMissing keys:\n{missing_keys}\n')
            if unexpected_keys:
                print (f'\nUnexpected keys:\n{unexpected_keys}\n')
        except Exception as e:
            print (f'unable to load network state: {e}')

        try:
            loaded_step = checkpoint['step']
            print (f'loaded step: {loaded_step}')
            current_epoch = checkpoint['epoch']
            print (f'epoch: {current_epoch + 1}')
        except Exception as e:
            print (f'unable to set step and epoch: {e}')

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded previously saved optimizer state')
            optimizer.param_groups[0]["lr"] = lr
        except Exception as e:
            print (f'unable to load optimizer state: {e}')

    else:
        traned_model_name = f'{model_info.get("name")}_{create_timestamp_uid()}.pth'
        trained_model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'trained_models'
        )
        if not os.path.isdir(trained_model_dir):
            os.makedirs(trained_model_dir)
        trained_model_path = os.path.join(trained_model_dir, traned_model_name)

    print (f'model state dict: {trained_model_path}')

    if args.reset_stats:
        step = 0
        loaded_step = 0
        current_epoch = 0
        preview_index = 0
        steps_loss = []
        epoch_loss = []
        psnr_list = []
        lpips_list = []

        weight_decay = 10 ** (-2 - 0.02 * (args.generalize - 1)) if args.generalize > 1 else 1e-4
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    if args.onecycle != -1:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            div_factor = 25,
            final_div_factor = 4,
            total_steps = 2 * len(dataset)*dataset.repeat_count if args.onecycle == -1 else args.onecycle * len(dataset)*dataset.repeat_count,
            last_epoch = -1 if loaded_step == 0 else loaded_step
            )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    start_timestamp = time.time()
    time_stamp = time.time()
    epoch = current_epoch if args.first_epoch == -1 else args.first_epoch
    step = loaded_step if args.first_epoch == -1 else step
    batch_idx = 0

    current_state_dict['step'] = int(step)
    current_state_dict['epoch'] = int(epoch)
    current_state_dict['start_timestamp'] = start_timestamp
    current_state_dict['lr'] = optimizer.param_groups[0]['lr']
    current_state_dict['model_info'] = model_info
    current_state_dict['net_state_dict'] = net.state_dict()
    current_state_dict['optimizer_state_dict'] = optimizer.state_dict()
    current_state_dict['trained_model_path'] = trained_model_path

    create_csv_file(
        f'{os.path.splitext(trained_model_path)[0]}.csv',
        [
            'Epoch',
            'Step',
            'Avg',
        ]
    )

    create_csv_file(
        f'{os.path.splitext(trained_model_path)[0]}.eval.csv',
        [
            'Epoch',
            'Step',
            'Avg',
        ]
    )

    import signal
    def create_graceful_exit(current_state_dict):
        def graceful_exit(signum, frame):
            print(f'\nSaving current state to {current_state_dict["trained_model_path"]}...')
            torch.save(current_state_dict, current_state_dict['trained_model_path'])
            exit(0)
        return graceful_exit
    signal.signal(signal.SIGINT, create_graceful_exit(current_state_dict))

    min_l1 = float(sys.float_info.max)
    avg_l1 = 0
    max_l1 = 0
    avg_loss = 0

    data_time = 0
    data_time1 = 0
    data_time2 = 0
    train_time = 0

    # LPIPS Init
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    import lpips
    os.environ['TORCH_HOME'] = os.path.abspath(os.path.dirname(__file__))
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.to(device)
    warnings.resetwarnings()

    if args.freeze:
        for param in net.lut.parameters():
            param.requires_grad = False
        '''
        for param in net.block0.conv0.parameters():
            param.requires_grad = False
        for param in net.block0.convblock0.parameters():
            param.requires_grad = False
        for param in net.block0.lastconv.parameters():
            param.requires_grad = False
        '''

        # for param in net.block0.inp_lut.parameters():
        #    param.requires_grad = False

        print ('\n---\nfreezing:')
        for name, param in net.named_parameters():
            if not param.requires_grad:
                print(name, param.requires_grad)
        print ('---')

    print('\n'*2)

    prev_idx = 0

    while True:
        time_stamp = time.time()

        img0, img1, idx = dataset[batch_idx]

        data_time = time.time() - time_stamp
        time_stamp = time.time()


        img0 = img0.to(device, non_blocking = True)
        img0_orig = img0[:, :3, :, :].detach().clone()
        img1 = img1.to(device, non_blocking = True)

        current_lr_str = str(f'{optimizer.param_groups[0]["lr"]:.2e}')
        optimizer.zero_grad()

        data_time1 = time.time() - time_stamp
        time_stamp = time.time()

        net.train()

        # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon
        x = img1
        y = img0
        ts = torch.randint(low = 1, high = net.time_steps, size = (args.batch, ))
        gamma = net.alpha_hats[ts].to(device)
        ts = ts.to(device = device)
        img_noise, target_noise = net.add_noise(y, ts)
        y = torch.cat([x, img_noise], dim = 1)
        predicted_noise = net.model(y, gamma)

        loss_l1 = criterion_l1(target_noise, predicted_noise)
        loss = loss_l1

        result = predicted_noise

        min_l1 = min(min_l1, float(loss_l1.item()))
        max_l1 = max(max_l1, float(loss_l1.item()))
        avg_l1 = float(loss_l1.item()) if batch_idx == 0 else (avg_l1 * (batch_idx - 1) + float(loss_l1.item())) / batch_idx 
        avg_loss = float(loss.item()) if batch_idx == 0 else (avg_loss * (batch_idx - 1) + float(loss.item())) / batch_idx

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

        if platform.system() == 'Darwin':
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize(device=device)

        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            try:
                if prev_idx != idx:
                    scheduler.step()
                    prev_idx = idx
            except Exception as e:
                optimizer = torch.optim.AdamW(net.parameters(), lr=lr/100, weight_decay=weight_decay)
                current_lr = float(optimizer.param_groups[0]["lr"])
                patience = 10
                clear_lines(2)
                print (f'switching to ReduceLROnPlateau scheduler with {current_lr} and patience {patience}\n')
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)

        train_time = time.time() - time_stamp
        time_stamp = time.time()

        current_state_dict['step'] = int(step)
        current_state_dict['epoch'] = int(epoch)
        current_state_dict['start_timestamp'] = start_timestamp
        current_state_dict['lr'] = optimizer.param_groups[0]['lr']
        current_state_dict['model_info'] = model_info
        current_state_dict['net_state_dict'] = net.state_dict()
        current_state_dict['optimizer_net_state_dict'] = optimizer.state_dict()
        current_state_dict['trained_model_path'] = trained_model_path

        if step % args.save == 1:
            write_model_state_queue.put(deepcopy(current_state_dict))

        if step % args.preview == 1:
            rgb_source = img0_orig
            rgb_result = img_noise # predicted_noise
            rgb_target = predicted_noise
            
            preview_index += 1
            preview_index = preview_index if preview_index < 10 else 0

            write_image_queue.put(
                {
                    'preview_folder': os.path.join(args.dataset_path, 'preview', os.path.splitext(os.path.basename(trained_model_path))[0]),
                    'preview_index': int(preview_index),
                    'sample_source': rgb_source[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_result': rgb_result[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                    'sample_target': rgb_target[0].clone().cpu().detach().numpy().transpose(1, 2, 0),
                }
            )

        data_time_str = str(f'{data_time:.2f}')
        data_time1_str = str(f'{data_time1:.2f}')
        data_time2_str = str(f'{data_time2:.2f}')
        train_time_str = str(f'{train_time:.2f}')

        epoch_time = time.time() - start_timestamp
        days = int(epoch_time // (24 * 3600))
        hours = int((epoch_time % (24 * 3600)) // 3600)
        minutes = int((epoch_time % 3600) // 60)

        clear_lines(2)
        print (f'\r[Epoch {(epoch + 1):04} Step {step} - {days:02}d {hours:02}:{minutes:02}], Time: {data_time_str}+{data_time1_str}+{train_time_str}+{data_time2_str}, Batch [{batch_idx+1}, Sample: {idx+1} / {len(dataset)}], Lr: {current_lr_str}')
        print(f'\r[Epoch] MinL1: {min_l1:.6f} AvgL1: {avg_l1:.6f}, MaxL1: {max_l1:.6f}, Combined: {avg_loss:.6f}')

        if ( idx + 1 ) == len(dataset):
            write_model_state_queue.put(deepcopy(current_state_dict))

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            rows_to_append = [
                {
                    'Epoch': epoch,
                    'Step': step, 
                    'Min': min_l1,
                    'Avg': avg_l1,
                    'Max': max_l1,
                 }
            ]
            for row in rows_to_append:
                append_row_to_csv(f'{os.path.splitext(trained_model_path)[0]}.csv', row)

            clear_lines(2)
            print (f'Epoch: {epoch+1}, L1: {avg_l1:.6f}, Combined: {avg_loss:.6f} Lr: {current_lr_str}\n\n')

            min_l1 = float(sys.float_info.max)
            max_l1 = 0
            avg_l1 = 0
            epoch = epoch + 1
            batch_idx = 0
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)

            while  ( idx + 1 ) == len(dataset):
                img0, img1, idx = dataset[batch_idx]

            dataset.reshuffle()

        batch_idx = batch_idx + 1
        step = step + 1

        data_time2 = time.time() - time_stamp

        if epoch == args.epochs:
            sys.exit()

if __name__ == "__main__":
    main()

