import os
import sys
import argparse
import platform
import torch
import numpy as np
import random
import queue
import threading
import time
import OpenImageIO as oiio

from pprint import pprint

def sample(model, lr_img, index=1, total_files=1, device='cuda'):
    with torch.no_grad():
        # y = torch.randn_like(lr_img, device = device)
        y, _ = model.add_noise(lr_img, [1])
        y = y.to(device)
        lr_img = lr_img.to(device)
        step = 1
        for i, t in enumerate(range(model.time_steps - 1, 0 , -1)):
            print (f'\rProcessing clip {index + 1} of {total_files}, step {step}', end='')

            alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]
    
            t = torch.tensor(t, device = device).long()
            pred_noise = model(torch.cat([lr_img, y], dim = 1), alpha_t_hat.view(-1).to(device))
            y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + 1e-2 * torch.sqrt(beta_t) * noise
            step += 1

    print ('\r'+' '*128, end='')
    return y

def read_image_file(file_path, header_only = False):
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

def write_image_file(file_path, image_data, image_spec):
    out = oiio.ImageOutput.create(file_path)
    if out:
        out.open(file_path, image_spec)
        out.write_image(image_data)
        out.close ()

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

def read_frames(source_exr_files, read_image_queue):
    timeout = 1e-8

    for index, file_path in enumerate(sorted(source_exr_files)):
            try:
                read_data = {}
                read_data['img'] = read_image_file(file_path)
                read_data['file_path'] = file_path
                read_data['index'] = index
                read_image_queue.put(read_data)
            except Exception as e:
                del read_data
                print (e)
                time.sleep(timeout)

    read_image_queue.put(None)

def write_frames(write_image_queue):
    while True:
        try:
            write_data = write_image_queue.get_nowait()
            if write_data is None:
                break
            write_image_file(
                write_data['out_file_path'],
                np.ascontiguousarray(write_data['image_data']),
                write_data['spec'])
        except queue.Empty:
            time.sleep(1e-8)
        except Exception as e:
            print (f'\n{e}\n')


def main():
    parser = argparse.ArgumentParser(description='Inferance script.')
    # Required argument
    parser.add_argument('source_folder', type=str, help='Path to source files')
    # Optional arguments
    parser.add_argument('--state_file', type=str, default=None, help='Path to the pre-trained model state dict file')
    parser.add_argument('--steps', type=int, default=2048, help='Steps of refinement')
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')

    args = parser.parse_args()

    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')


    if args.state_file is None:
        print ('please provide path to model state dict.')
        sys.exit()
    else:
        state_dict_path = os.path.abspath(args.state_file)

    checkpoint = torch.load(state_dict_path, map_location=device, weights_only=False)
    sr3Net = find_and_import_model(model_name=checkpoint['model_info']['name'])
    print ('sr3Net info:')
    pprint (sr3Net.get_info())
    sr3net = sr3Net().get_training_model()(time_steps=args.steps).to(device)
    sr3net.load_state_dict(checkpoint['net_state_dict'])

    source_folder_path = os.path.abspath(args.source_folder)
    result_folder = os.path.join(source_folder_path, 'sr3_result')
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    source_exr_files = [os.path.join(source_folder_path, file) for file in os.listdir(source_folder_path) if file.endswith('.exr')]
    source_exr_files = sorted(source_exr_files)

    total_files = len(source_exr_files)
    print (f'found {total_files} files.')

    read_image_queue = queue.Queue(maxsize=16)
    read_thread = threading.Thread(target=read_frames, args=(source_exr_files, read_image_queue, ))
    read_thread.daemon = True
    read_thread.start()
    
    write_image_queue = queue.Queue(maxsize=32)
    write_thread = threading.Thread(target=write_frames, args=(write_image_queue, ))
    write_thread.daemon = True
    write_thread.start()

    sr3net.eval()

    with torch.no_grad():
        while True:
            read_data = read_image_queue.get()
            if read_data is None:
                break
            index = read_data['index']
            file_path = read_data['file_path']
            img = read_data['img']

            # img = read_image_file(file_path)


            img0 = torch.from_numpy(img['image_data'])
            img0 = img0.to(device, dtype = torch.float32, non_blocking = True).permute(2, 0, 1).unsqueeze(0)
            # img0 = img0[:, :, 448:448+448, 448:448+448]
            _, c, h, w = img0.shape

            '''
            pvalue = 48
            ph = ((h - 1) // pvalue + 1) * pvalue
            pw = ((w - 1) // pvalue + 1) * pvalue
            padding = (0, pw - w, 0, ph -h)
            img0 = torch.nn.functional.pad(img0, (padding), mode='reflect')
            '''

            result = sample(sr3net, img0[:, :, ], index=index, total_files=total_files, device=device) # [:, :, :h, :w]
            result = torch.clamp(result, 0.00001, 0.999990)

            '''
            std_channels = [0.014, 0.0118, 0.025]
            
            noise = torch.empty_like(result)[0]
            for i in range(c):            
                noise[i, :, :] = torch.normal(mean=0., std=std_channels[i], size=(h, w))
            noise = torch.tanh(noise.unsqueeze(0) * 100)

            grain_img = torch.cat((result, noise), 1)
            grain_delta = grainnet(grain_img)
            result = result + grain_delta
            '''

            result = result[0].permute(1, 2, 0).numpy(force=True)

            spec = oiio.ImageSpec(w, h, 3, img['spec'].format)
            spec.extra_attribs = img['spec'].extra_attribs

            out_file_path = os.path.join(
                result_folder,
                os.path.basename(file_path)
            )

            result_item = {}
            result_item['spec'] = spec
            result_item['out_file_path'] = out_file_path
            result_item['image_data'] = result
            write_image_queue.put(result_item)

            # spec.erase_attribute('PixelAspectRatio')
            # spec.erase_attribute('screenWindowCenter')
            # spec.erase_attribute('screenWindowWidth')
            
    write_image_queue.put(None)
    # write_thread,join()
    print ('\n')

if __name__ == "__main__":
    main()
