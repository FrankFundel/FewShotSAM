import os
import json
import torch
import numpy as np
from pycocotools import mask as mask_utils
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def convert_json_to_pt(json_file_path):
    # Load json masks
    masks = json.load(open(json_file_path))['annotations']
    target = []

    for m in masks:
        seg = m['segmentation']
        if isinstance(seg, list):
            seg = seg[0]
        if isinstance(seg, list):  # Polygon format
            mask = mask_utils.decode(mask_utils.frPyObjects([seg], sample.size[1], sample.size[0]))
        elif isinstance(seg, dict):  # RLE format
            mask = mask_utils.decode(seg)
        if np.sum(mask) > 0:
            if len(mask.shape) == 3:
                target.append(torch.as_tensor(mask).permute(2, 0, 1))
            else:
                target.append(torch.as_tensor(mask).unsqueeze(0))

    if len(target) > 0:
        target = torch.cat(target)
    else:
        target = torch.zeros(1, sample.size[1], sample.size[0])

    return target

def process_batch(batch):
    for filename in batch:
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)

            # Convert and save as .pt
            target_tensor = convert_json_to_pt(json_file_path)
            pt_file_path = os.path.join(folder_path, f"{filename[:-5]}_mask.pt")
            torch.save(target_tensor, pt_file_path)

def divide_into_batches(files, batch_size):
    return [files[i:i+batch_size] for i in range(0, len(files), batch_size)]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py /path/to/your/json/files")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Ensure the folder path ends with a slash
    if not folder_path.endswith('/'):
        folder_path += '/'

    # Number of workers (default to the number of CPU cores)
    num_workers = cpu_count()

    # Batch size
    batch_size = 64  # Adjust as needed

    # Create a Pool of workers
    with Pool(num_workers) as pool:
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Divide files into batches
        file_batches = divide_into_batches(files, batch_size)

        # Use the pool to process batches in parallel
        list(tqdm(pool.imap(process_batch, file_batches), total=len(file_batches)))