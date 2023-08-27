import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor
from utils import Path_Dataset, PILToNumpy, NumpyToTensor, SAMPreprocess

import tqdm
import numpy as np
import os
import argparse

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

parser = argparse.ArgumentParser(description='Your script description')

parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
parser.add_argument('--n_max', type=int, default=1000, help='Maximum number of files')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')

args = parser.parse_args()

batch_size = args.batch_size
dataset_path = args.dataset_path
n_max = args.n_max

# Create an embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    sam_model = DataParallel(sam_model)
sam_model.to(device)

if torch.cuda.device_count() > 1:
    img_size = sam_model.module.image_encoder.img_size
else:
    img_size = sam_model.image_encoder.img_size

sam_transform = ResizeLongestSide(img_size)
transform = Compose([
    PILToNumpy(),
    sam_transform.apply_image, # rescale
    NumpyToTensor(),
    SAMPreprocess(img_size) # padding
])

def is_valid_file(path):
    if path.endswith(('.jpg', '.png')):
        if os.path.exists(f'{path[:-3]}json'):
            return True
    return False

dataset = Path_Dataset(root=dataset_path, transform=transform, is_valid_file=is_valid_file)
dataloader = DataLoader(dataset, batch_size=batch_size)

count_n = 0
with torch.no_grad():
    for images, paths in tqdm.tqdm(dataloader):
        batch_already_embedded = True
        for path in paths:
            filename = os.path.splitext(path)[0]
            if not os.path.exists(filename + '.pt'):
                batch_already_embedded = False 

        if batch_already_embedded:
            continue
            
        images = images.to(device)
        if torch.cuda.device_count() > 1:
            embeddings = sam_model.module.image_encoder(images)
        else:
            embeddings = sam_model.image_encoder(images)
        
        for embedding, path in zip(embeddings, paths):
            filename = os.path.splitext(path)[0]
            torch.save(embedding, filename + '.pt')
            count_n += 1
            if count_n >= n_max:
                exit()
    