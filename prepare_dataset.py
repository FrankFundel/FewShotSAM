import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import SA1B_Dataset, sample_point
from segment_anything import sam_model_registry
from torchvision import transforms
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.parallel import DataParallel
import tqdm
import random
import numpy as np

# Create a new HDF5 file
output_file = 'test_ds.h5'
h5_file = h5py.File(output_file, 'w')

# Create a new group in the HDF5 file to store the embedded data
ds_name = "EgoHOS"
embedded_group = h5_file.create_group(ds_name)

# Create an embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    sam_model = DataParallel(sam_model)
sam_model.to(device)

# Load the SA1B dataset
root = "FSSAM Datasets/" + ds_name

class PILToNumpy(object):
    def __call__(self, pil_img):
        numpy_img = np.array(pil_img)
        return numpy_img
class PILToTensor(object):
    def __call__(self, numpy_img):
        tensor_img = np.transpose(numpy_img, (2, 0, 1))
        return torch.from_numpy(tensor_img)

sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
transform = transforms.Compose([
    PILToNumpy(),
    sam_transform.apply_image,
    PILToTensor(),
    #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
mask_transform = transforms.Compose([
    sample_point
])
dataset = SA1B_Dataset(root=root, transform=transform, target_transform=mask_transform)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size)

# Store images and masks in memory
embedded_images = []
points = []
masks = []
    
with torch.no_grad():
    for image, (point, mask) in tqdm.tqdm(dataset):
        image = image.to(device)

        # Perform embedding
        if isinstance(sam_model, DataParallel):
            image = sam_model.module.preprocess(image[None, :, :, :])
            embedding = sam_model.module.image_encoder(image)[0]
        else:
            image = sam_model.preprocess(image[None, :, :, :])
            embedding = sam_model.image_encoder(image)[0]
        
        embedded_images.append(embedding.cpu())
        points.append(point)
        masks.append(mask)

# Concatenate and store embeddings in HDF5 file
embedded_images = torch.cat(embedded_images, dim=0)
embedded_group.create_dataset('image_embeddings', data=embedded_images)
embedded_group.create_dataset('points', data=points)
embedded_group.create_dataset('masks', data=masks)

h5_file.close()
