import torch
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os

def sample_point(masks):
    k = random.randint(0, len(masks)-1)
    mask = masks[k]
    _, height, width = mask.shape

    flat_mask = mask.view(-1)
    valid_indices = torch.nonzero(flat_mask).squeeze()
    if len(valid_indices.size()) == 0 or len(valid_indices) < 2:
        return mask, torch.Tensor([0, 0])
    random_index = random.randint(0, len(valid_indices)-1)
    point = [valid_indices[random_index] % width, valid_indices[random_index] // width]
    return mask, torch.Tensor(point)

class PILToNumpy(object):
    def __call__(self, pil_img):
        numpy_img = np.array(pil_img)
        return numpy_img
    
class NumpyToTensor(object):
    def __call__(self, numpy_img):
        tensor_img = np.transpose(numpy_img, (2, 0, 1))
        return torch.from_numpy(tensor_img).float()
    
class SAMPreprocess(object):
    def __init__(self, img_size, normalize=True):
        self.img_size = img_size
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.normalize = normalize
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        if self.normalize:
            x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
class SAMPostprocess(object):
    def __init__(self, img_size):
        self.img_size = img_size
        
    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks

def collate(batch):
    images, targets = zip(*batch)
    masks, points = zip(*targets)
    return torch.stack(images), torch.stack(masks), torch.stack(points)

def embedding_collate(batch):
    images, targets, embeddings = zip(*batch)
    masks, points = zip(*targets)
    return torch.stack(images), torch.stack(masks), torch.stack(points), torch.stack(embeddings)

def is_valid_file(path):
    if path.endswith(('.jpg', '.png')):
        if os.path.exists(f'{path[:-3]}json') and os.path.exists(f'{path[:-3]}pt'):
            return True
    return False