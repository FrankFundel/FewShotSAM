import torch
import torchvision
import torchvision.transforms as transforms
import json
from PIL import Image
from pycocotools import mask as mask_utils
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class SA1B_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)"""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        target = []

        for m in masks:
            # decode masks from COCO RLE format
            mask = mask_utils.decode(m['segmentation'])
            if len(mask.shape) == 3:
                target.append(torch.as_tensor(mask).permute(2, 0, 1))
            else:
                target.append(torch.as_tensor(mask))
        target = torch.stack(target)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.imgs)

class Path_Dataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.imgs)
    
class Embedding_Dataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, masks, embedding).
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        embedding = torch.load(f'{path[:-3]}pt') # load embedding
        target = []

        for m in masks:
            seg = m['segmentation']
            if isinstance(seg[0], list):  # Polygon format
                mask = mask_utils.decode(mask_utils.frPyObjects(seg, sample.size[1], sample.size[0]))
            elif isinstance(seg[0], dict):  # RLE format
                mask = mask_utils.decode(seg)
            if np.sum(mask) > 0:
                target.append(torch.as_tensor(mask).permute(2, 0, 1))
        
        if len(target) > 0:
            target = torch.concat(target).unsqueeze(1)
        else:
            target = torch.zeros(1, 1, sample.size[1], sample.size[0])
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, embedding

    def __len__(self):
        return len(self.imgs)

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