import torch
import torchvision
import json
from pycocotools import mask as mask_utils
import numpy as np

class Custom_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)"""
    
    def __init__(self, mask_json=True, **kwargs):
        """
        Args:
            mask_json (string, optional): Use json file, else use [img]_mask.pt file.
            **kwargs: Additional parameters to be forwarded to the super class.
        """
        super(Custom_Dataset, self).__init__(**kwargs)
        self.mask_json = mask_json

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        
        if self.mask_json:
            masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
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
        else:
            target = torch.load(f'{path[:-4]}_mask.pt', map_location='cpu').unsqueeze(1) # load masks
        
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
    
    def __init__(self, mask_json=True, **kwargs):
        """
        Args:
            mask_json (string, optional): Use json file, else use [img]_mask.pt file.
            **kwargs: Additional parameters to be forwarded to the super class.
        """
        super(Embedding_Dataset, self).__init__(**kwargs)
        self.mask_json = mask_json
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, masks, embedding).
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        sample = self.loader(path)
        embedding = torch.load(f'{path[:-3]}pt', map_location='cpu') # load embedding
        
        if self.mask_json:
            masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
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
                target = torch.cat(target).unsqueeze(1)
            else:
                target = torch.zeros(1, 1, sample.size[1], sample.size[0])
        else:
            target = torch.load(f'{path[:-4]}_mask.pt', map_location='cpu').unsqueeze(1) # load masks
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, embedding

    def __len__(self):
        return len(self.imgs)

class Cutout_Dataset(torch.utils.data.Dataset):
    def __init__(self, cutouts, transform=None):
        self.cutouts = cutouts
        self.transform = transform

    def __getitem__(self, index):
        sample = self.cutouts[index]
        if self.transform:
            sample = transform(sample)
        return sample

    def __len__(self):
        return len(self.cutouts)

class Cutout_Split_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        target = self.masks[index]
        if self.transform:
            sample = transform(sample)
        return sample, target

    def __len__(self):
        return len(self.images)
